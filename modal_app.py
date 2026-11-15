"""Modal orchestration: dataset prep, trajectory generation, stats, SFT.

  modal run modal_app.py::prep
  modal run --detach modal_app.py::gen --model Qwen/Qwen3.5-9B --tag base-dev --split dev --n 40
  modal run modal_app.py::stats --tag base-dev
  modal run --detach modal_app.py::train --tags t1,t2 --name v1
"""
import json
import os
from collections import defaultdict

import modal

app = modal.App("vals-cpp")


def _key() -> dict:
    if modal.is_local():
        from common import load_env

        load_env()
        return {"TINKER_API_KEY": os.environ["TINKER_API_KEY"]}
    return {}


image = (
    modal.Image.from_registry("gcc:14", add_python="3.12")
    .run_commands('H=$(find / -name "stdc++.h" -path "*bits*" | head -1) && g++ -std=c++20 -O2 "$H" -o "$H.gch"')
    .pip_install("tinker", "openai", "datasets", "transformers", "jinja2", "numpy")
    .add_local_python_source("common", "harness", "problems", "sft")
)
vol = modal.Volume.from_name("vals-data", create_if_missing=True)
secret = modal.Secret.from_dict(_key())

# $ per million tokens (prefill, sample), from the Tinker pricing page
PRICES = {
    "Qwen/Qwen3.5-9B": (0.30, 0.807),
    "Qwen/Qwen3.8-27B": (0.6975, 5.595),
    "deepseek-ai/DeepSeek-V4.1-Flash:peft:262144": (0.60, 3.60),
    "zai-org/GLM-5.3-Flash:peft:262144": (0.84, 2.11),
    "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16": (0.57, 1.44),
    "Qwen/Qwen3.6-35B-A3B": (0.45, 1.335),
}


def _load(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _dump(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@app.function(image=image, volumes={"/data": vol}, timeout=3600, memory=8192)
def prep(n_train: int = 3000, n_dev: int = 200):
    from datasets import load_dataset

    from problems import to_problem

    for split, name, n in (("test", "dev", n_dev), ("train", "train", n_train)):
        ds = load_dataset("open-r1/codeforces", "verifiable", split=split, streaming=True)
        if split == "train":
            ds = ds.shuffle(seed=0, buffer_size=100)
        out, seen = [], 0
        for row in ds:
            seen += 1
            p = to_problem(row)
            if p:
                out.append(p)
            if len(out) >= n:
                break
        _dump(f"/data/problems/{name}.jsonl", out)
        hist = defaultdict(int)
        for p in out:
            hist[p["rating"] // 400 * 400] += 1
        print(f"{name}: kept {len(out)} of {seen} rows; ratings {dict(sorted(hist.items()))}")
        if out:
            print(out[0]["text"][:1500])
    vol.commit()


def _large_path(pid: str) -> str:
    return f"/data/large/{pid.replace('/', '_')}.json"


@app.function(image=image, volumes={"/data": vol}, timeout=3600, memory=16384)
def prep_large(split: str = "dev", k: int = 6, max_bytes: int = 6_000_000):
    """Fetch the k largest generated tests per problem, to score small (40) vs large (60) like the grader."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    by_contest = defaultdict(list)
    for p in _load(f"/data/problems/{split}.jsonl"):
        by_contest[p["id"].split("/")[0]].append(p["id"])
    os.makedirs("/data/large", exist_ok=True)
    for cid, ids in by_contest.items():
        try:
            path = hf_hub_download("open-r1/codeforces", f"generated_tests/test_cases_{cid}.parquet", repo_type="dataset")
            table = pq.read_table(path, filters=[("problem_id", "in", ids)]).to_pylist()
        except Exception as e:
            print("no generated tests for contest", cid, repr(e)[:150])
            continue
        for pid in ids:
            rows = sorted((r for r in table if r["problem_id"] == pid and r["output"].strip()), key=lambda r: -len(r["input"]))
            pick = [{"input": r["input"], "output": r["output"]} for r in rows if len(r["input"]) + len(r["output"]) <= max_bytes][:k]
            if pick:
                with open(_large_path(pid), "w") as f:
                    json.dump(pick, f)
            print(pid, "generated:", len(rows), "kept input sizes:", [len(x["input"]) for x in pick])
        os.remove(path)
    vol.commit()


@app.function(image=image, secrets=[secret], volumes={"/data": vol}, timeout=3000, cpu=2, memory=4096,
              max_containers=100, retries=1)
def episode(problem: dict, cfg: dict) -> dict:
    from harness import run_episode

    large = None
    if os.path.exists(_large_path(problem["id"])):
        with open(_large_path(problem["id"])) as f:
            large = json.load(f)
    return run_episode(problem, large_tests=large, **cfg)


def _summary(rows: list) -> str:
    if not rows:
        return "no results"
    n = len(rows)
    solved = [r for r in rows if r["solved"]]
    usable = [r for r in solved if r["clean"] and r["exited"]]
    pin, pout = PRICES.get(rows[0]["model"], (0.3, 0.807))
    cost = sum(r["prompt_tokens"] * pin * 0.3 + r["completion_tokens"] * pout for r in rows) / 1e6
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        b = (r.get("rating") or 0) // 400 * 400
        by[b][0] += r["solved"]
        by[b][1] += 1
    scored = [r for r in rows if r.get("large_total")]
    score = ""
    if scored:
        strict = sum(40 * (r["passed"] == r["total"]) + 60 * (r["large_passed"] == r["large_total"]) for r in scored)
        frac = sum(40 * r["passed"] / r["total"] + 60 * r["large_passed"] / r["large_total"] for r in scored)
        score = f"SCORE(strict)={strict / len(scored):.1f} SCORE(frac)={frac / len(scored):.1f} on {len(scored)} | "
    return (
        f"{score}n={n} solved={len(solved)} ({len(solved) / n:.0%}) usable={len(usable)} "
        f"sample_ok={sum(r['sample_ok'] for r in rows)} exited={sum(r['exited'] for r in rows)} "
        f"errors={sum(bool(r['error']) for r in rows)} "
        f"avg_turns={sum(r['turns'] for r in rows) / n:.1f} "
        f"avg_out_tok={sum(r['completion_tokens'] for r in rows) / n:.0f} "
        f"partial={sum(r['passed'] / max(r['total'], 1) for r in rows) / n:.0%} "
        f"est_cost=${cost:.2f}\n  by rating: "
        + " ".join(f"{k}:{v[0]}/{v[1]}" for k, v in sorted(by.items()))
    )


@app.function(image=image, volumes={"/data": vol}, timeout=6 * 3600)
def gen(model: str, tag: str, split: str = "train", n: int = 40, offset: int = 0, rmin: int = 800,
        rmax: int = 3500, max_tokens: int = 16000, temperature: float = 0.6, reasoning: str = "true",
        samples: int = 1, max_turns: int = 12, pass_reasoning: bool = True, strategy: bool = False,
        brute_for: str = "", base_url: str = "", extra: str = "", resume_from: str = "", first_turn: str = ""):
    """brute_for=<tag>: sample only a first turn (brute force) for the problems of that trajectory file,
    with the same tool-name variant, so the two can be spliced together."""
    probs = [p for p in _load(f"/data/problems/{split}.jsonl") if rmin <= p["rating"] <= rmax]
    variants = {}
    if brute_for:
        variants = {t["id"]: t["variant"] for t in _load(f"/data/traj/{brute_for}.jsonl") if t.get("rewritten")}
        probs = [p for p in probs if p["id"] in variants]
        first_turn = first_turn or "brute"
    if first_turn:  # sample only the first turn, with the hidden "brute" or "best" request
        max_turns = 1
        strategy = first_turn
    probs = probs[offset: offset + n]
    r = {"true": True, "false": False, "none": None}[reasoning]
    if resume_from:
        # One more brute-force turn for first turns that did not end in a passing submission
        # (failed samples, compile error, or tested with the run tool first).
        by_id = {p["id"]: p for p in probs}
        inputs = []
        for src in resume_from.split(","):
            for t in _load(f"/data/traj/{src}.jsonl"):
                calls = [c for m in t["messages"] for c in m.get("tool_calls") or []]
                passed = t["sample_ok"] and "submit" in calls[0]["function"]["name"] if calls else False
                if len(calls) == 1 and t["clean"] and not passed and t["id"] in by_id:
                    inputs.append((by_id[t["id"]], dict(
                        model=model, seed=len(inputs), max_tokens=max_tokens, temperature=temperature, reasoning=r,
                        max_turns=1, strategy=first_turn or "brute", variant=t["variant"], resume=t["messages"])))
    else:
        inputs = [
            (p, dict(model=model, seed=s, max_tokens=max_tokens, temperature=temperature, reasoning=r,
                     max_turns=max_turns, pass_reasoning=pass_reasoning,
                     strategy=strategy, variant=variants.get(p["id"]),
                     base_url=base_url or None, extra=json.loads(extra) if extra else None))
            for s in range(samples) for p in probs
        ]
    path = f"/data/traj/{tag}.jsonl"
    rows = _load(path)
    print(f"{tag}: {len(inputs)} episodes with {model}; {len(rows)} already on disk")
    done = 0
    for res in episode.starmap(inputs, order_outputs=False, return_exceptions=True):
        done += 1
        if isinstance(res, Exception):
            print("episode failed:", repr(res)[:300])
            continue
        rows.append(res)
        if done % 10 == 0 or done == len(inputs):
            _dump(path, rows)
            vol.commit()
            print(f"[{done}/{len(inputs)}] {_summary(rows)}")
    _dump(path, rows)
    vol.commit()
    print("DONE", tag, _summary(rows))


@app.function(image=image, volumes={"/data": vol})
def stats(tag: str = "", show: int = 0):
    vol.reload()
    tags = [tag] if tag else sorted(f[:-6] for f in os.listdir("/data/traj"))
    for t in tags:
        rows = _load(f"/data/traj/{t}.jsonl")
        print(f"== {t}: {_summary(rows)}")
        for r in rows[:show]:
            print("-" * 100, "\n", r["id"], "solved", r["solved"], "turns", r["turns"], r["verdicts"], r["error"])
            for m in r["messages"][1:]:
                body = m.get("content") or ""
                if m.get("reasoning_content"):
                    body = f"<think>{m['reasoning_content'][:1500]}</think>\n{body}"
                for c in m.get("tool_calls") or []:
                    body += f"\n[CALL {c['function']['name']}] {c['function']['arguments'][:1500]}"
                print(f"[{m['role']}] {body[:3500]}")


REWRITE_PROMPT = """PROBLEM:
{problem}

BRUTE-FORCE CODE THE CONTESTANT IS ABOUT TO SUBMIT:
{code}

Write the contestant's private thinking right before submitting this brute force: first person, plain text, 80 to 200 words, no headings, no code. Cover, in this order: a quick restatement of what is asked; the observation that the limits are too large for a naive approach to pass the biggest tests, so the plan is to first submit a simple, obviously correct brute force to secure the small tests and only afterwards look for the efficient solution; and how this brute force works. Do not describe or hint at any efficient algorithm. Finish with a short sentence saying you submit the brute force now."""


@app.function(image=image, secrets=[secret], volumes={"/data": vol}, timeout=3600, memory=4096)
def rewrite(tags: str, out: str, max_turns: int = 10):
    """Restructure brute-first trajectories: short brute-force-only reasoning in turn 1,
    and the teacher's original turn-1 analysis moved to turn 2 (after the brute force is banked)."""
    import re
    from concurrent.futures import ThreadPoolExecutor

    from openai import OpenAI

    from common import OAI_BASE_URL
    from sft import select, shape

    trajs = []
    for t in tags.split(","):
        trajs += _load(f"/data/traj/{t}.jsonl")
    sel = select(trajs, max_turns, True)
    client = OpenAI(base_url=OAI_BASE_URL, api_key=os.environ["TINKER_API_KEY"], max_retries=8, timeout=600)
    teacher = "deepseek-ai/DeepSeek-V4.1-Flash:peft:262144"

    def one(t: dict):
        if not re.match(r"^S[Rr]", shape(t)):
            return t  # direct solve, or brute force needed fixing: keep as is
        msgs = t["messages"]
        ai = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        a1, a2 = msgs[ai[0]], msgs[ai[1]]
        code = next(iter(json.loads(a1["tool_calls"][0]["function"]["arguments"]).values()))
        problem = next(m["content"] for m in msgs if m["role"] == "user")
        try:
            r = client.chat.completions.create(
                model=teacher, max_tokens=4000, temperature=0.7,
                messages=[{"role": "user", "content": REWRITE_PROMPT.format(problem=problem, code=code)}],
            )
            short = (r.choices[0].message.content or "").strip()
        except Exception as e:
            print("rewrite failed", t["id"], repr(e)[:200])
            return None
        if not (200 <= len(short) <= 2500) or "```" in short:
            return None
        a2["reasoning_content"] = (
            "The brute force is in. Now the efficient solution.\n\n"
            + (a1.get("reasoning_content") or "").strip() + "\n\n" + (a2.get("reasoning_content") or "").strip()
        ).strip()
        a1["reasoning_content"] = short
        a1["content"] = ""
        t["rewritten"] = True
        return t

    with ThreadPoolExecutor(32) as ex:
        res = [t for t in ex.map(one, sel) if t]
    _dump(f"/data/traj/{out}.jsonl", res)
    vol.commit()
    n_rw = sum(bool(t.get("rewritten")) for t in res)
    print(f"REWRITE {out}: {len(sel)} selected -> {len(res)} kept, {n_rw} restructured")
    ex1 = next(t for t in res if t.get("rewritten"))
    a = [m for m in ex1["messages"] if m["role"] == "assistant"]
    print("EXAMPLE T1:", a[0]["reasoning_content"])
    print("EXAMPLE T2 (head):", a[1]["reasoning_content"][:600])


@app.function(image=image, volumes={"/data": vol}, timeout=1800)
def derive(tags: str = "st-a,st-b,st-c,st-d,st-e,ds-a", out: str = "plain1", keep_tail: int = 6000):
    """Protocol-only data: think, submit the final solution directly, EXIT when the samples pass.
    All long reasoning is kept as loss-free context, so only the actions are learned."""
    import re

    from sft import scrub, shape

    def tail(s: str) -> str:
        s = scrub(s or "").strip()
        if len(s) > keep_tail:
            s = s[-keep_tail:]
            s = s[s.find("\n") + 1:]
        return s

    best = {}
    for tag in tags.split(","):
        for t in _load(f"/data/traj/{tag}.jsonl"):
            if not (t["solved"] and t["exited"]):
                continue
            msgs = t["messages"]
            ai = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
            sh = shape(t)
            subs = [i for i in ai if any("submit" in c["function"]["name"] for c in msgs[i].get("tool_calls") or [])]
            if not subs or len(msgs[subs[-1]]["tool_calls"]) != 1:
                continue
            last = subs[-1]
            result = msgs[last + 1]["content"]
            if "FAILED" in result or "Compilation failed" in result:
                continue
            if re.match(r"^S+$", sh) and t["clean"] and len(sh) >= 2:
                kind, new_msgs = "fix", [dict(m) for m in msgs]  # failed samples first, then fixed: keep whole
            else:
                kind = "direct"
                parts = [msgs[i].get("reasoning_content") or "" for i in ai if i <= last]
                think = "\n\n".join(p for p in parts if len(p) >= 200) or parts[-1]
                a = {"role": "assistant", "content": "", "reasoning_content": tail(think),
                     "tool_calls": msgs[last]["tool_calls"]}
                res = dict(msgs[last + 1])
                res["content"] = re.sub(r"^Submission \d+", "Submission 1", result)
                new_msgs = msgs[: ai[0]] + [a, res, {"role": "assistant", "content": "EXIT"}]
            new_msgs[-1] = {"role": "assistant", "content": "EXIT"}
            new = {k: v for k, v in t.items() if k != "messages"}
            new.update(messages=new_msgs, mask_long=True, mask_all=True, clean=True, kind=kind,
                       turns=sum(m["role"] == "assistant" for m in new_msgs))
            rank = (kind != "fix", t["completion_tokens"])  # prefer fix examples, then shorter ones
            if t["id"] not in best or rank < best[t["id"]][0]:
                best[t["id"]] = (rank, new)
    rows = [v for _, v in best.values()]
    _dump(f"/data/traj/{out}.jsonl", rows)
    vol.commit()
    print(f"DERIVE {out}: {len(rows)} trajectories, {sum(r['kind'] == 'fix' for r in rows)} with a failed first submission")


@app.function(image=image, volumes={"/data": vol}, timeout=1800)
def compose(base_tag: str = "rw1", brute_tag: str = "bt1", out: str = "cmp1", keep_tail: int = 6000):
    """Splice the student's own verified first turn (its reasoning + brute force) onto the teacher's
    later turns (stress test, efficient submission, EXIT). Teacher reasoning becomes loss-free context."""
    import random
    import re

    from sft import scrub

    base = {t["id"]: t for t in _load(f"/data/traj/{base_tag}.jsonl") if t.get("rewritten")}
    direct, fixed, seen = {}, {}, 0
    for tag in brute_tag.split(","):
        for y in _load(f"/data/traj/{tag}.jsonl"):
            seen += 1
            a = [m for m in y["messages"] if m["role"] == "assistant"]
            v = y["verdicts"]
            ok = (
                len(a) in (1, 2) and all(len(m.get("tool_calls") or []) == 1 for m in a) and y["clean"]
                and y["sample_ok"] and "submit" in a[-1]["tool_calls"][0]["function"]["name"]
                and v and v.count("AC") > 0 and not {"WA", "RE", "CE"} & set(v)
                and y["id"] in base and base[y["id"]]["variant"] == y["variant"]
            )
            pool = direct if len(a) == 1 else fixed
            if ok and (y["id"] not in pool or y["completion_tokens"] < pool[y["id"]]["completion_tokens"]):
                pool[y["id"]] = y
    rows = []
    for pid in sorted(set(direct) | set(fixed)):
        # keep a healthy share of "first attempt failed, then fixed" examples
        use_fix = pid in fixed and (pid not in direct or random.Random(pid).random() < 0.4)
        y = fixed[pid] if use_fix else direct[pid]
        x = base[pid]
        xm, ym = x["messages"], y["messages"]
        ai = [i for i, m in enumerate(xm) if m["role"] == "assistant"]
        yi = next(i for i, m in enumerate(ym) if m["role"] == "assistant")
        brute = [dict(m) for m in ym[yi:]]
        for m in brute:
            if m["role"] == "assistant":
                m["reasoning_content"] = scrub(m.get("reasoning_content") or "")
                m["content"] = scrub(m.get("content") or "")
        rest = [dict(m) for m in xm[ai[1]:]]
        r2 = (rest[0].get("reasoning_content") or "").replace("The brute force is in. Now the efficient solution.\n\n", "")
        if len(r2) > keep_tail:
            r2 = r2[-keep_tail:]
            r2 = r2[r2.find("\n") + 1:]
        rest[0]["reasoning_content"] = r2
        rest[0]["_mask"] = True  # teacher analysis: context only, the student keeps its own way of thinking
        for m in rest[1:]:
            if m["role"] == "assistant" and len(m.get("reasoning_content") or "") > 1500:
                m["_mask"] = True
        msgs, k = xm[: ai[0]] + brute + rest, 0
        for i, m in enumerate(msgs):  # renumber "Submission k" after splicing
            if m["role"] == "tool":
                call = next(c for p in msgs[:i] for c in p.get("tool_calls") or [] if c["id"] == m["tool_call_id"])
                if "submit" in call["function"]["name"]:
                    k += 1
                    m["content"] = re.sub(r"^Submission \d+", f"Submission {k}", m["content"])
        new = dict(x)
        new.update(messages=msgs, composed="fix" if use_fix else "direct",
                   turns=sum(m["role"] == "assistant" for m in msgs))
        rows.append(new)
    _dump(f"/data/traj/{out}.jsonl", rows)
    vol.commit()
    n_fix = sum(r["composed"] == "fix" for r in rows)
    print(f"COMPOSE {out}: {seen} student first turns; {len(direct)} problems with a direct pass, {len(fixed)} with a "
          f"fixed one -> {len(rows)} trajectories ({n_fix} fail-then-fix) of {len(base)} problems")
    if rows:
        a = [m for m in rows[0]["messages"] if m["role"] == "assistant"]
        print("EXAMPLE T1 reasoning:", a[0]["reasoning_content"][:1200])


@app.function(image=image, volumes={"/data": vol}, timeout=1800)
def compose3(best_tag: str = "best1", brute_tag: str = "bt2,bt3,btfix", out: str = "cmp4", max_brute: int = 50,
             max_chars: int = 22000):
    """Pure self-distillation: the student's own verified first turn(s), then EXIT.
    best_tag: its best shot (efficient when sure, brute force otherwise); brute_tag: brute-force turns."""
    import random

    from sft import scrub

    def ok(y: dict) -> bool:
        a = [m for m in y["messages"] if m["role"] == "assistant"]
        v = y["verdicts"]
        size = sum(len(m.get("reasoning_content") or "") + len(str(m.get("tool_calls") or "")) for m in a)
        return bool(
            y["clean"] and len(a) in (1, 2) and all(len(m.get("tool_calls") or []) == 1 for m in a)
            and "submit" in a[-1]["tool_calls"][0]["function"]["name"] and y["sample_ok"]
            and v and v.count("AC") > 0 and not {"WA", "RE", "CE"} & set(v) and size <= max_chars
        )

    def finish(y: dict, kind: str) -> dict:
        msgs = [dict(m) for m in y["messages"]]
        for m in msgs:
            if m["role"] == "assistant":
                m["reasoning_content"] = scrub(m.get("reasoning_content") or "")
                m["content"] = scrub(m.get("content") or "")
        msgs.append({"role": "assistant", "content": "EXIT"})
        new = {k: v for k, v in y.items() if k != "messages"}
        new.update(messages=msgs, kind=kind, solved=True, clean=True, exited=True,
                   turns=sum(m["role"] == "assistant" for m in msgs))
        return new

    def pick(tags: str) -> dict:
        best = {}
        for tag in tags.split(","):
            for y in _load(f"/data/traj/{tag}.jsonl"):
                if not ok(y):
                    continue
                n_a = sum(m["role"] == "assistant" for m in y["messages"])
                rank = ("TLE" in y["verdicts"], n_a, y["completion_tokens"])  # complete, direct, short first
                if y["id"] not in best or rank < best[y["id"]][0]:
                    best[y["id"]] = (rank, y)
        return {k: v[1] for k, v in best.items()}

    shots = pick(best_tag)
    brute = {k: v for k, v in pick(brute_tag).items() if k not in shots}
    fixes = [y for y in brute.values() if sum(m["role"] == "assistant" for m in y["messages"]) == 2]
    direct = [y for y in brute.values() if sum(m["role"] == "assistant" for m in y["messages"]) == 1]
    random.Random(0).shuffle(direct)
    rows = [finish(y, "best") for y in shots.values()] + [finish(y, "brute-fix") for y in fixes] \
        + [finish(y, "brute") for y in direct[:max_brute]]
    _dump(f"/data/traj/{out}.jsonl", rows)
    vol.commit()
    full = sum("TLE" not in y["verdicts"] for y in shots.values())
    print(f"COMPOSE3 {out}: {len(shots)} best shots ({full} pass every test in time), {len(fixes)} brute-force fixes, "
          f"{min(len(direct), max_brute)} brute forces -> {len(rows)} trajectories")


REVERT_PROMPT = (
    "A contestant first submitted a correct brute force, then two faster attempts that both failed the sample "
    "tests. Write 12 different short first-person thoughts (one or two sentences each, plain text, one per line, "
    "no numbering, no quotes) for the moment they decide to submit the brute force again so that it is the last "
    "submission, the one that gets graded, and then stop."
)


@app.function(image=image, secrets=[secret], volumes={"/data": vol}, timeout=1800)
def compose2(base_tag: str = "rw1", brute_tag: str = "bt2,bt3", out: str = "cmp2", keep_tail: int = 6000):
    """Simple routine, with the student's own first turn(s):
      A  brute force passes -> efficient submission passes -> EXIT
      B  brute force passes -> an attempt fails the samples -> efficient submission passes -> EXIT
      C  brute force passes -> two attempts fail the samples -> brute force resubmitted -> EXIT
    Teacher reasoning is loss-free context; failed attempts are context only."""
    import random
    import re

    from openai import OpenAI

    from common import OAI_BASE_URL
    from sft import _LEAK, scrub

    base = {t["id"]: t for t in _load(f"/data/traj/{base_tag}.jsonl") if t.get("rewritten")}
    direct, fixed, failed, seen = {}, {}, defaultdict(dict), 0
    for tag in brute_tag.split(","):
        for y in _load(f"/data/traj/{tag}.jsonl"):
            seen += 1
            if y["id"] not in base or base[y["id"]]["variant"] != y["variant"] or not y["clean"]:
                continue
            a = [m for m in y["messages"] if m["role"] == "assistant"]
            if len(a) not in (1, 2) or any(len(m.get("tool_calls") or []) != 1 for m in a):
                continue
            call = a[-1]["tool_calls"][0]["function"]
            v = y["verdicts"]
            if "submit" not in call["name"]:
                continue
            if y["sample_ok"] and v and v.count("AC") > 0 and not {"WA", "RE", "CE"} & set(v):
                pool = direct if len(a) == 1 else fixed
                if y["id"] not in pool or y["completion_tokens"] < pool[y["id"]]["completion_tokens"]:
                    pool[y["id"]] = y
            elif not y["sample_ok"] and len(a) == 1:
                failed[y["id"]][call["arguments"]] = y  # distinct failing attempts, keyed by code

    client = OpenAI(base_url=OAI_BASE_URL, api_key=os.environ["TINKER_API_KEY"], max_retries=8, timeout=600)

    def variants(prompt: str) -> list:
        r = client.chat.completions.create(model="deepseek-ai/DeepSeek-V4.1-Flash:peft:262144", max_tokens=4000,
                                           temperature=0.8, messages=[{"role": "user", "content": prompt}])
        out = [s.strip(" -*\"") for s in (r.choices[0].message.content or "").split("\n") if len(s.strip()) > 30]
        out = [s for s in out if not _LEAK.search(s)]
        assert len(out) >= 5, out
        return out

    thoughts = variants(REVERT_PROMPT)
    # Short teacher-written openers that ARE trained, so the student knows to keep going after the
    # brute force passes (or after a failed attempt) instead of stopping; the analysis itself stays its own.
    go_on = variants(
        "A contestant has just submitted a simple brute force and it passed the sample tests, which secures the "
        "small tests. Write 12 different short first-person thoughts (one or two sentences each, plain text, one "
        "per line, no numbering, no quotes) for the moment they turn to finding an efficient solution that will "
        "also pass the large tests.")
    retry = variants(
        "A contestant's brute force is already submitted and correct. Their faster solution has just failed the "
        "sample tests. Write 12 different short first-person thoughts (one or two sentences each, plain text, one "
        "per line, no numbering, no quotes) for the moment they decide to find the mistake and try the faster "
        "solution once more.")
    print("openers:", go_on[:2], retry[:2], thoughts[:2])

    def turn_of(y: dict, mask_all: bool) -> list:
        ms = y["messages"]
        i = next(j for j, m in enumerate(ms) if m["role"] == "assistant")
        a = {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": ms[i]["tool_calls"]}
        if mask_all:
            a["_skip"] = True
        return [a, dict(ms[i + 1])]

    rows, kinds = [], defaultdict(int)
    for pid in sorted(set(direct) | set(fixed)):
        rng = random.Random(pid)
        use_fix = pid in fixed and (pid not in direct or rng.random() < 0.4)
        y, x = (fixed[pid] if use_fix else direct[pid]), base[pid]
        xm, ym = x["messages"], y["messages"]
        ai = [i for i, m in enumerate(xm) if m["role"] == "assistant"]
        yi = next(i for i, m in enumerate(ym) if m["role"] == "assistant")
        brute = [dict(m) for m in ym[yi:]]
        for m in brute:
            if m["role"] == "assistant":
                m["reasoning_content"] = scrub(m.get("reasoning_content") or "")
                m["content"] = scrub(m.get("content") or "")
        # teacher's efficient submission, with its analysis as loss-free context
        last = max(i for i in ai if any("submit" in c["function"]["name"] for c in xm[i].get("tool_calls") or []))
        r2 = (xm[ai[1]].get("reasoning_content") or "").replace("The brute force is in. Now the efficient solution.\n\n", "")
        if len(r2) > keep_tail:
            r2 = r2[-keep_tail:]
            r2 = r2[r2.find("\n") + 1:]
        fails = list(failed.get(pid, {}).values())
        rng.shuffle(fails)
        p = rng.random()
        is_b = not (len(fails) >= 2 and p < 0.5) and len(fails) >= 1 and p < 0.85
        lead = rng.choice(retry if is_b else go_on)
        eff = [{"role": "assistant", "content": "", "reasoning_content": lead + "\n\n" + r2,
                "tool_calls": xm[last]["tool_calls"], "_mask": True, "_lead": lead}, dict(xm[last + 1])]
        if len(fails) >= 2 and p < 0.5:
            kind = "C"
            b_call = next(m for m in reversed(brute) if m["role"] == "assistant")
            b_res = brute[-1]
            revert = [{"role": "assistant", "content": "", "reasoning_content": rng.choice(thoughts),
                       "tool_calls": b_call["tool_calls"]}, dict(b_res)]
            mid = turn_of(fails[0], True) + turn_of(fails[1], True) + revert
        elif len(fails) >= 1 and p < 0.85:
            kind, mid = "B", turn_of(fails[0], True) + eff
        else:
            kind, mid = "A", eff
        msgs = [dict(m) for m in xm[: ai[0]] + brute + mid] + [{"role": "assistant", "content": "EXIT"}]
        n_call = n_sub = 0
        for i, m in enumerate(msgs):  # fresh call ids, and renumber "Submission k"
            if m.get("tool_calls"):
                n_call += 1
                c = m["tool_calls"][0]
                m["tool_calls"] = [{"id": f"call_{n_call}", "type": "function", "function": dict(c["function"])}]
                is_sub = "submit" in c["function"]["name"]
                res = msgs[i + 1] = dict(msgs[i + 1])
                res["tool_call_id"] = f"call_{n_call}"
                if is_sub and res["content"].startswith("Submission"):
                    n_sub += 1
                    res["content"] = re.sub(r"^Submission \d+", f"Submission {n_sub}", res["content"])
                elif is_sub:
                    n_sub += 1
        new = {k: v for k, v in x.items() if k != "messages"}
        new.update(messages=msgs, composed=kind, brute="fix" if use_fix else "direct", solved=True, clean=True,
                   exited=True, turns=sum(m["role"] == "assistant" for m in msgs))
        kinds[kind + ("/fix" if use_fix else "")] += 1
        rows.append(new)
    _dump(f"/data/traj/{out}.jsonl", rows)
    vol.commit()
    print(f"COMPOSE2 {out}: {seen} student turns; direct pass for {len(direct)} problems, fixed for {len(fixed)}, "
          f"failing attempts for {len(failed)} -> {len(rows)} trajectories {dict(kinds)} of {len(base)} problems")


@app.function(image=image, secrets=[secret], volumes={"/data": vol}, timeout=6 * 3600, memory=8192)
def train(tags: str, name: str, lr: float = 1e-4, epochs: float = 2.0, batch_size: int = 8, rank: int = 32,
          max_turns: int = 10, save_every: int = 0, shapes: bool = True, max_t1_chars: int = 0,
          b_frac: float = 0.4):
    from collections import Counter

    from sft import select, shape
    from sft import train as run

    trajs = []
    for t in tags.split(","):
        trajs += _load(f"/data/traj/{t}.jsonl")
    sel = select(trajs, max_turns, shapes, max_t1_chars)
    print(f"{len(trajs)} trajectories loaded, {len(sel)} selected; shapes {Counter(shape(t) for t in sel).most_common(12)}")
    paths = run(sel, name, lr=lr, epochs=epochs, batch_size=batch_size, rank=rank, save_every=save_every,
                b_frac=b_frac)
    _dump(f"/data/ckpt/{name}.jsonl", [{"name": name, "paths": paths, "tags": tags, "n": len(sel)}])
    vol.commit()
    print("PATHS", paths)
