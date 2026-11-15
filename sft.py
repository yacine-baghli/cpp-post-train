"""Supervised fine-tuning on successful tool-use trajectories.

Rendering uses the base model's own chat template, which we verified matches the
Tinker chat endpoint token-for-token. Each trajectory yields:
  A) one sequence with reasoning kept in history, loss on every assistant turn;
  B) per-turn sequences with reasoning stripped from history (covers a grader that
     does not send reasoning back), loss on the final turn only.
"""
import json
import math
import random
import re
import time

import numpy as np

from common import BASE_MODEL
from harness import VARIANTS, make_tools


def _tmsgs(msgs: list, keep_reasoning: bool) -> list:
    out = []
    for m in msgs:
        if m["role"] != "assistant":
            out.append({k: m[k] for k in ("role", "content", "tool_call_id") if k in m})
            continue
        d = {"role": "assistant", "content": m.get("content") or ""}
        if keep_reasoning and m.get("reasoning_content"):
            d["reasoning_content"] = m["reasoning_content"]
        if m.get("tool_calls"):
            d["tool_calls"] = [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])}}
                for c in m["tool_calls"]
            ]
        out.append(d)
    return out


def _ids(tok, msgs, tools, gen: bool) -> list:
    return tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=gen, tokenize=True, return_dict=False)


def _turn(tok, hist: list, target: dict, tools: list, end_id: int):
    """(prefix ids, target ids) for one assistant turn; None if rendering is inconsistent."""
    prefix = _ids(tok, hist, tools, True)
    full = _ids(tok, hist + [target], tools, False)
    if full[: len(prefix)] == prefix:
        tgt = full[len(prefix):]
    else:
        # Token boundaries can differ right after "<think>\n" (e.g. empty reasoning merges into "\n\n").
        # The model continues from the prompt's tokens, so tokenise the continuation on its own.
        p_txt = tok.apply_chat_template(hist, tools=tools, add_generation_prompt=True, tokenize=False)
        f_txt = tok.apply_chat_template(hist + [target], tools=tools, add_generation_prompt=False, tokenize=False)
        if not f_txt.startswith(p_txt):
            return None
        tgt = tok.encode(f_txt[len(p_txt):], add_special_tokens=False)
    if end_id not in tgt:
        return None
    tgt = tgt[: len(tgt) - tgt[::-1].index(end_id)]  # cut after the last <|im_end|>
    return prefix, tgt


def build_sequences(traj: dict, tok, max_len: int = 24000, b_frac: float = 0.4) -> list:
    """Returns a list of (tokens, weights). Variant B is built for a b_frac share of trajectories."""
    tools = make_tools(VARIANTS[traj["variant"]])
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    think_end = tok.convert_tokens_to_ids("</think>")
    msgs = traj["messages"]
    a_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    full_r = _tmsgs(msgs, True)
    full_n = _tmsgs(msgs, False)
    seqs = []

    def weights(i: int, tgt: list) -> list:
        """With mask_long, long teacher reasoning after turn 1 is context only (no loss):
        the student keeps its own way of thinking and only learns the actions."""
        if msgs[i].get("_skip"):  # e.g. an attempt that failed the samples: context only
            return [0.0] * len(tgt)
        w = [1.0] * len(tgt)
        long_r = len(msgs[i].get("reasoning_content") or "") > 2600
        first_ok = traj.get("mask_all") or i != a_idx[0]
        by_rule = traj.get("mask_long") and first_ok and long_r
        if (by_rule or msgs[i].get("_mask")) and think_end in tgt:
            k = tgt.index(think_end)
            w[: k + 1] = [0.0] * (k + 1)
            if msgs[i].get("_lead"):  # the opening sentence of the turn is learned, the analysis is not
                n = min(k, len(tok.encode(msgs[i]["_lead"], add_special_tokens=False)))
                w[:n] = [1.0] * n
        return w

    # A) reasoning kept everywhere. One sequence when the turns chain token-for-token; when a turn's
    # prompt does not extend the previous tokens (empty reasoning re-tokenises), start a new sequence
    # from that turn's exact prompt so the context always matches what the server would send.
    toks, w, ok = [], [], True
    for i in a_idx:
        r = _turn(tok, full_r[:i], full_r[i], tools, end_id)
        if r is None:
            ok = False
            break
        prefix, tgt = r
        if prefix[: len(toks)] != toks:
            if len(toks) <= max_len and any(w):
                seqs.append((toks, w))
            toks, w = [], []
        w += [0.0] * (len(prefix) - len(toks)) + weights(i, tgt)
        toks = prefix + tgt
    if ok and len(toks) <= max_len and any(w):
        seqs.append((toks, w))

    # B) reasoning stripped from history; first turn is identical to A, so skip it
    if random.Random(traj["id"]).random() >= b_frac:
        return seqs
    for i in a_idx[1:]:
        r = _turn(tok, full_n[:i], full_r[i], tools, end_id)
        if r is None:
            continue
        prefix, tgt = r
        if len(prefix) + len(tgt) <= max_len:
            seqs.append((prefix + tgt, [0.0] * len(prefix) + weights(i, tgt)))
    return seqs


def shape(traj: dict) -> str:
    """Action string: S=submit, r=run whose output says ALL OK, R=any other run."""
    out, msgs = "", traj["messages"]
    for i, m in enumerate(msgs):
        for c in m.get("tool_calls") or []:
            if "submit" in c["function"]["name"]:
                out += "S"
            else:
                res = next((x["content"] for x in msgs[i + 1:] if x.get("tool_call_id") == c["id"]), "")
                out += "r" if "ALL OK" in res else "R"
    return out


# The routine we want cloned: direct submit; or brute force, stress test passes, submit;
# or brute force, stress test keeps failing, stop with the brute force still last.
GOOD_SHAPES = re.compile(r"^(S|S+[Rr]*rS|S+[Rr]*R)$")


_LEAK = re.compile(
    r"instruct|habit|workflow|system prompt|i was told|i am told|i'm told|told to|the rules? say"
    r"|step 1|do only this|user (wants|says|said|asked|asks|is asking)|as requested|the request", re.I)


def scrub(text: str) -> str:
    """Drop lines where the teacher refers to its hidden strategy prompt."""
    if not text or not _LEAK.search(text):
        return text
    return "\n".join(line for line in text.split("\n") if not _LEAK.search(line))


def select(trajs: list, max_turns: int = 6, shapes: bool = False, max_t1_chars: int = 0) -> list:
    """Clean, solved, EXIT-terminated; one per problem, fewest turns then fewest tokens."""
    best = {}
    for t in trajs:
        if not (t.get("solved") and t.get("clean") and t.get("exited") and t["turns"] <= max_turns):
            continue
        if shapes and not GOOD_SHAPES.match(shape(t)):
            continue
        first = next(m for m in t["messages"] if m["role"] == "assistant")
        if max_t1_chars and len(first.get("reasoning_content") or "") > max_t1_chars:
            continue
        key = (t["turns"], t["completion_tokens"])
        if t["id"] not in best or key < best[t["id"]][0]:
            best[t["id"]] = (key, t)
    out = []
    for _, t in best.values():
        t["messages"][-1]["content"] = "EXIT"  # drop any closing explanation: the grader looks for EXIT
        for m in t["messages"]:
            if m["role"] == "assistant":
                m["content"] = scrub(m.get("content") or "")
                if m.get("reasoning_content"):
                    m["reasoning_content"] = scrub(m["reasoning_content"])
        out.append(t)
    return out


def train(trajs: list, name: str, *, lr: float = 1e-4, epochs: float = 2, batch_size: int = 8,
          rank: int = 32, seed: int = 0, save_every: int = 0, b_frac: float = 0.4, log=print) -> list:
    import tinker
    from tinker import types

    service = tinker.ServiceClient(display_name=name)
    tc = service.create_lora_training_client(base_model=BASE_MODEL, rank=rank).result()
    tok = tc.get_tokenizer()

    seqs = []
    for t in trajs:
        seqs += build_sequences(t, tok, b_frac=b_frac)
    n_tok = sum(len(s[0]) for s in seqs)
    n_w = sum(sum(s[1]) for s in seqs)
    log(f"loss on {n_w:,.0f} of {n_tok:,} tokens")
    log(f"{len(trajs)} trajectories -> {len(seqs)} sequences, {n_tok:,} tokens/epoch "
        f"(~${n_tok * epochs * 1.463e-6:.2f} for {epochs} epochs)")

    def datum(toks, w):
        return types.Datum(
            model_input=types.ModelInput.from_ints(toks[:-1]),
            loss_fn_inputs={
                "target_tokens": types.TensorData.from_numpy(np.array(toks[1:], dtype=np.int64)),
                "weights": types.TensorData.from_numpy(np.array(w[1:], dtype=np.float32)),
            },
        )

    rng = random.Random(seed)
    order = []
    for _ in range(math.ceil(epochs)):
        ep = list(range(len(seqs)))
        rng.shuffle(ep)
        order += ep
    order = order[: int(len(seqs) * epochs)]
    steps = math.ceil(len(order) / batch_size)
    paths, t0 = [], time.time()
    for step in range(steps):
        idx = order[step * batch_size: (step + 1) * batch_size]
        cur_lr = lr * max(0.1, 1 - step / steps)
        fb = tc.forward_backward([datum(*seqs[i]) for i in idx], "cross_entropy")
        op = tc.optim_step(types.AdamParams(learning_rate=cur_lr))
        out = fb.result()
        op.result()
        wsum = sum(sum(seqs[i][1][1:]) for i in idx)
        nll = out.metrics.get("loss:sum", float("nan")) / max(wsum, 1)
        log(f"step {step + 1}/{steps} nll={nll:.4f} lr={cur_lr:.2e} t={time.time() - t0:.0f}s")
        if save_every and (step + 1) % save_every == 0 and step + 1 < steps:
            p = tc.save_weights_for_sampler(name=f"{name}-s{step + 1}").result().path
            paths.append(p)
            log(f"CHECKPOINT {p}")
    p = tc.save_weights_for_sampler(name=f"{name}-final").result().path
    paths.append(p)
    log(f"CHECKPOINT {p}")
    return paths
