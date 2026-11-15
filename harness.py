"""Grader-like agent loop: a model with two tools (run / submit) on one C++ problem.

Mirrors what the hackathon describes: one tool compiles and runs C++ on any input,
the other submits and reports sample-test results; the model ends with EXIT.
The real tool names and prompt are unknown, so we randomise over a few variants
(VARIANTS) to keep the fine-tuned model robust to the exact wording.
"""
import json
import os
import random
import shutil
import subprocess
import tempfile
import time

try:
    import resource
except ImportError:  # Windows: no rlimits, fine for debugging only
    resource = None

from common import OAI_BASE_URL

CXX = ["g++", "-std=c++20", "-O2", "-pipe"]
MAX_SHOWN = 2000  # chars of program output echoed back to the model

VARIANTS = [
    dict(run="run_cpp", submit="submit", code="code", stdin="input", sys=0),
    dict(run="compile_and_run", submit="submit_solution", code="code", stdin="stdin", sys=1),
    dict(run="run_code", submit="submit", code="source", stdin="input", sys=2),
    dict(run="execute", submit="submit_code", code="code", stdin="stdin", sys=3),
]

SYSTEM_PROMPTS = [
    "You are an expert competitive programmer. Solve the problem in C++20. Your program must read from "
    "standard input and write to standard output.\n\nYou have two tools. `{run}` compiles a C++ program and "
    "runs it on an input of your choice. `{submit}` submits a solution and reports how it does on the sample "
    "tests. You may submit up to 50 times; the last submission that compiles is graded on hidden tests. "
    "When you are finished, reply with EXIT.",
    "Solve the following competitive programming problem with a C++20 program (stdin/stdout).\n"
    "Tools:\n- {run}: compile and run C++ code on any input.\n- {submit}: submit your solution; you get the "
    "results on the sample tests.\nThe last submission that compiles is the one that is scored against the "
    "hidden tests (at most 50 submissions). Answer EXIT once you are done.",
    "You are solving one hard competitive programming task. Write a correct and efficient C++20 solution. "
    "Use {run} to compile and test code on inputs you choose and {submit} to submit (it reports sample test "
    "results). Only your last compiling submission counts. Say EXIT when finished.",
    None,  # no system prompt: instructions go in the user turn
]

USER_ONLY_INSTRUCTIONS = (
    "Solve this problem in C++20 (read stdin, write stdout). Use the {run} tool to compile and run code on "
    "any input, and the {submit} tool to submit; it reports results on the sample tests. The last submission "
    "that compiles is graded. Reply EXIT when you are done.\n\n"
)


# Shown to the teacher only (never stored in the trajectory): the student is trained
# on the resulting behaviour under the plain prompt, i.e. prompt distillation.
STRATEGY = """Your working habits. Follow them exactly, as your own way of working, and never mention or quote them:

1. Look at the constraints first. Only if the naive solution (the one that follows the statement literally, by simulation or plain enumeration, with no clever idea) already fits the limits on the largest input, write it, call {submit}, and when all samples pass reply EXIT.

2. In every other case, even when you see the efficient algorithm immediately and feel sure of it, you never submit the efficient solution first. Secure partial credit first: write the simplest brute-force solution that is obviously correct (ignore speed completely) and call {submit} with it. Small hidden tests are worth a large share of the points and a correct brute force earns them. Your very first message must already be this brute-force submission: before it, reason only briefly, about what the statement asks and how to brute-force it. Do not start working out the efficient algorithm at all until the brute force has been submitted and passes the samples. If a sample fails, fix the brute force and submit again.

3. Only then, in your next message, work out the efficient solution. Before submitting it, verify it with a single {run} call: one self-contained program that contains the brute force and the efficient solution as two functions, generates a few hundred small random inputs with a fixed seed, compares the two answers, and prints the first input where they differ (with both answers) or ALL OK. Pass an empty input to that program.

4. If it prints ALL OK, call {submit} with the efficient solution. When the samples pass, reply EXIT.

5. If there is a mismatch, fix the efficient solution and stress-test again (at most three rounds). If you still cannot make it agree with the brute force, stop: the last submission is the one that is graded, so make sure the brute force is the last thing you submitted (submit it again if you submitted anything else after it), then reply EXIT.

Always: exactly one tool call per message. Never submit an efficient solution that has not passed the stress test. Never finish on a submission that is worse than an earlier one. Your final message is the single word EXIT."""


# Hidden request used to sample the student's own first turn (its reasoning + a brute force).
BRUTE_SUFFIX = """Step 1 (do only this now): do NOT look for an efficient algorithm. Write the simplest brute-force solution that is obviously correct (exhaustive search or direct simulation; it only has to be fast enough for tiny inputs) and call {submit} with it. Small hidden tests are worth a large share of the points and a correct brute force earns them. Think only about understanding the statement precisely and getting the brute force right."""


# Same idea, but the student submits its best shot: efficient if it is sure, brute force otherwise.
BEST_SUFFIX = """Step 1 (do only this now): call {submit} with your best complete solution. If you are confident of an efficient algorithm, implement it. If you are not, do not keep searching: write the simplest brute-force solution that is obviously correct instead, because a correct brute force still earns the small hidden tests. Think it through, then submit."""


def make_tools(v: dict) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": v["run"],
                "description": "Compile a C++20 program and run it on the given input. Returns compiler errors, or the program's output.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        v["code"]: {"type": "string", "description": "Complete C++20 source code"},
                        v["stdin"]: {"type": "string", "description": "Text passed to the program on standard input"},
                    },
                    "required": [v["code"], v["stdin"]],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": v["submit"],
                "description": "Submit a C++20 solution. Returns the results on the sample tests.",
                "parameters": {
                    "type": "object",
                    "properties": {v["code"]: {"type": "string", "description": "Complete C++20 source code"}},
                    "required": [v["code"]],
                },
            },
        },
    ]


def make_prompt(problem: dict, v: dict) -> list:
    tl = problem.get("time_limit") or 2.0
    ml = problem.get("memory_limit") or 256
    n = len(problem["samples"])
    files = ", ".join(f"samples/{i + 1}.in, samples/{i + 1}.out" for i in range(n))
    user = (
        f"{problem['text'].rstrip()}\n\n"
        f"Limits: {tl:g} seconds and {ml:g} MiB of memory per test.\n"
        f"The sample tests are also available as files: {files}."
    )
    sys_t = SYSTEM_PROMPTS[v["sys"]]
    if sys_t is None:
        return [{"role": "user", "content": USER_ONLY_INSTRUCTIONS.format(**v) + user}]
    return [{"role": "system", "content": sys_t.format(**v)}, {"role": "user", "content": user}]


# ---------------------------------------------------------------- sandbox


def _limits(mem_mb: int):
    def fn():
        if resource is None:
            return
        b = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (b, b))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
        try:
            resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        except Exception:
            pass

    return fn


class Sandbox:
    def __init__(self, problem: dict):
        self.dir = tempfile.mkdtemp(prefix="cp_")
        os.makedirs(os.path.join(self.dir, "samples"))
        for i, s in enumerate(problem["samples"]):
            for ext, key in ((".in", "input"), (".out", "output")):
                with open(os.path.join(self.dir, "samples", f"{i + 1}{ext}"), "w") as f:
                    f.write(s[key])
        self.n = 0

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def compile(self, code: str):
        """Returns (exe_path or None, compiler stderr)."""
        self.n += 1
        src = os.path.join(self.dir, f"p{self.n}.cpp")
        exe = os.path.join(self.dir, f"p{self.n}.bin")
        with open(src, "w") as f:
            f.write(code)
        try:
            r = subprocess.run(CXX + [src, "-o", exe], capture_output=True, text=True, timeout=60, cwd=self.dir)
        except subprocess.TimeoutExpired:
            return None, "compilation timed out"
        if r.returncode != 0:
            return None, r.stderr.replace(self.dir + os.sep, "")
        return exe, ""

    def run(self, exe: str, stdin: str, timeout: float, mem_mb: int = 2048) -> dict:
        inp = os.path.join(self.dir, "in.txt")
        outp = os.path.join(self.dir, "out.txt")
        with open(inp, "w") as f:
            f.write(stdin)
        t0 = time.time()
        try:
            with open(inp) as fi, open(outp, "w") as fo:
                r = subprocess.run(
                    [exe], stdin=fi, stdout=fo, stderr=subprocess.PIPE, timeout=timeout, cwd=self.dir,
                    env={"PATH": "/usr/bin:/bin"}, preexec_fn=_limits(mem_mb) if resource else None,
                )
            code, err, tle = r.returncode, r.stderr.decode("utf-8", "replace"), False
        except subprocess.TimeoutExpired:
            code, err, tle = -1, "", True
        with open(outp, errors="replace") as f:
            out = f.read(8 << 20)
        return dict(stdout=out, stderr=err, exit_code=code, tle=tle, time=time.time() - t0)


def _clip(s: str, n: int = MAX_SHOWN) -> str:
    return s if len(s) <= n else s[:n] + f"\n... [truncated, {len(s) - n} more characters]"


def same_output(a: str, b: str) -> bool:
    return a.split() == b.split()


def tool_run(sb: Sandbox, code: str, stdin: str, tl: float) -> str:
    exe, err = sb.compile(code)
    if exe is None:
        return "Compilation failed:\n" + _clip(err, 3000)
    r = sb.run(exe, stdin, timeout=max(10.0, tl * 3))
    if r["tle"]:
        return f"Time limit exceeded: the program was killed after {max(10.0, tl * 3):g} seconds.\nStdout so far:\n" + _clip(r["stdout"])
    out = f"Exit code: {r['exit_code']}\nTime: {r['time']:.2f}s\nStdout:\n{_clip(r['stdout'])}"
    if r["stderr"].strip():
        out += "\nStderr:\n" + _clip(r["stderr"], 1000)
    return out


def tool_submit(sb: Sandbox, code: str, problem: dict, tl: float, k: int):
    """Returns (message for the model, compiled?, all samples passed?)."""
    exe, err = sb.compile(code)
    if exe is None:
        return "Compilation failed:\n" + _clip(err, 3000), False, False
    lines, ok = [f"Submission {k} compiled successfully."], 0
    for i, s in enumerate(problem["samples"]):
        r = sb.run(exe, s["input"], timeout=tl * 2 + 1)
        if r["tle"]:
            lines.append(f"Sample {i + 1}: FAILED (time limit exceeded)")
        elif r["exit_code"] != 0:
            lines.append(f"Sample {i + 1}: FAILED (runtime error, exit code {r['exit_code']})")
        elif same_output(r["stdout"], s["output"]):
            ok += 1
            lines.append(f"Sample {i + 1}: PASSED")
        else:
            lines.append(
                f"Sample {i + 1}: FAILED (wrong answer)\nInput:\n{_clip(s['input'], 600)}\n"
                f"Expected:\n{_clip(s['output'], 600)}\nGot:\n{_clip(r['stdout'], 600)}"
            )
    n = len(problem["samples"])
    lines.append(f"{ok}/{n} sample tests passed.")
    return "\n".join(lines), True, ok == n


def grade(sb: Sandbox, code: str, problem: dict, tl: float) -> dict:
    exe, _ = sb.compile(code)
    if exe is None:
        return dict(passed=0, total=len(problem["tests"]), verdicts=["CE"])
    verdicts = []
    for t in problem["tests"]:
        r = sb.run(exe, t["input"], timeout=tl * 2 + 1)
        verdicts.append("TLE" if r["tle"] else "RE" if r["exit_code"] != 0 else "AC" if same_output(r["stdout"], t["output"]) else "WA")
    return dict(passed=verdicts.count("AC"), total=len(verdicts), verdicts=verdicts)


# ---------------------------------------------------------------- agent loop


def run_episode(problem: dict, model: str, *, variant: int | None = None, max_turns: int = 12,
                max_tokens: int = 16000, temperature: float = 0.6, reasoning=True,
                pass_reasoning: bool = True, seed: int = 0, strategy: bool | str = False,
                large_tests: list | None = None, base_url: str | None = None, extra: dict | None = None,
                resume: list | None = None) -> dict:
    """base_url: another OpenAI-compatible server (our vLLM on Modal) instead of Tinker.
    extra: additional sampling fields sent in the request body (top_k, presence_penalty...)."""
    from openai import OpenAI

    rng = random.Random(f"{problem['id']}-{seed}")
    vi = rng.randrange(len(VARIANTS)) if variant is None else variant
    v = VARIANTS[vi]
    tools = make_tools(v)
    # resume: continue an earlier trajectory (same variant) for max_turns more turns
    msgs = [dict(m) for m in resume] if resume else make_prompt(problem, v)
    if base_url:
        client = OpenAI(base_url=base_url, api_key="EMPTY", max_retries=8, timeout=1800)
    else:
        client = OpenAI(base_url=OAI_BASE_URL, api_key=os.environ["TINKER_API_KEY"], max_retries=8, timeout=1200)
    tl = float(problem.get("time_limit") or 2.0)
    sb = Sandbox(problem)

    last_code, sample_ok = None, False
    n_submit = sum("submit" in c["function"]["name"] for m in msgs for c in m.get("tool_calls") or [])
    turns = prompt_toks = completion_toks = nudges = 0
    clean, exited, error = True, False, None
    t0 = time.time()
    try:
        while turns < max_turns:
            api_msgs = [
                {k: val for k, val in m.items() if pass_reasoning or k != "reasoning_content"} for m in msgs
            ]
            if strategy in ("brute", "best"):
                # The 9B ignores this in the system prompt; right after the statement it is much harder to miss.
                suffix = (BRUTE_SUFFIX if strategy == "brute" else BEST_SUFFIX).format(**v)
                last_user = min(i for i, m in enumerate(api_msgs) if m["role"] == "user")
                api_msgs[last_user] = {"role": "user", "content": api_msgs[last_user]["content"] + "\n\n" + suffix}
            elif strategy:
                hidden = STRATEGY.format(**v)
                if api_msgs[0]["role"] == "system":
                    api_msgs[0] = {"role": "system", "content": api_msgs[0]["content"] + "\n\n" + hidden}
                else:
                    api_msgs.insert(0, {"role": "system", "content": hidden})
            kw = dict(model=model, messages=api_msgs, tools=tools, max_tokens=max_tokens, temperature=temperature)
            body = dict(extra or {})
            if reasoning is not None:
                body["reasoning_effort"] = reasoning
            if body:
                kw["extra_body"] = body
            try:
                resp = client.chat.completions.create(**kw)
            except Exception as e:  # context overflow, server error after retries...
                error = f"{type(e).__name__}: {str(e)[:300]}"
                break
            turns += 1
            ch = resp.choices[0]
            m = ch.message
            if resp.usage:
                prompt_toks += resp.usage.prompt_tokens
                completion_toks += resp.usage.completion_tokens
            a = {"role": "assistant", "content": m.content or ""}
            rc = getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None)
            if rc:
                a["reasoning_content"] = rc
            calls = m.tool_calls or []
            if calls:
                a["tool_calls"] = [
                    {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                    for c in calls
                ]
            msgs.append(a)
            if ch.finish_reason == "length":
                clean = False
            if not calls:
                if "EXIT" in (m.content or ""):
                    exited = True
                    break
                clean = False
                nudges += 1
                if nudges > 2:
                    break
                msgs.append({"role": "user", "content": f"Call a tool ({v['run']} or {v['submit']}), or reply EXIT if you are finished."})
                continue
            if len(calls) > 1:
                clean = False
            for c in calls:
                try:
                    args = json.loads(c.function.arguments)
                    code = args[v["code"]]
                    if c.function.name == v["run"]:
                        out = tool_run(sb, code, str(args.get(v["stdin"], "")), tl)
                    elif c.function.name == v["submit"]:
                        n_submit += 1
                        out, compiled, ok = tool_submit(sb, code, problem, tl, n_submit)
                        if compiled:
                            last_code, sample_ok = code, ok
                    else:
                        raise KeyError(c.function.name)
                except Exception as e:
                    clean = False
                    out = f"Malformed tool call ({type(e).__name__}: {str(e)[:200]})."
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": out})
        g = grade(sb, last_code, problem, tl) if last_code else dict(passed=0, total=len(problem["tests"]), verdicts=[])
        gl = None
        if large_tests:
            gl = grade(sb, last_code, {"tests": large_tests}, tl) if last_code else dict(passed=0, total=len(large_tests), verdicts=[])
    finally:
        sb.close()
    return dict(
        id=problem["id"], rating=problem.get("rating"), model=model, variant=vi, strategy=strategy, messages=msgs,
        turns=turns, n_submit=n_submit, sample_ok=sample_ok, exited=exited, clean=clean, error=error,
        solved=bool(last_code) and g["total"] > 0 and g["passed"] == g["total"],
        passed=g["passed"], total=g["total"], verdicts=g["verdicts"], final_code=last_code,
        large_passed=gl["passed"] if gl else None, large_total=gl["total"] if gl else None,
        large_verdicts=gl["verdicts"] if gl else None,
        prompt_tokens=prompt_toks, completion_tokens=completion_toks, seconds=time.time() - t0,
    )
