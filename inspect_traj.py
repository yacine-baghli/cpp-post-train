"""Download a trajectory file from the Modal volume and summarise behaviour.

  uv run python inspect_traj.py st-b            # table
  uv run python inspect_traj.py st-b --show 2   # also print 2 multi-submit trajectories
"""
import json
import subprocess
import sys
from pathlib import Path


def fetch(tag: str) -> list:
    Path("data").mkdir(exist_ok=True)
    subprocess.run(["uv", "run", "modal", "volume", "get", "--force", "vals-data", f"traj/{tag}.jsonl", f"data/{tag}.jsonl"],
                   capture_output=True, env={**__import__("os").environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    with open(f"data/{tag}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


from sft import shape  # noqa: E402  (S=submit, r=run printing ALL OK, R=other run)


def main() -> None:
    tag = sys.argv[1]
    rows = fetch(tag)
    usable = [r for r in rows if r["solved"] and r["clean"] and r["exited"]]
    brute_first = [r for r in usable if shape(r).count("S") >= 2]
    print(f"{tag}: n={len(rows)} solved={sum(r['solved'] for r in rows)} usable={len(usable)} "
          f"brute-first usable={len(brute_first)} avg_out_tok={sum(r['completion_tokens'] for r in rows) / max(len(rows), 1):.0f}")
    for r in rows:
        print(f"  {r['id']:>9} {r['rating']} solved={int(r['solved'])} clean={int(r['clean'])} exit={int(r['exited'])} "
              f"turns={r['turns']:>2} tok={r['completion_tokens']:>6} small={r['passed']}/{r['total']} "
              f"large={r.get('large_passed')}/{r.get('large_total')} {shape(r)} {r['error'] or ''}")
    n = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 0
    for r in [x for x in rows if shape(x).count("S") >= 2][:n]:
        print("=" * 100, "\n", r["id"], r["rating"], shape(r))
        for m in r["messages"][1:]:
            if m["role"] == "assistant":
                print("--- THINK:", (m.get("reasoning_content") or "")[:900].replace("\n", " | "))
                print("    SAY:", m["content"][:300].replace("\n", " | "))
                for c in m.get("tool_calls") or []:
                    a = json.loads(c["function"]["arguments"])
                    code = next(iter(a.values()))
                    print(f"    CALL {c['function']['name']} ({len(code)} chars):", code[:500].replace("\n", " | "))
            else:
                print(f"--- {m['role'].upper()}:", m["content"][:400].replace("\n", " | "))


if __name__ == "__main__":
    main()
