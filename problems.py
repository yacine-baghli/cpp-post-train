"""Turn open-r1/codeforces rows into plain-text problems shaped like the hackathon example."""
import re

_MACROS = [
    (r"\\leq?(?![a-zA-Z])", "<="), (r"\\geq?(?![a-zA-Z])", ">="), (r"\\neq?(?![a-zA-Z])", "!="),
    (r"\\cdot(?![a-zA-Z])", "*"), (r"\\times(?![a-zA-Z])", "x"), (r"\\[lc]?dots(?![a-zA-Z])", "..."),
    (r"\\(?:rightarrow|to)(?![a-zA-Z])", "->"), (r"\\leftarrow(?![a-zA-Z])", "<-"),
    (r"\\oplus(?![a-zA-Z])", "XOR"), (r"\\b?mod(?![a-zA-Z])", "mod"), (r"\\in(?![a-zA-Z])", "in"),
    (r"\\lfloor(?![a-zA-Z])", "floor("), (r"\\rfloor(?![a-zA-Z])", ")"),
    (r"\\lceil(?![a-zA-Z])", "ceil("), (r"\\rceil(?![a-zA-Z])", ")"),
    (r"\\infty(?![a-zA-Z])", "infinity"), (r"\\(?:left|right|displaystyle|limits)(?![a-zA-Z])", ""),
    (r"\\[,;! ]", " "), (r"\\([%_{}#&$])", r"\1"),
    (r"\\(sum|max|min|gcd|lcm|log|sqrt|and|or|xor)(?![a-zA-Z])", r"\1"),
]
_WRAP = re.compile(r"\\(?:text|texttt|textbf|textit|mathrm|mathit|mathbf|mathtt|operatorname|overline|underline)\{([^{}]*)\}")
_FRAC = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")


def clean_tex(s: str | None) -> str:
    if not s:
        return ""
    s = s.replace("$$$", "")
    for _ in range(3):
        s = _WRAP.sub(r"\1", s)
        s = _FRAC.sub(r"(\1)/(\2)", s)
    for pat, rep in _MACROS:
        s = re.sub(pat, rep, s)
    s = re.sub(r"([_^])\{([A-Za-z0-9])\}", r"\1\2", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def format_problem(row: dict) -> str:
    parts = [row["title"].upper(), "", clean_tex(row["description"]), ""]
    if row.get("input_format"):
        parts += ["Input", clean_tex(row["input_format"]), ""]
    if row.get("output_format"):
        parts += ["Output", clean_tex(row["output_format"]), ""]
    ex = row["examples"]
    for i, e in enumerate(ex):
        tag = f" {i + 1}" if len(ex) > 1 else ""
        parts += [f"Sample input{tag}", e["input"].rstrip(), "", f"Sample output{tag}", e["output"].rstrip(), ""]
    if row.get("note"):
        parts += ["Note", clean_tex(row["note"]), ""]
    return "\n".join(parts).rstrip() + "\n"


def to_problem(row: dict, max_tests: int = 25, max_bytes: int = 1_500_000) -> dict | None:
    """Keep stdin/stdout, non-interactive, exact-match problems with enough usable tests."""
    if row.get("interaction_format") or row.get("input_mode") != "stdio" or row.get("generated_checker"):
        return None
    if not row.get("description") or not row.get("examples") or not row.get("rating"):
        return None
    if "<image>" in row["description"] or "![" in row["description"]:
        return None
    tests, used = [], 0
    for t in row.get("official_tests") or []:
        sz = len(t["input"]) + len(t["output"])
        if not t["output"].strip() or used + sz > max_bytes:
            continue
        tests.append({"input": t["input"], "output": t["output"]})
        used += sz
        if len(tests) >= max_tests:
            break
    if len(tests) < 5:
        return None
    return dict(
        id=row["id"], title=row["title"], rating=row["rating"], tags=list(row.get("tags") or []),
        text=format_problem(row), samples=[{"input": e["input"], "output": e["output"]} for e in row["examples"]],
        tests=tests, time_limit=row.get("time_limit") or 2.0, memory_limit=row.get("memory_limit") or 256,
        complete=bool(row.get("official_tests_complete")),
    )
