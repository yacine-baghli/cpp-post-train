"""Shared config: loads .env and defines the model + grader-like tool schema."""
import os
from pathlib import Path

ROOT = Path(__file__).parent
BASE_MODEL = "Qwen/Qwen3.5-9B"
OAI_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY missing: put it in .env (see .env.example)")


# Our best guess at the grader's two tools. The real names/schemas are unknown
# until we see a trace from the hackathon site -- update here when we do.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_cpp",
            "description": "Compile a C++20 program and run it on the given stdin. Returns compiler errors or stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Full C++20 source"},
                    "input": {"type": "string", "description": "stdin for the program"},
                },
                "required": ["code", "input"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit a C++20 solution. Reports results on the sample tests.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Full C++20 source"}},
                "required": ["code"],
            },
        },
    },
]

EXAMPLE_PROBLEM = """DISTINCT STALLS

A market street has N stalls in a row, numbered 1 to N. Stall i sells fruit of kind A_i. You will be asked Q questions. Each question gives two stalls L and R, and asks how many different kinds of fruit are sold by the stalls L, L+1, ..., R.

Input
The first line contains N and Q.
The second line contains A_1 ... A_N.
Each of the next Q lines contains L and R.

Output
For each question print the answer on its own line.

Constraints
1 <= N, Q <= 200000
1 <= A_i <= 10^9
1 <= L <= R <= N

Sample input
5 3
1 2 1 3 2
1 3
2 5
3 3

Sample output
2
3
1
"""
