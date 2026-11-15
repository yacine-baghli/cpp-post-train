"""End-to-end check of the Tinker account, and a baseline checkpoint to submit.

  uv run python smoke_test.py            # check key, model list, tool-calling sample
  uv run python smoke_test.py --save     # also save an untrained LoRA -> tinker:// path
"""
import sys

import tinker
from openai import OpenAI

from common import BASE_MODEL, EXAMPLE_PROBLEM, OAI_BASE_URL, TOOLS, load_env
import os


def main() -> None:
    load_env()
    service = tinker.ServiceClient(display_name="vals-smoke")

    caps = service.get_server_capabilities()
    names = sorted(m.model_name for m in caps.supported_models)
    print(f"{len(names)} models available; {BASE_MODEL} present: {BASE_MODEL in names}")
    for n in names:
        print("  ", n)

    client = OpenAI(base_url=OAI_BASE_URL, api_key=os.environ["TINKER_API_KEY"])
    reply = client.chat.completions.create(
        model=BASE_MODEL,
        messages=[
            {"role": "system", "content": "Solve the problem in C++20. Test with run_cpp, then submit."},
            {"role": "user", "content": EXAMPLE_PROBLEM},
        ],
        tools=TOOLS,
        max_tokens=8000,
        temperature=0.6,
    )
    msg = reply.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None) or ""
    print("\n--- sample ---")
    print("finish_reason:", reply.choices[0].finish_reason, "| usage:", reply.usage)
    print("reasoning chars:", len(reasoning))
    print("content:", (msg.content or "")[:500])
    for tc in msg.tool_calls or []:
        print("tool_call:", tc.function.name, tc.function.arguments[:800])

    if "--save" in sys.argv:
        training = service.create_lora_training_client(base_model=BASE_MODEL, rank=32)
        ckpt = training.save_weights_for_sampler(name="baseline-untrained").result()
        print("\nBASELINE CHECKPOINT:", ckpt.path)


if __name__ == "__main__":
    main()
