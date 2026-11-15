"""Serve the base student (Qwen3.5-9B) with vLLM on Modal GPUs, OpenAI-compatible.

Lets us sample the student for free (Modal credits) instead of spending Tinker credits:
on-policy data generation and sampling-settings sweeps.

  modal deploy serve.py      # prints the endpoint URL; scales to zero when idle
"""
import subprocess

import modal

MODEL = "Qwen/Qwen3.5-9B"

app = modal.App("vals-vllm")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    image=image, gpu="L40S", max_containers=8, scaledown_window=300, timeout=6 * 3600,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
)
@modal.concurrent(max_inputs=48)
@modal.web_server(port=8000, startup_timeout=30 * 60)
def serve():
    cmd = (
        f"vllm serve {MODEL} --served-model-name {MODEL} --host 0.0.0.0 --port 8000 "
        "--max-model-len 65536 --gpu-memory-utilization 0.92 --language-model-only "
        "--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder"
    )
    subprocess.Popen(cmd, shell=True)
