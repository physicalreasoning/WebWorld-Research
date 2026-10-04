import json
import modal

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

MODEL_NAME = "Qwen/WebWorld-8B"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("webworld-8b-serve")

N_GPU = 1
VLLM_PORT = 8000
MINUTES = 60


@app.server(
    image=vllm_image,
    gpu="A100",
    scaledown_window=15 * MINUTES,
    startup_timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    port=VLLM_PORT,
    max_containers=4,
    target_concurrency=25,
    unauthenticated=True,
)
class Server:
    @modal.enter()
    def start(self):
        import subprocess

        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--tensor-parallel-size",
            str(N_GPU),
            "--max-model-len",
            "65536",
        ]
        print(*cmd)
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.local_entrypoint()
async def test():
    import aiohttp
    import asyncio
    import time

    url = await Server.get_url.aio()

    # Wait for vLLM to be ready
    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Waiting for server at {url} ...")
        deadline = time.time() + 10 * MINUTES
        while time.time() < deadline:
            try:
                async with session.get(
                    "/health", timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        break
                    await asyncio.sleep(1)
            except Exception:
                await asyncio.sleep(1)
        else:
            raise RuntimeError("Server failed to start")

        print("Server is ready. Sending test request...")
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": "Hello, this is a test message."}
            ],
        }
        async with session.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as resp:
            result = await resp.json()
            print("Response:", result["choices"][0]["message"]["content"])
