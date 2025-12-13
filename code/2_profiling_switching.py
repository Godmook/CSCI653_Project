# profiling/2_profile_switching.py
import torch
import time
import pandas as pd
import os

from diffusers import AutoPipelineForText2Image

def measure_once(fn):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.time()
    fn()
    torch.cuda.synchronize()
    return time.time() - start


def profile_switching():
    DATA_DIR = "../data"
    os.makedirs(DATA_DIR, exist_ok=True)
    
    MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    LORA_A = "nerijs/pixel-art-xl"
    LORA_B = "minimaxir/sdxl-wrong-lora"
    DEVICE = "cuda"

    print("🚀 Initializing AutoPipelineForText2Image...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(DEVICE)

    measurements = []

    print("\n⚡ Measuring Load Latency... (50 trials recommended)")
    
    def load_A():
        pipe.load_lora_weights(LORA_A)

    def load_B():
        pipe.load_lora_weights(LORA_B)

    # 50번 반복
    for i in range(50):
        t = measure_once(load_A)
        measurements.append({
            "operation": "load_A",
            "iteration": i,
            "time": t
        })
        pipe.unload_lora_weights()

    print("\n⚡ Measuring unload latency...")
    def unload():
        pipe.unload_lora_weights()

    for i in range(20):
        pipe.load_lora_weights(LORA_A)
        t = measure_once(unload)
        measurements.append({
            "operation": "unload",
            "iteration": i,
            "time": t
        })

    # inference baseline
    pipe.load_lora_weights(LORA_A)
    pipe.fuse_lora()

    print("\n⚡ Measuring inference(1 step)...")
    def infer_1():
        pipe("test", num_inference_steps=1)

    t = measure_once(infer_1)
    measurements.append({
        "operation": "inference_1step",
        "iteration": 0,
        "time": t
    })

    df = pd.DataFrame(measurements)
    df.to_csv(f"{DATA_DIR}/switching_cost.csv", index=False)
    print(f"✅ Saved to {DATA_DIR}/switching_cost.csv")


if __name__ == "__main__":
    profile_switching()
