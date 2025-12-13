import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from peft import PeftModel # 명시적 임포트 (에러 방지)

def profile_sdxl_cliff_final():
    # ---------------------------------------------------------
    # 1. Setup: SDXL + LoRA (Memory Heavy)
    # ---------------------------------------------------------
    MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    LORA_ID = "nerijs/pixel-art-xl" 
    
    DEVICE = "cuda"
    GPU_NAME = torch.cuda.get_device_name(0)
    TOTAL_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"🚀 Initializing SDXL + LoRA on {GPU_NAME} ({TOTAL_VRAM_GB:.1f} GB)...")

    # Load VAE (FP16 fix)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

    # Load Pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, 
        vae=vae,
        torch_dtype=torch.float16, 
        use_safetensors=True,
        variant="fp16"
    ).to(DEVICE)

    # Load LoRA
    print("📥 Loading LoRA Adapter...")
    pipe.load_lora_weights(LORA_ID)
    pipe.fuse_lora() # Merge for speed

    # Disable safety checker
    if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
        pipe.safety_checker = None 

    # ---------------------------------------------------------
    # 2. Dense Sampling Strategy
    # ---------------------------------------------------------
    # 초기엔 듬성듬성(Exp), Cliff 예상 구간(10~24)은 촘촘하게(Linear)
    # A100 40GB 기준 SDXL은 보통 Batch 14~18 근처에서 터짐
    batch_sizes = [1, 2, 4, 8] + \
                  list(range(10, 25, 1)) + \
                  [26, 28, 30, 32, 40, 48]
    
    results = []
    print("\nStarting Dense Stress Test...")
    print(f"{'Batch':<5} | {'Throughput':<10} | {'VRAM(GB)':<10} | {'Status'}")
    print("-" * 50)

    oom_batch = None

    try:
        for b in batch_sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            prompts = ["pixel art, cat"] * b
            
            try:
                # Measure
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                # Steps를 15로 줄여서 빠르게 측정 (경향성은 동일함)
                pipe(prompts, num_inference_steps=15) 
                end_event.record()
                
                torch.cuda.synchronize()
                latency = start_event.elapsed_time(end_event) / 1000.0
                
                throughput = b / latency
                max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
                
                print(f"{b:<5} | {throughput:<10.2f} | {max_mem:<10.2f} | OK")
                
                results.append({
                    "batch": b, "throughput": throughput, "vram": max_mem, "status": "Success"
                })

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"{b:<5} | {'0.0':<10} | {TOTAL_VRAM_GB:<10.1f} | 💥 OOM (Cliff)")
                    oom_batch = b
                    break 
                else:
                    print(f"Error: {e}")
                    break
    except KeyboardInterrupt:
        print("\nStopped by user.")

    # ---------------------------------------------------------
    # 3. Analyze & Plot (Theoretical Line)
    # ---------------------------------------------------------
    if not results:
        print("No data collected.")
        return

    df = pd.DataFrame(results)
    df.to_csv("sdxl_cliff_data.csv", index=False)

    # --- Linear Regression for Memory Model ---
    # Model: VRAM = Slope * Batch + Intercept
    # 성공한 데이터들로 추세선을 만듦
    coeffs = np.polyfit(df['batch'], df['vram'], 1)
    slope, intercept = coeffs
    
    # 추세선 생성 (OOM 지점 너머까지 그림)
    max_x = oom_batch + 5 if oom_batch else df['batch'].max() + 5
    x_theory = np.linspace(0, max_x, 100)
    y_theory = slope * x_theory + intercept
    
    print(f"\n📏 Theoretical Model: VRAM ≈ {slope:.2f} * Batch + {intercept:.2f} GB")

    # --- Visualization ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 1. Throughput (Bar/Line)
    color_tp = 'tab:blue'
    ax1.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Throughput (img/s)', color=color_tp, fontsize=12, fontweight='bold')
    ax1.plot(df['batch'], df['throughput'], marker='o', color=color_tp, linewidth=2, label='Measured Throughput')
    ax1.tick_params(axis='y', labelcolor=color_tp)
    ax1.set_ylim(bottom=0)

    # 2. VRAM (Right Axis)
    ax2 = ax1.twinx()
    color_mem = 'tab:purple'
    ax2.set_ylabel('VRAM Usage (GB)', color=color_mem, fontsize=12, fontweight='bold')
    
    # Real Data
    ax2.scatter(df['batch'], df['vram'], color=color_mem, label='Measured VRAM', zorder=3)
    
    # Theoretical Line (Dashed)
    ax2.plot(x_theory, y_theory, color='gray', linestyle='--', alpha=0.7, label=f'Theory: {slope:.2f}x + {intercept:.2f}')
    
    # Hardware Limit Line
    ax2.axhline(y=TOTAL_VRAM_GB, color='red', linestyle='-', linewidth=2, label='Physical Limit (A100)')
    
    # 3. Highlight The Cliff (Intersection)
    if oom_batch:
        # 이론적으로 한계를 넘는 지점 계산
        crossover_batch = (TOTAL_VRAM_GB - intercept) / slope
        
        plt.scatter([crossover_batch], [TOTAL_VRAM_GB], color='red', s=150, marker='X', zorder=5)
        plt.annotate(f'The Memory Cliff\n(OOM @ Batch {oom_batch})', 
                     xy=(crossover_batch, TOTAL_VRAM_GB), 
                     xytext=(crossover_batch + 5, TOTAL_VRAM_GB - 5),
                     arrowprops=dict(facecolor='red', shrink=0.05),
                     fontsize=11, fontweight='bold', color='red')

    # Legend & Style
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(f'Analysis: Memory Scaling & The Cliff (SDXL + LoRA)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save to current directory
    plt.savefig("sdxl_memory_cliff_analysis.png", dpi=300)
    print(f"✅ Analysis graph saved to ./sdxl_memory_cliff_analysis.png")

if __name__ == "__main__":
    profile_sdxl_cliff_final()