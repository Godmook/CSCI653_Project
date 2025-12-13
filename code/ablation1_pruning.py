import os
import time
import math
import random
import numpy as np
import pandas as pd
from collections import deque
from typing import List, Dict
from tqdm import tqdm

# ============================================
# 0. SDXL + LPIPS Profiling (Real Data on A100)
# ============================================

def run_quality_profiling():
    """
    SDXL로 Simple / Complex 프롬프트를 다양한 step으로 돌려보고
    LPIPS 기준 품질 저하 곡선을 CSV로 저장하는 함수.
    - 이미지: ../assets/quality_study_60_60_60/
    - CSV:    ../data/sdxl_step_quality_lpips_60_60_60.csv
    """
    import torch
    from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DDIMScheduler
    from PIL import Image
    import torchvision.transforms as T
    import lpips
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    IMG_DIR = "../assets/quality_study_60_60_60"
    CSV_PATH = "../data/sdxl_step_quality_lpips_60_60_60.csv"
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    device = "cuda"

    print("\n🧠 [Profiling] Loading SDXL + fixed VAE on A100...")
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16,
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    print("🧮 [Profiling] Loading LPIPS (VGG backbone)...")
    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    lpips_fn.eval()

    def pil_to_lpips_tensor(img: Image.Image):
        transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2.0 - 1.0),
        ])
        return transform(img).unsqueeze(0).to(device)

    def compute_lpips(img_ref: Image.Image, img: Image.Image) -> float:
        with torch.no_grad():
            t0 = pil_to_lpips_tensor(img_ref)
            t1 = pil_to_lpips_tensor(img)
            dist = lpips_fn(t0, t1)
        return float(dist.item())

    prompts = {
        # --- Group A: Simple / Minimalist ---
        "Simple_Apple": "minimalist flat vector art of a single red apple on a white background, simple shapes, no texture",
        "Simple_Cat": "a close-up photo of a white cat sitting on a clean white background, studio shot, soft lighting",
        "Simple_Logo": "a simple black and white logo design, geometric triangle shape, vector graphics, white background",
        "Simple_Icon": "flat design app icon, weather sun symbol, yellow and orange, minimal",
        "Simple_Sketch": "a quick charcoal sketch of a circle, rough lines, white paper texture",
        
        # --- Group B: Portraits / Characters ---
        "Portrait_Woman": "extreme close-up portrait of a young woman with freckles, visible skin pores, peach fuzz, natural lighting, heterochromia eyes, stray hairs, bokeh background, 85mm lens, f/1.8, raw photo",
        "Portrait_Man": "rugged portrait of an old fisherman with a thick beard, deep wrinkles, weathered skin, dramatic chiaroscuro lighting, rembrandt style, intense gaze, black background, highly detailed texture",
        "Anime_Girl": "anime style illustration of a magical girl casting a spell, glowing magic circle, floating crystals, intricate frilled dress, rainbow colored hair with gradients, dynamic pose, wind effects, cel shaded, 4k",
        "Celebrity_Mix": "hyper-realistic fusion face of a pop star and a movie actress, blonde wavy hair, diamond necklace, red carpet event, camera flashes in background, detailed makeup, skin texture, award ceremony atmosphere",
        "Fantasy_Warrior": "full body shot of a paladin in ornate gold and silver armor, engraved patterns, holding a glowing runeblade, standing in a ruined temple, god rays, dust particles, cinematic composition, fantasy art",

        # --- Group C: Landscapes / Architecture ---
        "Landscape_Mountain": "majestic snow-capped himalayan peaks, reflecting in a turquoise glacial lake, jagged rocks in foreground, alpine wildflowers, dramatic sunset clouds, volumetric lighting, tyndall effect, landscape photography",
        "Arch_Modern": "futuristic parametric skyscraper designed by zaha hadid, organic fluid shapes, glass and steel facade reflecting the sky, aerial view, surrounding park, hyper-realistic rendering, unreal engine 5, ray tracing",
        "Arch_Gothic": "interior of a massive gothic cathedral, flying buttresses, intricate stained glass rose window, thousands of burning candles, incense smoke, high vaulted ceiling, detailed stone carvings, mysterious atmosphere",
        "Cyberpunk_City": "dense futuristic tokyo street at night, neon signs in kanji and katakana, heavy rain, wet pavement reflections, flying cars with trails, steam from vents, crowded sidewalks with cyborgs, cinematic lighting",
        "Nature_Forest": "deep ancient rainforest floor, moss-covered roots, giant ferns, exotic orchids, sunlight filtering through dense canopy, morning mist, water droplets on leaves, macro details, 8k nature documentary",

        # --- Group D: Nightmare / Chaos ---
        "Chaos_Battle": (
            "wide angle shot of an epic chaotic battlefield, thousands of soldiers clashing, "
            "dragons breathing fire in the sky, intricate golden armor details, "
            "hyper-realistic smoke simulations, volumetric lightning storms, "
            "foreground showing a shattered sword with rune engravings, "
            "background showing a burning castle, 8k resolution, ray tracing, "
            "cinematic lighting, masterpiece, best quality, ultradetailed"
        ),
        "Chaos_Fractal": "infinite 3d fractal math pattern, mandelbrot bulb, iridescent bismuth crystal structure, recursive geometry, psychedelic colors, mesmerizing details, mathematical art, 8k resolution",
        "Texture_Fabric": "extreme macro shot of royal brocade fabric, interwoven gold and silver threads, raised embroidery, velvet texture, microscopic fiber details, sharp focus, depth of field, luxury textile",
        "SciFi_Space": "gigantic space station exploding in orbit, thousands of debris fragments, laser beams crossing, nebula background with stars, lens flare, cinematic composition, star wars style space battle, hyper-realistic CGI",
        "Crowd_Concert": "massive stadium rock concert, view from stage, sea of thousands of fans, individual faces visible, laser light show, confetti rain, smoke machines, electric atmosphere, wide angle fisheye lens, 8k",
    }
    
    SEED = 42
    steps_list = [60, 50, 40, 30, 20, 10, 5]

    rows = []

    print("\n🚀 [Profiling] Generating baseline + variants for LPIPS curve...")

    for ptype, ptext in prompts.items():
        print(f"\n=== [{ptype}] prompt profiling ===")

        gen = torch.Generator(device).manual_seed(SEED)
        with torch.amp.autocast('cuda'):
            baseline_img = pipe(
                ptext,
                num_inference_steps=60,
                generator=gen,
            ).images[0]

        baseline_path = os.path.join(IMG_DIR, f"{ptype}_baseline_step60.png")
        baseline_img.save(baseline_path)
        print(f"  -> Baseline saved: {baseline_path}")

        for steps in steps_list:
            gen = torch.Generator(device).manual_seed(SEED)
            start_t = time.time()
            with torch.amp.autocast('cuda'):
                img = pipe(
                    ptext,
                    num_inference_steps=steps,
                    generator=gen,
                ).images[0]
            elapsed = time.time() - start_t

            fname = f"{ptype}_step{steps}.png"
            save_path = os.path.join(IMG_DIR, fname)
            img.save(save_path)

            lp = compute_lpips(baseline_img, img)
            print(f"  [step={steps:2d}] LPIPS={lp:.4f}  time={elapsed:.2f}s  -> {fname}")

            rows.append({
                "prompt_type": ptype,
                "prompt_text": ptext,
                "steps": int(steps),
                "lpips": lp,
                "gen_time_sec": elapsed,
                "image_path": save_path,
            })

    df_profile = pd.DataFrame(rows)
    df_profile.to_csv(CSV_PATH, index=False)
    print(f"\n✅ [Profiling] LPIPS profiling saved to: {CSV_PATH}")
    print(f"✅ [Profiling] Images saved under: {IMG_DIR}")


# ============================================================
# ⚙️ 1. Global System Config (A100 기반)
# ============================================================

TOTAL_VRAM = 40.0
TOTAL_COMPUTE_CAPACITY = 22 * 50.0

CLIFF_THRESHOLD = 22
SAFE_MARGIN = 18

# 기본값 (switching_cost.csv로 덮어씀)
SWITCH_COST_BASE = 1.0   # "기준 LoRA" 스위칭 시간 (초)
STEP_TIME = 0.06         # 1 step 추론 시간 (초, switching_cost로 보정)

FULL_STEPS = 60
PRUNED_STEPS = 20

NUM_REQUESTS = 1000

# --- LoRA 파일 크기 (MB) : rank proxy ---
LORA_REAL_SIZES = {
    "lora_minimal":  18,   # Rank 8 (~18MB)
    "lora_portrait": 36,   # Rank 16
    "lora_anime":    72,   # Rank 32
    "lora_scenery":  144,  # Rank 64
    "lora_fantasy":  144,  # Rank 64
    "lora_arch":     288,  # Rank 128
    "lora_scifi":    288,  # Rank 128
    "lora_detail":   576,  # Rank 256
    "lora_ultra":    1152, # Rank 512
}

LORA_IO_COSTS: Dict[str, float] = {}  # 각 LoRA별 스위칭 시간(초)


# ============================================================
# 1-2. switching_cost.csv 읽어서 보정
# ============================================================

def load_switch_profile(csv_path: str = "../data/switching_cost.csv"):
    """
    switching_cost.csv에서:
    - load_A 의 median -> 기본 LoRA 스위칭 시간 (초)
    - inference_1step -> 1 step 추론 시간 (초)
    둘을 읽어 전역 SWITCH_COST_BASE, STEP_TIME을 세팅하는 용도.
    """
    global SWITCH_COST_BASE, STEP_TIME

    if not os.path.exists(csv_path):
        print(f"\n⚠️ [Switching] {csv_path} not found. Using default SWITCH_COST_BASE={SWITCH_COST_BASE}, STEP_TIME={STEP_TIME}")
        return

    df = pd.read_csv(csv_path)

    load_mask = df["operation"].str.contains("load", case=False, na=False)
    infer_mask = df["operation"].str.contains("inference", case=False, na=False)

    if load_mask.any():
        base_switch = float(np.median(df.loc[load_mask, "time"].values))
        SWITCH_COST_BASE = base_switch
    if infer_mask.any():
        step_t = float(np.median(df.loc[infer_mask, "time"].values))
        STEP_TIME = step_t

    print(f"\n🧪 [Switching] Loaded from {csv_path}")
    print(f"   - SWITCH_COST_BASE (median load_A)   = {SWITCH_COST_BASE:.4f} s")
    print(f"   - STEP_TIME (inference_1step)        = {STEP_TIME:.4f} s")


def build_lora_io_costs():
    """
    실제 rank는 모르지만,
    - 스위칭 시간은 LoRA 파라미터 크기에 비례한다고 가정
    - switching_cost.csv에서 얻은 SWITCH_COST_BASE 를
      '기준 크기(base_size_mb)'의 LoRA에 대한 시간으로 보고,
      다른 LoRA는 크기 비율만큼 선형 스케일링
    """
    base_size_mb = 36.0  # 기준: rank 16 LoRA ~36MB (lora_portrait)
    for lid, size_mb in LORA_REAL_SIZES.items():
        factor = size_mb / base_size_mb
        LORA_IO_COSTS[lid] = SWITCH_COST_BASE * factor

    print("\n📏 [Switching] LoRA I/O cost per profile (scaled by size):")
    for lid, t in LORA_IO_COSTS.items():
        print(f"   - {lid:<12}: {t:.4f} s")


# --- Prompt Prefix -> (LoRA ID, Complexity) 매핑 ---
PROMPT_TO_LORA_MAP = {
    "Simple":    ("lora_minimal",  0.2),
    "Portrait":  ("lora_portrait", 0.5),
    "Anime":     ("lora_anime",    0.6),
    "Celebrity": ("lora_portrait", 0.7),
    "Fantasy":   ("lora_fantasy",  0.8),
    "Landscape": ("lora_scenery",  0.6),
    "Arch":      ("lora_arch",     0.7),
    "Cyberpunk": ("lora_scifi",    1.0),
    "Nature":    ("lora_scenery",  0.6),
    "Chaos":     ("lora_ultra",    1.0),
    "Texture":   ("lora_detail",   0.9),
    "SciFi":     ("lora_scifi",    1.0),
    "Crowd":     ("lora_detail",   1.0),
}


# ============================================================
# 🧩 2. Data Structures & Helper Functions
# ============================================================

class Request:
    def __init__(self, req_id, complexity, lora_id, switch_cost, arrival_time):
        self.id = req_id
        self.complexity = float(complexity)
        self.lora_id = lora_id
        # "rank proxy": LoRA 교체에 드는 실제 시간 (초)
        self.switch_cost = float(switch_cost)
        self.arrival_time = float(arrival_time)
        
        self.is_prunable = (self.complexity < 0.3)
        self.start_time = None
        self.finish_time = None
        self.executed_steps = 0

def get_normalized_vector(vram_gb, steps):
    return np.array([
        vram_gb / TOTAL_VRAM,
        steps / TOTAL_COMPUTE_CAPACITY
    ])

def get_demand_vector(req: Request, apply_pruning: bool) -> np.ndarray:
    vram = 1.5
    if apply_pruning and req.is_prunable:
        steps = PRUNED_STEPS
    else:
        steps = FULL_STEPS
    return np.array([vram, float(steps)])


# ============================================================
# 🧠 3. Scheduler Algorithms
# ============================================================

def sched_tetris(queue: List[Request], current_lora, current_time):
    if not queue: return []
    
    residual_vram = TOTAL_VRAM
    residual_compute = TOTAL_COMPUTE_CAPACITY
    
    batch = []
    temp_queue = list(queue)
    
    while len(batch) < CLIFF_THRESHOLD and temp_queue:
        A = get_normalized_vector(residual_vram, residual_compute)
        if A[0] <= 1e-6 or A[1] <= 1e-6:
            break
        
        best_req = None
        best_score = -1.0
        best_idx = -1
        
        for i, req in enumerate(temp_queue):
            d_raw = get_demand_vector(req, apply_pruning=False)
            r = get_normalized_vector(d_raw[0], d_raw[1])
            
            dot_val = np.dot(r, A)
            norm_r = np.linalg.norm(r)
            norm_A = np.linalg.norm(A)
            
            if norm_r == 0 or norm_A == 0:
                score = 0.0
            else:
                score = dot_val / (norm_r * norm_A)
            
            if score > best_score:
                best_score = score
                best_req = req
                best_idx = i
        
        if best_req is None:
            break
        
        d_best = get_demand_vector(best_req, apply_pruning=False)
        if residual_vram - d_best[0] < 0:
            break
        
        batch.append(best_req)
        residual_vram -= d_best[0]
        residual_compute -= d_best[1]
        temp_queue.pop(best_idx)
        
    return batch


def sched_drf(queue: List[Request], lora_usage_history: Dict[str, List[float]], current_time):
    if not queue: return []
    if lora_usage_history is None: 
        lora_usage_history = {}
    
    lora_shares = {}
    unique_loras = {req.lora_id for req in queue}
    
    for lid in unique_loras:
        usage = lora_usage_history.get(lid, [0.0, 0.0])
        share_vram = usage[0] / (TOTAL_VRAM * 100)
        share_compute = usage[1] / (TOTAL_COMPUTE_CAPACITY * 100)
        lora_shares[lid] = max(share_vram, share_compute)
    
    sorted_queue = sorted(queue, key=lambda r: lora_shares.get(r.lora_id, 0.0))
    
    batch = []
    curr_vram = TOTAL_VRAM
    
    for req in sorted_queue:
        if len(batch) >= CLIFF_THRESHOLD:
            break
        
        d = get_demand_vector(req, apply_pruning=False)
        
        if curr_vram - d[0] < 0:
            break
        
        batch.append(req)
        curr_vram -= d[0]
        
    return batch


def sched_clockwork(queue: List[Request], current_lora, current_time):
    if not queue: return []
    
    def get_score(req):
        is_cached = (req.lora_id == current_lora)
        return (1 if is_cached else 0, -req.arrival_time)
    
    sorted_queue = sorted(queue, key=get_score, reverse=True)
    
    batch = []
    curr_vram = TOTAL_VRAM
    
    for req in sorted_queue:
        if len(batch) >= CLIFF_THRESHOLD:
            break
        d = get_demand_vector(req, apply_pruning=False)
        if curr_vram - d[0] < 0:
            break
        
        batch.append(req)
        curr_vram -= d[0]
        
    return batch


def sched_pats(queue: List[Request], current_lora, current_time):
    """
    Vision-constrained Vector Bin Packing with Sequence-dependent Setup
    Approximate Solver: PATS
    """
    if not queue: return []
    
    BETA = 2.0
    GAMMA = 0.05
    
    residual_vram = TOTAL_VRAM
    residual_compute = TOTAL_COMPUTE_CAPACITY
    current_load = len(queue)
    
    enable_pruning_mode = (current_load / CLIFF_THRESHOLD) > 0.7
    
    remaining = list(queue)
    batch = []
    
    while remaining and len(batch) < CLIFF_THRESHOLD:
        A = get_normalized_vector(residual_vram, residual_compute)
        if A[0] <= 1e-6:
            break
        
        best_req = None
        best_score = -1.0
        best_idx = -1
        
        for i, req in enumerate(remaining):
            should_prune = False
            steps = PRUNED_STEPS if should_prune else FULL_STEPS
            
            is_cached = (req.lora_id == current_lora)
            t_switch = 0.0 if is_cached else req.switch_cost
            t_compute = steps * STEP_TIME
            
            eff_time = t_switch + t_compute
            if eff_time <= 0:
                eff_time = 1e-3
            efficiency = 1.0 / eff_time
            
            d_vec = get_demand_vector(req, apply_pruning=should_prune)
            r = get_normalized_vector(d_vec[0], d_vec[1])
            
            dot = np.dot(r, A)
            norm_r = np.linalg.norm(r)
            norm_A = np.linalg.norm(A)
            
            cos_sim = 0.0
            if norm_r > 0 and norm_A > 0:
                cos_sim = dot / (norm_r * norm_A)
            alignment = math.exp(-BETA * (1.0 - cos_sim))
            
            wait = max(0.0, current_time - req.arrival_time)
            aging = 1.0 + (GAMMA * wait)
            
            score = efficiency * alignment * aging
            
            if score > best_score:
                best_score = score
                best_req = req
                best_idx = i
                
        if best_req is None:
            break
        
        should_prune_final = enable_pruning_mode and best_req.is_prunable
        d_final = get_demand_vector(best_req, apply_pruning=should_prune_final)
        
        if residual_vram - d_final[0] < 0:
            break
        
        batch.append(best_req)
        residual_vram -= d_final[0]
        residual_compute -= d_final[1]
        remaining.pop(best_idx)
        
    return batch


# ============================================================
# 🏃 4. Execution Engine
# ============================================================

def run_simulation(algo_name, scheduler_func, request_pool):
    print(f"\n🚀 Running Algorithm: [{algo_name}]")
    
    queue = deque(sorted(request_pool, key=lambda x: x.arrival_time))
    current_lora = None
    current_time = 0.0
    
    processed_jobs = []
    lora_usage_history: Dict[str, List[float]] = {}
    switches = 0
    
    with tqdm(total=len(request_pool), desc=f"Processing {algo_name}", unit="req") as pbar:
        consecutive_empty_batches = 0
        
        while queue:
            if algo_name.startswith("DRF"):
                batch = scheduler_func(list(queue), lora_usage_history, current_time)
            else:
                batch = scheduler_func(list(queue), current_lora, current_time)
                
            if not batch:
                current_time += 0.1
                consecutive_empty_batches += 1
                if consecutive_empty_batches > 1000:
                    print(f"\n⚠️ [Deadlock Detected] Force scheduling first request.")
                    req = queue.popleft()
                    batch = [req]
                else:
                    continue
            
            consecutive_empty_batches = 0
            
            for req in batch:
                try:
                    queue.remove(req)
                except ValueError:
                    pass
            
            target_lora = batch[0].lora_id
            target_switch_cost = batch[0].switch_cost
            
            if current_lora != target_lora:
                current_time += target_switch_cost
                current_lora = target_lora
                switches += 1
            
            max_steps_in_batch = 0
            
            for req in batch:
                req.start_time = current_time
                steps = FULL_STEPS
                if algo_name.startswith("PATS"):
                    if (len(queue) / CLIFF_THRESHOLD > 0.7) and req.is_prunable:
                        steps = PRUNED_STEPS
                
                max_steps_in_batch = max(max_steps_in_batch, steps)
                req.executed_steps = steps
                
                usage = lora_usage_history.get(req.lora_id, [0.0, 0.0])
                usage[0] += 1.5
                usage[1] += steps
                lora_usage_history[req.lora_id] = usage
                
            batch_duration = max_steps_in_batch * STEP_TIME
            current_time += batch_duration
            
            for req in batch:
                req.finish_time = current_time
                processed_jobs.append(req)
            
            pbar.update(len(batch))
            
    duration = current_time
    throughput = len(request_pool) / duration
    avg_latency = np.mean([j.finish_time - j.arrival_time for j in processed_jobs])
    
    return {
        "Algorithm": algo_name,
        "Total Time (s)": duration,
        "Throughput (req/s)": throughput,
        "Avg Latency (s)": avg_latency,
        "Switches": switches
    }


# ============================================================
# 🏁 Main
# ============================================================

if __name__ == "__main__":
    random.seed(42)
    
    # 1) switching_cost.csv 로딩 (실제 LoRA 스위칭 / step 시간)
    load_switch_profile("../data/switching_cost.csv")
    build_lora_io_costs()

    # 2) SDXL LPIPS 프로파일 (없으면 생성)
    profile_csv = "../data/sdxl_step_quality_lpips_60_60_60.csv"
    if not os.path.exists(profile_csv):
        try:
            run_quality_profiling()
        except Exception as e:
            print(f"\n⚠️ [Profiling] Error occurred, skipping profiling: {e}")
    else:
        print(f"\nℹ️ Found existing profiling CSV: {profile_csv} (skip profiling)")

    if not os.path.exists(profile_csv):
        print(f"\n❌ Critical: Profiling data {profile_csv} not found and generation failed.")
        exit(1)
        
    print(f"\n📂 [Workload] Loading Master Suite data from: {profile_csv}")
    df_profile = pd.read_csv(profile_csv)
    unique_prompt_types = df_profile['prompt_type'].unique()
    
    # 3) Workload 생성
    requests: List[Request] = []
    
    print(f"generating {NUM_REQUESTS} requests from {len(unique_prompt_types)} scenarios...")
    
    for i in range(NUM_REQUESTS):
        ptype = random.choice(unique_prompt_types)
        
        lora_id = "lora_portrait"
        complexity = 0.5
        
        for prefix, spec in PROMPT_TO_LORA_MAP.items():
            if ptype.startswith(prefix):
                lora_id, complexity = spec
                break
        
        switch_cost = LORA_IO_COSTS.get(lora_id, SWITCH_COST_BASE)
        arrival = i * 0.05
        
        req = Request(i, complexity, lora_id, switch_cost, arrival)
        requests.append(req)
            
    print(f"\n🧪 Generated {NUM_REQUESTS} requests.")
    print(f"   - Example Cost: minimal={LORA_IO_COSTS.get('lora_minimal',0):.3f}s vs ultra={LORA_IO_COSTS.get('lora_ultra',0):.3f}s")

    # 4) Scheduler Benchmark
    results = []
    results.append(run_simulation("Tetris",      sched_tetris,     list(requests)))
    results.append(run_simulation("DRF",         sched_drf,        list(requests)))
    results.append(run_simulation("Clockwork",   sched_clockwork,  list(requests)))
    results.append(run_simulation("PATS (Ours)", sched_pats,       list(requests)))
    
    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("🏆 FINAL 4-WAY BENCHMARK RESULTS (A100 Model)")
    print("="*70)
    print(df.to_string(index=False, float_format="%.2f"))
    print("="*70)

    os.makedirs("../data", exist_ok=True)
    output_path = "../data/scheduler_results_ablation1.csv"
    df.to_csv(output_path, index=False)
    print(f"\n📁 CSV saved to: {output_path}")

    timestamped_path = f"../data/scheduler_results_{int(time.time())}_ablation1.csv"
    df.to_csv(timestamped_path, index=False)
    print(f"📁 Timestamped CSV saved to: {timestamped_path}")
