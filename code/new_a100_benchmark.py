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
    - 이미지: ../assets/quality_study/
    - CSV:    ../data/sdxl_step_quality_lpips.csv
    """
    import torch
    from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DDIMScheduler
    from PIL import Image
    import torchvision.transforms as T
    import lpips
    import warnings

    # 불필요한 경고 메시지 억제 (LPIPS, torchvision 관련)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    IMG_DIR = "../assets/quality_study_fixed_60_60"
    CSV_PATH = "../data/sdxl_step_quality_lpips_fixed_60_60.csv"
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    device = "cuda"

    # ---- 0-1. 모델 로드 ----
    print("\n🧠 [Profiling] Loading SDXL + fixed VAE on A100...")
    
    # accelerate 라이브러리 부재시 경고 억제 및 호환성 확보
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

    # ---- 0-2. LPIPS 모델 ----
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

    # ---- 0-3. 프롬프트 (연구용 Extreme Contrast 버전) ----
    prompts = {
        "Simple": (
            "minimalist flat vector art of a single red apple on a white background, "
            "simple shapes, clean lines, high contrast, no texture, no shadow"
        ),
        "Complex": (
            "an immense chaotic battlefield with thousands of soldiers, dragons flying in the sky, "
            "exploding magic spells, intricate armor details, smoke, fire, lightning, "
            "hyper-realistic texture, 8k resolution, ray tracing, volumetric lighting, "
            "glowing runes, crowd scene, wide angle lens"
        ),
    }

    SEED = 42
    steps_list = [60, 50, 40, 30, 20, 10, 5]

    rows = []

    print("\n🚀 [Profiling] Generating baseline + variants for LPIPS curve...")

    for ptype, ptext in prompts.items():
        print(f"\n=== [{ptype}] prompt profiling ===")

        # baseline: max step (60)
        gen = torch.Generator(device).manual_seed(SEED)
        
        # Updated Syntax for PyTorch 2.x+
        with torch.amp.autocast('cuda'):
            baseline_img = pipe(
                ptext,
                num_inference_steps=60,
                generator=gen,
            ).images[0]

        baseline_path = os.path.join(IMG_DIR, f"{ptype}_baseline_step60.png")
        baseline_img.save(baseline_path)
        print(f"  -> Baseline saved: {baseline_path}")

        # 각 step 마다 이미지 + LPIPS
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

    # ---- 0-4. CSV 저장 ----
    df_profile = pd.DataFrame(rows)
    df_profile.to_csv(CSV_PATH, index=False)
    print(f"\n✅ [Profiling] LPIPS profiling saved to: {CSV_PATH}")
    print(f"✅ [Profiling] Images saved under: {IMG_DIR}")


# ============================================================
# ⚙️ 1. System Configuration & Constants (A100 Profiling Based)
# ============================================================
# Resource Constraints
TOTAL_VRAM = 40.0       # GB (A100 Limit)
# Compute Capacity를 "한 배치에서 처리 가능한 총 Step 수"로 정의 (Vector Packing을 위해)
# 가정: 배치 22개 * 50 Step = 1100 Step을 100% Capacity로 설정
TOTAL_COMPUTE_CAPACITY = 22 * 50.0 

CLIFF_THRESHOLD = 22    # Batch Size Limit (Hard Constraint)
SAFE_MARGIN = 18        # Pruning Trigger Point

# Costs (Measured from your profiling)
SWITCH_COST_BASE = 1.6286  # sec
STEP_TIME = 0.06           # sec/step

# Job Specs (from your LPIPS experiment: FULL=60, PRUNED=30)
FULL_STEPS = 60         # Standard
PRUNED_STEPS = 20       # Optimized for Simple prompts (Safe Margin for Quality)

# Simulation Settings
NUM_REQUESTS = 1000


# ============================================================
# 🧩 2. Data Structures & Helper Functions
# ============================================================

class Request:
    def __init__(self, req_id, complexity, lora_id, lora_rank, arrival_time):
        self.id = req_id
        self.complexity = float(complexity) # 0.0 ~ 1.0
        self.lora_id = lora_id
        self.lora_rank = float(lora_rank)
        self.arrival_time = float(arrival_time)
        
        # 0.3 미만이면 Simple -> Pruning 대상
        self.is_prunable = (self.complexity < 0.3)
        
        # 실행 기록용
        self.start_time = None
        self.finish_time = None
        self.executed_steps = 0

def get_normalized_vector(vram_gb, steps):
    """자원 벡터를 [0,1]로 정규화 (Tetris 내적 계산용)"""
    return np.array([
        vram_gb / TOTAL_VRAM,
        steps / TOTAL_COMPUTE_CAPACITY
    ])

def get_demand_vector(req: Request, apply_pruning: bool) -> np.ndarray:
    """
    작업의 요구 자원 벡터 반환 [VRAM, Steps]
    apply_pruning=True일 경우 줄어든 Step 반환
    """
    # VRAM: Activation + KV Cache (Roughly 1.5GB per req)
    vram = 1.5
    
    # Compute: Steps
    if apply_pruning and req.is_prunable:
        steps = PRUNED_STEPS
    else:
        steps = FULL_STEPS
        
    return np.array([vram, float(steps)])

# ============================================================
# 🧠 3. Scheduler Algorithms (Canonical Implementation)
# ============================================================

# --- 3.1 Tetris (SIGCOMM '14) ---
def sched_tetris(queue: List[Request], current_lora, current_time):
    if not queue: return []
    
    residual_vram = TOTAL_VRAM
    residual_compute = TOTAL_COMPUTE_CAPACITY
    
    batch = []
    temp_queue = list(queue)
    
    while len(batch) < CLIFF_THRESHOLD and temp_queue:
        A = get_normalized_vector(residual_vram, residual_compute)
        if A[0] <= 1e-6 or A[1] <= 1e-6: break
        
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
        
        if best_req is None: break
        
        d_best = get_demand_vector(best_req, apply_pruning=False)
        if residual_vram - d_best[0] < 0: break
        
        batch.append(best_req)
        residual_vram -= d_best[0]
        residual_compute -= d_best[1]
        temp_queue.pop(best_idx)
        
    # [FIX] Do not remove from local copy of queue. Let the caller handle it.
    return batch


# --- 3.2 DRF (NSDI '11) ---
def sched_drf(queue: List[Request], lora_usage_history: Dict[str, List[float]], current_time):
    if not queue: return []
    if lora_usage_history is None: lora_usage_history = {}
    
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
        if len(batch) >= CLIFF_THRESHOLD: break
        
        d = get_demand_vector(req, apply_pruning=False)
        
        if curr_vram - d[0] < 0: break
        
        batch.append(req)
        curr_vram -= d[0]
        
    # [FIX] Do not remove from local copy of queue. Let the caller handle it.
    return batch


# --- 3.3 Clockwork (OSDI '20 style) ---
def sched_clockwork(queue: List[Request], current_lora, current_time):
    if not queue: return []
    
    def get_score(req):
        is_cached = (req.lora_id == current_lora)
        return (1 if is_cached else 0, -req.arrival_time)
    
    sorted_queue = sorted(queue, key=get_score, reverse=True)
    
    batch = []
    curr_vram = TOTAL_VRAM
    
    for req in sorted_queue:
        if len(batch) >= CLIFF_THRESHOLD: break
        d = get_demand_vector(req, apply_pruning=False)
        if curr_vram - d[0] < 0: break
        
        batch.append(req)
        curr_vram -= d[0]
        
    # [FIX] Do not remove from local copy of queue. Let the caller handle it.
    return batch


# --- 3.4 PATS (Vision-Aware Tetris - Ours) ---
def sched_pats(queue: List[Request], current_lora, current_time):
    """
    Approximate Solver for "Vision-constrained Vector Bin Packing with Sequence-dependent Setup"
    Objective: Maximize [ Efficiency * Alignment * Fairness ]
    """
    if not queue: return []
    
    # Hyperparameters for Objective Function
    BETA = 2.0     # Weight for Vector Alignment (Tetris)
    GAMMA = 0.05   # Weight for Aging (Fairness)
    
    residual_vram = TOTAL_VRAM
    residual_compute = TOTAL_COMPUTE_CAPACITY
    current_load = len(queue)
    
    # Pruning Policy: High Load (>70%) -> Trigger Vision Approximation
    enable_pruning_mode = (current_load / CLIFF_THRESHOLD) > 0.7
    
    remaining = list(queue)
    batch = []
    
    while remaining and len(batch) < CLIFF_THRESHOLD:
        # A vector: Residual Capacity (남은 자원 벡터)
        A = get_normalized_vector(residual_vram, residual_compute)
        if A[0] <= 1e-6: break
        
        best_req = None
        best_score = -1.0
        best_idx = -1
        
        for i, req in enumerate(remaining):
            # 1. Decision Variable s_j: Pruning Step
            should_prune = enable_pruning_mode and req.is_prunable
            steps = PRUNED_STEPS if should_prune else FULL_STEPS
            
            # 2. Sequence-Dependent Setup Cost (TSP Component)
            is_cached = (req.lora_id == current_lora)
            t_switch = 0.0 if is_cached else (SWITCH_COST_BASE * req.lora_rank)
            t_compute = steps * STEP_TIME
            
            # [Term 1] Efficiency (Marginal Gain) = Utility / Effective_Time
            # EffTime(j) includes switching penalty -> induces Clustering
            eff_time = t_switch + t_compute
            if eff_time <= 0: eff_time = 0.001
            efficiency = 1.0 / eff_time
            
            # [Term 2] Vector Alignment (Bin Packing Component)
            d_vec = get_demand_vector(req, apply_pruning=should_prune)
            r = get_normalized_vector(d_vec[0], d_vec[1])
            
            dot = np.dot(r, A)
            norm_r = np.linalg.norm(r)
            norm_A = np.linalg.norm(A)
            
            cos_sim = 0.0
            if norm_r > 0 and norm_A > 0:
                cos_sim = dot / (norm_r * norm_A)
                
            alignment = math.exp(-BETA * (1.0 - cos_sim))
            
            # [Term 3] Aging (Fairness Component)
            wait = max(0.0, current_time - req.arrival_time)
            aging = 1.0 + (GAMMA * wait)
            
            # Final Score Formulation
            score = efficiency * alignment * aging
            
            if score > best_score:
                best_score = score
                best_req = req
                best_idx = i
                
        if best_req is None: break
        
        # Final Check against Hard Constraints (CLIFF)
        should_prune_final = enable_pruning_mode and best_req.is_prunable
        d_final = get_demand_vector(best_req, apply_pruning=should_prune_final)
        
        if residual_vram - d_final[0] < 0: break
        
        batch.append(best_req)
        residual_vram -= d_final[0]
        residual_compute -= d_final[1]
        
        remaining.pop(best_idx)
        
    # [FIX] Do not remove from local copy of queue. Let the caller handle it.
    return batch


# ============================================================
# 🏃 4. Execution Engine (Benchmark Loop)
# ============================================================

def run_simulation(algo_name, scheduler_func, request_pool):
    print(f"\n🚀 Running Algorithm: [{algo_name}]")
    
    queue = deque(sorted(request_pool, key=lambda x: x.arrival_time))
    current_lora = None
    current_time = 0.0
    
    processed_jobs = []
    lora_usage_history = {}
    switches = 0
    
    # tqdm으로 진행상황 시각화
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
                
                # [Safety] 1000번 연속 공회전 시 강제 처리 (Deadlock 방지)
                if consecutive_empty_batches > 1000:
                    print(f"\n⚠️ [Deadlock Detected] Force scheduling first request.")
                    req = queue.popleft() # 강제로 하나 꺼냄
                    batch = [req]
                    # 주의: 스케줄러 함수 내부에서 remove를 못했으므로 여기서 수동 처리됨
                    # 하지만 아래 로직은 batch를 받아서 처리하므로 OK
                else:
                    continue
            
            # 배치가 성공적으로 만들어지면 카운터 리셋
            consecutive_empty_batches = 0
            
            # [FIX] Remove scheduled requests from the MAIN queue
            for req in batch:
                # deque.remove(x) removes the first occurrence of value x
                try:
                    queue.remove(req)
                except ValueError:
                    pass # Already removed (should not happen logic-wise but safe)
            
            target_lora = batch[0].lora_id
            target_rank = batch[0].lora_rank
            
            if current_lora != target_lora:
                time_cost = SWITCH_COST_BASE * target_rank
                current_time += time_cost
                current_lora = target_lora
                switches += 1
            
            max_steps_in_batch = 0
            
            for req in batch:
                req.start_time = current_time
                steps = FULL_STEPS
                if algo_name.startswith("PATS"):
                    if (len(queue)/CLIFF_THRESHOLD > 0.7) and req.is_prunable:
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
            
            # 배치 크기만큼 진행바 업데이트
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
# 🏁 Main Entry Point
# ============================================================
if __name__ == "__main__":
    random.seed(42)

    # 0. SDXL 프로파일링 먼저 (CSV 없을 때만)
    profile_csv = "../data/sdxl_step_quality_lpips_fixed_60_60.csv"
    if not os.path.exists(profile_csv):
        try:
            run_quality_profiling()
        except Exception as e:
            print(f"\n⚠️ [Profiling] Error occurred, skipping profiling: {e}")
    else:
        print(f"\nℹ️ Found existing profiling CSV: {profile_csv} (skip profiling)")

    # 1. Workload 생성 (실제 프로파일링 데이터 기반)
    requests = []
    
    if os.path.exists(profile_csv):
        print(f"\n📂 [Workload] Loading real profiling data from: {profile_csv}")
        df_profile = pd.read_csv(profile_csv)
        unique_types = df_profile['prompt_type'].unique()
        
        for i in range(NUM_REQUESTS):
            # 실제 프롬프트 타입 중 하나를 랜덤 선택 (분포 반영 가능)
            ptype = random.choice(unique_types)
            
            if ptype == "Simple":
                # Simple: 낮은 복잡도 -> Pruning 대상, 가벼운 LoRA 가정
                complexity = 0.2
                lora_id = "lora_simple"
                lora_rank = 0.8 
            else:
                # Complex: 높은 복잡도 -> Pruning 불가, 무거운 LoRA 가정
                complexity = 0.9
                lora_id = "lora_complex"
                lora_rank = 1.5
            
            arrival = i * 0.05
            requests.append(Request(i, complexity, lora_id, lora_rank, arrival))
            
    else:
        # 만약 CSV가 없으면 (Fallback)
        print("\n⚠️ Profiling data not found. Using random fallback.")
        for i in range(NUM_REQUESTS):
            comp = random.random()
            # Fallback LoRA pool
            lora_opts = [("lora_A", 1.0), ("lora_B", 1.2)]
            l_info = random.choice(lora_opts)
            req = Request(i, comp, l_info[0], l_info[1], i * 0.05)
            requests.append(req)
        
    print(f"\n🧪 Generated {NUM_REQUESTS} requests based on Real Profiling Data.")

    # 2. Benchmarks 실행
    results = []
    results.append(run_simulation("Tetris", sched_tetris, list(requests)))
    results.append(run_simulation("DRF", sched_drf, list(requests)))
    results.append(run_simulation("Clockwork", sched_clockwork, list(requests)))
    results.append(run_simulation("PATS (Ours)", sched_pats, list(requests)))
    
    # 3. 결과 출력 + CSV 저장
    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("🏆 FINAL 4-WAY BENCHMARK RESULTS (A100 Model)")
    print("="*70)
    print(df.to_string(index=False, float_format="%.2f"))
    print("="*70)

    os.makedirs("../data", exist_ok=True)
    output_path = "../data/scheduler_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\n📁 CSV saved to: {output_path}")

    timestamped_path = f"../data/scheduler_results_{int(time.time())}.csv"
    df.to_csv(timestamped_path, index=False)
    print(f"📁 Timestamped CSV saved to: {timestamped_path}")
