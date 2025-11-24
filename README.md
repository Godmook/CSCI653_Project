# 🚀 PATS: Perception-Aware Tetris Scheduling  
### Efficient Multi-LoRA Diffusion Serving under VRAM, Switching, and Perception Constraints

---

<p align="center">
  <img width="650" src="https://dummyimage.com/1200x260/000/fff&text=PATS:+Perception-Aware+Tetris+Scheduling">
</p>

---

# 1. Motivation

Modern multi-tenant LoRA-based diffusion serving faces three fundamental bottlenecks:

1. **VRAM Memory Cliffs** on A100 GPUs  
2. **Expensive LoRA Switching Cost**  
3. **Prompt-dependent Perception-Quality Trade-offs** (LPIPS)

Because of these conflicting constraints, naive FIFO or fairness-based schedulers fail.  
PATS introduces a perception-aware, VRAM-aware, switching-aware scheduling algorithm.

---

## 1.1 Memory Cliff (VRAM Wall)

GPU profiling on A100 (40/80GB) reveals:

- Beyond a specific batch size → **OOM or memory swapping**
- Latency rises sharply due to **VRAM cliff**
- Scheduling must respect a **Vector Bin Packing constraint**



---

## 1.2 Cost of LoRA Switching

From `switching_cost.csv`:

| Operation        | Time (s) |
|------------------|----------|
| LoRA Switch      | **1.16** |
| One SDXL Step    | **0.06** |

📌 **1 LoRA switch ≈ 20 inference steps**

Thus, batching should cluster requests using the same LoRA.

---

## 1.3 Perception-Compute Trade-off (LPIPS)

From `sdxl_step_quality_lpips.csv`:

- Simple prompts converge at **20 steps**
- Complex prompts require **50+ steps**

📌 Perception-aware pruning reduces up to **66% compute** without quality loss.




---

# 2. Problem Formulation (MILP + Lagrangian)

We define:

- $M$: VRAM capacity  
- $\tau_{\mathrm{switch}}$: LoRA switching overhead  
- $c_j \in [0,1]$: Prompt complexity  
- $S_{\mathrm{prune}} = 20$, $S_{\mathrm{full}} = 60$

---

## 2.1 Objective Function (Lagrangian Relaxation)

Latency is relaxed into a **Lagrangian penalty** to enable real-time decision making.

$$
\max 
\sum_{t=1}^{T} \sum_{j} x_{j,t} \, Q(s_j, c_j)
\;-\;
\lambda \sum_{j,t} x_{j,t}(t - a_j)
\;-\;
\sum_{t} I(y_t \ne y_{t-1}) \, \tau_{\text{switch}}
$$

### Interpretation

- **Quality term:** maximize perceptual quality  
- **Lagrangian term:** penalize latency  
- **Switching term:** discourage unnecessary LoRA swaps  

\(\lambda\) acts as a **Lagrange multiplier**, providing a principled latency–quality trade-off.

---

## 2.2 Constraints

### VRAM Constraint (Vector Bin Packing)

$$
\sum_{j} x_{j,t} \cdot \text{Mem}(j) \le M
$$

### Switching Constraint

$$
I(y_t \ne y_{t-1}) \Rightarrow \tau_{\text{switch}}
$$

---

# 3. NP-Hardness (Formal Restriction Proof)

### **Theorem: The LoRA scheduling problem is Strongly NP-Hard.**

### Proof Sketch

We show NP-hardness through **restriction**, following Garey & Johnson.

---

### Case 1 — Reduction to Vector Bin Packing

Impose:

- $\tau_{\mathrm{switch}} = 0$
- All jobs arrive at $t = 0$
- Identical complexities

Then scheduling reduces to:

$$
\mathrm{Pack\ jobs\ \ such\ that}\quad 
\sum \mathrm{Mem}(j) \le M
$$

This is exactly **Bin Packing**, which is strongly NP-hard.

---

### Case 2 — Reduction to TSP / Sequence-Dependent Scheduling

Set:

- $M = \infty$
- Sequential execution (1 job at a time)

Goal becomes:

$$
\min_{\sigma} \sum_i \mathrm{Cost}(\sigma(i), \sigma(i+1))
$$

Equivalent to:

- **Sequence-Dependent Setup Scheduling (SDSS)**
- **Path-TSP**

Both NP-hard.


---

### Conclusion

Because LoRA scheduling **strictly generalizes** both Vector Bin Packing and TSP:

### ✔ The full problem is **Strongly NP-Hard**.

Thus exact optimization is impractical for online serving → motivates PATS.

---

# 4. Proposed Algorithm: PATS  
### *Perception-Aware Tetris Scheduling*

PATS is a greedy-but-principled scheduler that maximizes:

> **marginal utility per unit time**

while respecting VRAM, switching cost, and perceptual complexity.

---

## 4.1 Dynamic Context Awareness (Intra-Batch Simulation)

Traditional schedulers treat each request independently.  
PATS performs **intra-batch GPU state simulation**:

1. Leader request determines the next LoRA  
2. Simulated GPU state \( L_{\text{sim}} \) is updated  
3. Followers using the same LoRA incur **zero switching cost**  
4. Forms a **LoRA train**

This greatly amortizes switching overhead across multiple requests.

---

## 4.2 Efficiency Score (Priority Rule)

For request $j$:

$$
\mathrm{Score}(j) =
\frac{1}{\mathrm{EffTime}(j)}
\cdot
\cos(\theta_{j,\mathrm{res}})
\cdot
\left(1 + \gamma \cdot \mathrm{Wait}(j)\right)
$$

### Effective Time

$$
\mathrm{EffTime}(j)=
\begin{cases}
S_{\mathrm{prune}} \, T_{\mathrm{step}} & \mathrm{(Cached,\ Simple)} \\
S_{\mathrm{full}} \, T_{\mathrm{step}}  & \mathrm{(Cached,\ Complex)} \\
S_{\mathrm{prune}} \, T_{\mathrm{step}} + \tau_{\mathrm{switch}} & \mathrm{(Uncached,\ Simple)} \\
S_{\mathrm{full}} \, T_{\mathrm{step}} + \tau_{\mathrm{switch}} & \mathrm{(Uncached,\ Complex)}
\end{cases}
$$

This formulation allows globally optimal, sometimes counterintuitive scheduling choices—
for example, selecting *Simple + Uncached* over *Complex + Cached* when the marginal efficiency is higher.


---

# 5. Experimental Evaluation (To Be Added)

You can insert the following later:

