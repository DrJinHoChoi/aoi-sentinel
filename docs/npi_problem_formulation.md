# NPI Problem Formulation

> Reducing Saki AOI false-call rate over the lifetime of a new product (NPI = New Product Introduction), starting from t=0 with no product-specific labels.

## 1. Setting

A new automotive electronics product enters mass production. Saki AOI inspects every board with rule-based logic tuned for prior products. Result:
- Saki call rate (NG flags) is high — typical 10–30%
- Of those, ~70% are false calls (true defect rate ~3–10%)
- Operator re-inspection is the only ground-truth source — slow and expensive

Standard practice: tune Saki rules manually over weeks. We replace this with an online learning agent that adapts the *acceptance policy* itself, gated by a hard safety constraint on escapes.

## 2. MDP / Constrained-MDP

Per-ROI decision step `t`:

- **State** `s_t = (φ_img(I_t), φ_seq(h_t))`
  - `φ_img`: Vision-Mamba (MambaVision) embedding of the current ROI image
  - `φ_seq`: Mamba-SSM over the recent inspection history `h_t` of length `L` — encodes line phase, drift, recent label distribution
- **Action** `a_t ∈ {DEFECT, PASS, ESCALATE}`
  - DEFECT: agree with Saki, route to rework
  - PASS: override Saki, accept the part
  - ESCALATE: send to human operator, get ground truth `y_t`
- **Reward (cost) matrix** `r_t = -C[y_t, a_t]` after `y_t` is revealed
  - Revealed immediately if `a_t = ESCALATE`
  - Revealed lazily (final inspection / customer return) if `a_t ∈ {DEFECT, PASS}` — modeled as delayed reward; in simulator, oracle reveals at action time
- **Cost matrix** (defaults — tunable per product):

  |               | a=DEFECT | a=PASS  | a=ESCALATE |
  |---------------|----------|---------|------------|
  | y=TRUE_DEFECT | 0        | **C_E** | C_op       |
  | y=FALSE_CALL  | C_fc     | 0       | C_op       |

  with `C_E ≫ C_fc, C_op` (escape ≈ 1000× false-call). Concrete: `C_E=1000`, `C_fc=1`, `C_op=5`.

- **Constraint** (Constrained MDP):  `E[1{escape_t}] ≤ ε` over a moving window. Default ε=0.001.

The agent maximises expected return subject to the constraint via a **Lagrangian PPO**:

```
J(π) = E[Σ r_t]   s.t.   E[Σ c_t] ≤ ε,  c_t = 1{y=TRUE_DEFECT, a=PASS}
L(π, λ) = J(π) - λ (E[Σ c_t] - ε),   λ ≥ 0
```

`λ` is updated by gradient ascent — automatic tuning of safety strictness.

## 3. Why Mamba

Two distinct uses, both linear-time:

1. **Image encoder** — MambaVision-T/S over 224² ROI. Fine-grained texture, small-defect friendly. Pretrained on ImageNet + benchmark PCB/SMT data (Phase 0).
2. **Sequence encoder** — vanilla Mamba-SSM over the line's recent inspection history `h_t = [(φ_img_{t-L}, a_{t-L}, y_{t-L}), ...]`. With `L=512–4096`, transformer self-attention is `O(L²)`, untenable inline. Mamba is `O(L)`. The sequence carries:
   - Drift signal (recent defect rate spike → tighten policy)
   - Phase signal (early NPI vs stabilized vs end-of-life)
   - Local correlation (neighboring boards from same SMT setup)

## 4. Pipeline

```
[Saki ROI image]                         [Inspection history (last L steps)]
         │                                          │
         ▼                                          ▼
  MambaVision encoder φ_img            Mamba-SSM sequence encoder φ_seq
         │                                          │
         └────────────────────┬─────────────────────┘
                              ▼
                      Concat → Policy head (actor)
                              ├──> action ∈ {DEFECT, PASS, ESCALATE}
                              ├──> value V(s)        (reward critic)
                              └──> cost-value V_c(s) (safety critic)

                       Lagrangian PPO update with λ-dual ascent
```

## 5. Phases

| Phase | Goal | Data | Algorithm |
|-------|------|------|-----------|
| 0. Pretrain | Strong cold-start image encoder | VisA, DeepPCB, SolDef_AI, AI-Hub | Cost-sensitive focal + selective head |
| 1. Sim build | NPI streaming env on benchmark data | Same datasets, replayed time-ordered | Gymnasium env + simulated operator oracle |
| 2. Online RL | Mamba RL agent learns to reduce false calls under escape constraint | NPI simulator | Lagrangian PPO with Mamba encoders |
| 3. Real Saki | Cold-start on real production line | Live Saki stream | Same agent, frozen image encoder, live sequence encoder + policy |
| 4. 3D + RAG | Phase 3D analyzer, LLM-RAG cause inference | Saki 3D + history DB | Separate modules, fused at decision time |

## 6. Open Risks

- **Escape constraint violation during exploration** — mitigated by warm-starting policy from Phase 0 selective head (high ESCALATE rate, near-zero PASS) and gradually lowering λ.
- **Distribution shift after line stabilization** — handled by experience-replay decay and periodic offline RL refresh.
- **Operator label noise** — handle as label-smoothing on reveal; flag suspected disagreements for re-review.
- **Sequence cold start** (`t < L`) — pad with neutral tokens; sequence encoder must be robust to short prefixes.

## 7. Evaluation

Primary, on the simulator and (later) on real lines:
- **Cumulative cost over time** vs. baselines (Saki-only, fixed-threshold selective, no-RL)
- **False-call rate trajectory** — should decrease over time
- **Escape rate** — must stay ≤ ε at all times
- **ESCALATE rate** — should also decrease as confidence grows
- **Sample efficiency** — boards-to-target performance

Baselines:
1. Saki raw (all flags accepted) — upper bound on cost
2. Pretrained selective classifier with fixed reject threshold (Phase 0 model only)
3. Online learning *without* sequence encoder (image-only)
4. Online learning with transformer sequence encoder (cost ablation)
5. Full Mamba RL (ours)
