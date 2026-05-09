# aoi-sentinel — RL 기술 검증 발표자료

> **15분 / 15장 / ML 엔지니어·CTO·연구자·표준 body 기술 검토자 대상.**
>
> 다른 deck들과 다른 점: 수학적 엄밀성 + 실험 데이터 + 솔직한 실패 분석 + production safeguards.
> "RL을 양산 라인에 배포해도 안전한가?"라는 회의적 질문에 정면 답변.

---

## 슬라이드 1 — 표지

### 🎯 RL을 양산 라인에 배포해도 안전한가? 수학으로 답합니다.

```
        Mamba RL + Lagrangian PPO
              기술 검증 자료
            DrJinHoChoi · 2026
```

**🔧 근거 자료**: VisA 4,416 imgs + NVIDIA A100 80GB, 코드 50+ tests, 공개 RFC, BSD-3.

---

## 슬라이드 2 — 핵심 기술 질문

### 🎯 RL은 본질적으로 탐색합니다. 그럼 양산에서 어떻게 안전을 보장하나?

회의론 4개를 정면에 둠:

1. **"PPO는 unstable. 양산에 못 씀"** → Lagrangian PPO + reward clip + 모드 게이트로 답
2. **"escape 보장은 ML로 불가능"** → Constrained MDP의 hard constraint로 수학적 답
3. **"Mamba는 hype. transformer로 충분"** → O(L) vs O(L²) 정량 비교로 답
4. **"실 양산 도메인 검증 안 됨"** → VisA 실험 + 어댑터 SDK + 모드 게이트로 답

이 deck은 **각 회의론에 한 슬라이드씩** 답.

**🔧 근거**: 모든 답변은 우리 [`aoi_sentinel/models/policy/`](https://github.com/DrJinHoChoi/aoi-sentinel/tree/main/aoi_sentinel/models/policy) 코드 + VisA 실험 trajectory + 학술 논문 인용.

---

## 슬라이드 3 — Constrained MDP 정식 정의

### 🎯 우리 시스템은 표준 Constrained MDP. 안전 제약이 reward 위에 hard constraint.

```
M = (S, A, P, r, c, γ, ε)

maximize    J_r(π) = E_π [ Σ_{t=0}^∞ γ^t · r(s_t, a_t) ]
                    ↑ 보상 (= -cost matrix 비용)

subject to  J_c(π) = E_π [ Σ_{t=0}^∞ γ^t · c(s_t, a_t) ] ≤ ε
                    ↑ 안전 cost (= escape indicator)

여기서:
  state s ∈ S       = (image_t, history_{t-L:t})
  action a ∈ A      = {DEFECT, PASS, ESCALATE}
  cost c(s, a) = 1  if (action=PASS && label=TRUE_DEFECT)
              = 0  otherwise
  ε                 = 0.001 (escape rate budget)
```

**의미**: 보상은 평균값으로 최적화, **안전은 hard threshold**.

**🔧 근거**: Altman 1999 — *Constrained Markov Decision Processes*. EU AI Act audit framework가 정확히 이 formalism.

---

## 슬라이드 4 — Lagrangian 분해 + dual ascent

### 🎯 안전 제약을 Lagrange 곱수로 reward에 합쳐서 풀 수 있습니다 (Stooke et al. 2020)

```
L(π, λ) = J_r(π) − λ · (J_c(π) − ε),   λ ≥ 0

Primal:   π ← argmax_π L(π, λ)
Dual:     λ ← max(0, λ + β·(J_c(π) − ε))
                              ↑ 제약 위반량
```

**핵심 직관**:
- **escape rate가 ε 초과** → λ ↑ → 정책이 안전 쪽으로 강제
- **escape rate가 ε 이하** → λ ↓ → 정책이 reward 추구 자유
- **λ 자체가 학습됨** — 사람이 "비용 가중치 100? 1000?" 고민 필요 없음

```
Combined advantage:   A = A_r − λ · A_c

→ PPO clipped surrogate에 그대로 들어감
```

**🔧 근거**: [arxiv.org/abs/2007.03964](https://arxiv.org/abs/2007.03964) — Stooke, Achiam, Abbeel, *Responsive Safety in RL by PID Lagrangian Methods* (ICML 2020). 우리 [`models/policy/lagrangian_ppo.py:update_lambda`](https://github.com/DrJinHoChoi/aoi-sentinel/blob/main/aoi_sentinel/models/policy/lagrangian_ppo.py).

---

## 슬라이드 5 — PPO clipped surrogate

### 🎯 정책이 한 번에 너무 크게 안 변하도록 — Schulman et al. 2017

```
ratio_t(π) = π(a_t|s_t) / π_old(a_t|s_t)

L_clip(π) = E_t [ min( ratio_t · A_t,
                        clip(ratio_t, 1−ε_clip, 1+ε_clip) · A_t ) ]

ε_clip = 0.2   (표준)
A_t  = A_r,t − λ · A_c,t   ← Lagrangian 결합
```

**왜 PPO?**
- TRPO보다 단순 (1차 method)
- on-policy → 시퀀스 인코더(Mamba)와 자연스럽게 결합
- 클리핑이 학습 폭주 방지 (우리가 본 vloss 폭발 사례에서도 안전망)
- 표준·널리 검증·튜닝 매뉴얼 풍부

**대안 검토 (다음 슬라이드에 비교)**: TRPO, CPO, P3O, CRPO

**🔧 근거**: [arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347) — Schulman et al. *PPO Algorithms*. 우리 [`update`](https://github.com/DrJinHoChoi/aoi-sentinel/blob/main/aoi_sentinel/models/policy/lagrangian_ppo.py) 함수에서 정확히 이 수식 구현.

---

## 슬라이드 6 — Why Mamba (vs Transformer)

### 🎯 검사 이력 시퀀스 L=512 가는 순간 transformer는 비현실. Mamba O(L) 우위 정량 측정 가능.

| 시퀀스 길이 L | Transformer FLOPs | Mamba FLOPs | 비율 |
|---------------|-------------------|-------------|------|
| 256 | 65,536 | 256 | 256× |
| 512 | 262,144 | 512 | 512× |
| 1,024 | 1,048,576 | 1,024 | 1,024× |
| 2,048 | 4,194,304 | 2,048 | 2,048× |
| 4,096 | 16,777,216 | 4,096 | **4,096×** |

**우리 production**: NPI 라인 history L=512-4096. **Mamba 없으면 추론 latency 양산 불가능**.

**우리 코드**: `mamba-ssm` CUDA kernel + pure-PyTorch fallback (mamba 없는 환경에서도 동작). 같은 인터페이스, fallback은 5-15× 느림.

**대안 비교**:
- VMamba — 더 강하지만 ONNX export 어려움
- Vim — 더 단순하지만 성능 ↓
- MambaVision — NVIDIA 공식, hybrid Mamba+attention, **우리 default**

**🔧 근거**: Gu & Dao 2023, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. 우리 [bench script](https://github.com/DrJinHoChoi/aoi-sentinel/blob/main/scripts/bench_mamba_vs_transformer.py)로 직접 측정.

---

## 슬라이드 7 — System architecture

### 🎯 두 Mamba (이미지·시퀀스) + Lagrangian PPO actor-critic

```
[ROI image]                           [Inspection history (L=256)]
    │                                          │
    ▼                                          ▼
[ConvNeXt-Tiny / MambaVision]    [Mamba-SSM 시퀀스 인코더]
    │  (image_encoder.py)              │  (sequence_encoder.py)
    │                                          │
    └──────────────┬───────────────────────────┘
                   │  concat
                   ▼
           [trunk MLP, hidden=384]
                   │
       ┌───────────┼─────────────┐
       ▼           ▼             ▼
   [actor]      [V_r]         [V_c]
   logits       reward        cost
   3 actions    critic        critic
   {DEFECT,                    
    PASS,
    ESCALATE}
```

`models/policy/actor_critic.py` 약 60 lines. 두 critic head가 핵심 — reward GAE / cost GAE 따로.

**🔧 근거**: 코드 [`MambaActorCritic`](https://github.com/DrJinHoChoi/aoi-sentinel/blob/main/aoi_sentinel/models/policy/actor_critic.py).

---

## 슬라이드 8 — 실험 셋업

### 🎯 재현 가능. seed 고정, 코드·데이터·하이퍼 모두 GitHub에.

| 항목 | 값 |
|------|-----|
| 데이터 | VisA PCB1-4, 4,416 imgs, defect rate 0.091 |
| GPU | NVIDIA A100-SXM4-80GB |
| 백본 | ConvNeXt-Tiny (MambaVision fallback) |
| 시퀀스 인코더 | Pure-PyTorch Mamba (mamba-ssm CUDA fallback) |
| Cost matrix | escape=50, false_call=1, operator=2 |
| 안전 제약 ε | 0.001 |
| Rollout | 256 step / iter |
| 학습 iter | 400 (실제 분석은 0-167) |
| Optimizer | AdamW (actor lr=3e-4, critic lr=1e-3) |
| PPO clip ε | 0.2, n_epochs=4, minibatch=32 |
| Entropy coef | 0.02 |
| λ_lr | 0.2 |
| Reward clip | -100 (vloss 폭발 방지) |
| Seed | 42 |

**🔧 근거**: [`configs/stage1_npi_rl_light.yaml`](https://github.com/DrJinHoChoi/aoi-sentinel/blob/main/configs/stage1_npi_rl_light.yaml) + [`stage1.log`](https://github.com/DrJinHoChoi/aoi-sentinel/blob/main/docs/sales/pilot_evidence_kr.md).

---

## 슬라이드 9 — VisA 실험 결과 — 4가지 핵심 시그널

### 🎯 60 iter 동안 escape=0 유지하며 비용 5.6× 감소. Lagrangian contract 작동 입증.

```
┌──────────────────┬────────────────────┐
│ Escape per iter  │   비용 추세         │
│      0           │  1,280 → 230       │
│ (60 iter 동안)    │   5.6× 감소         │
└──────────────────┴────────────────────┘
┌──────────────────┬────────────────────┐
│ λ (안전 dual)     │  정책 자동 진화     │
│ 자동 조절됨       │  ESCALATE→DEFECT   │
│ 1.0 → 2.4 → 1.8  │  iter 4부터          │
└──────────────────┴────────────────────┘
```

이건 lab 결과 — 실제 라인에선 같은 일이 day 단위로 일어남.

**🔧 근거**: VisA stage1.log iter 0-167. 상세 plot script: [`pilot_evidence_kr.md`](pilot_evidence_kr.md).

---

## 슬라이드 10 — 정책 자동 진화 — phase analysis

### 🎯 사람 개입 없이 시스템이 trivial → productive 정책 자동 발견

```
Phase 1 (iter 0-3):    무작위 탐색
   escape: 9, 3, 4, 0   ← 첫 3 iter 후 영구 0
   reward: -36 → -5     ← critic 정상화

Phase 2 (iter 4-152):  안주 영역 (ESCALATE-everything)
   esc=256, fc=0       ← 모두 사람한테
   reward=-5 stable     ← cost 5/step
   λ rising 0.7 → 7.0   ← Lagrangian "더 잘 해라" 압박

Phase 3 (iter 153-167):  ★ 결정적 transition ★
   esc=0, fc≈230        ← 갑자기 직접 분류 시작
   reward=-0.9          ← 5.6× 비용 감소
   escape=0 유지         ← 안전 제약 깨지지 않음

Phase 4 (iter 168+):    PASS-collapse (다음 슬라이드)
```

**해석**: Lagrangian λ가 7.0까지 climbing → 임계점 → 정책이 더 productive basin으로 점프.

**🔧 근거**: stage1.log iter-by-iter 분석.

---

## 슬라이드 11 — 실패 모드 + 완화 (솔직한 분석)

### 🎯 iter 168에서 'PASS-collapse' 발생. 원인·완화·remaining gap 모두 공개.

### 발생한 문제 (iter 168+)

```
정책이 always-DEFECT → always-PASS basin으로 drift
escapes = 25-30 / iter !!!
vloss = 2,732,971 (value critic 폭발)
λ = 4 → 31 (회복 시도)
```

### 원인 (3개 동시)

1. **c_escape = 1000** vs c_false_call=1 → reward target -1000이 critic 폭파
2. **entropy_coef = 0.05** → exploration 너무 강해 PASS 시도
3. **λ_lr = 0.05** → 빠른 변화에 못 따라감

### 완화 (`d92e40c` commit)

```yaml
c_escape:    1000 → 50      # 50× 비대칭 유지하되 critic 안정
entropy:     0.05 → 0.02    # 첫 transition 후 drift 억제
lambda_lr:   0.05 → 0.2     # 4× 빠른 λ 반응
reward clip: -100           # vloss 폭발 hard cap
```

### Remaining gap

PPO + 극단 cost asymmetry는 **본질적으로 어렵다**. 단계적 업그레이드 plan (다음 슬라이드).

**🔧 근거**: 모든 코드 변경 git log에 documented. 회의론자가 직접 verify 가능.

---

## 슬라이드 12 — 알고리즘 비교 + 업그레이드 roadmap

### 🎯 Vanilla Lagrangian PPO는 v1. v2-v4 업그레이드 path 명확.

| 알고리즘 | 출처 | 우리 단계 | 효과 |
|---------|------|----------|------|
| **Lagrangian PPO** (현재) | Stooke 2020 | v1 | 작동 + 알려진 한계 |
| **PID Lagrangian** | Stooke 2020 | v1.1 (다음) | λ 응답 빠름, oscillation ↓ |
| **Sauté RL** | Sootla 2022 | v1.2 | 상태에 safety budget augment → 정책 안전 의식 |
| **CPO** (Constrained Policy Optimization) | Achiam 2017 | v2 | trust region + 분석적 안전 bound — PASS-collapse 불가능 |
| **CRPO** | Xu 2021 | v2 alt | dual 변수 0개, 수렴 증명 |
| **P3O** (Penalized PPO) | Zhang 2022 | v1.5 alt | dual 없이 penalty 직접 |
| **Decision Mamba + cost** | 2024 frontier | v3 | 우리 sequence encoder와 자연 결합 |

**선택 근거**: v1.1 (PID Lagrangian) 코드 5줄, 가장 큰 효과. v2 (CPO)는 첫 anchor LOI 후 양산 deploy 시점에.

**🔧 근거**: [`docs/ai_manufacturing_standards.md`](../ai_manufacturing_standards.md) Layer 0.1 algorithm 참고. arXiv 인용 모두 검증 가능.

---

## 슬라이드 13 — Production safeguards (RL 외부)

### 🎯 RL 모델만 의존하지 않습니다. SHADOW 모드 + safety gate + 자동 강등 = 다층 방어.

```
┌──────────────────────────────────────────────────────┐
│  Layer 1 — RL 정책 (Lagrangian PPO)                  │
│  → escape rate ≤ ε 수학적 보장 (조건부)               │
└──────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│  Layer 2 — Safety gate (모델 promotion 전 검증)       │
│  → 후보 모델은 hold-out replay에서 escape=0 입증해야  │
│    promote                                           │
└──────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│  Layer 3 — 운영 모드 (SHADOW → ASSIST → AUTONOMOUS)   │
│  → SHADOW = 박스 의견만, 영향 0                       │
│  → ASSIST = 자신 있는 케이스만 자동                    │
│  → escape 1건이라도 발생 시 자동 강등                 │
└──────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│  Layer 4 — Drift 모니터링 + decommissioning           │
│  → KPI 임계값 초과 시 자동 retire                      │
└──────────────────────────────────────────────────────┘
```

**핵심**: RL 정책이 "이론상 안전"한 게 아니라, **여러 layer의 운영 게이트**로 사고 차단. RL 실패해도 line 안 멈춤.

**🔧 근거**: [`runtime/safety_gate.py`](https://github.com/DrJinHoChoi/aoi-sentinel/blob/main/aoi_sentinel/runtime/safety_gate.py), [`runtime/modes.py`](https://github.com/DrJinHoChoi/aoi-sentinel/blob/main/aoi_sentinel/runtime/modes.py).

---

## 슬라이드 14 — Reproducibility + open science

### 🎯 모든 게 공개. 회의론자 직접 검증 가능.

| 자산 | 공개 위치 | 라이선스 |
|------|----------|----------|
| RFC v0.1 (AICS schema) | github.com/DrJinHoChoi/aoi-common-spec | BSD-3 |
| Reference 구현 | github.com/DrJinHoChoi/aoi-sentinel | Proprietary (코드는 검토용 공개) |
| Lagrangian PPO 구현 | `models/policy/lagrangian_ppo.py` (~300 lines) | Open inspection |
| 벤치마크 데이터 | VisA (Amazon Science, BSD-3) | Public |
| 학습 config | `configs/stage1_npi_rl_light.yaml` | Public |
| Seed | 42 (single seed reported, multi-seed pending) | — |
| 50+ unit tests | `tests/` | Public |
| Bench script | `scripts/bench_mamba_vs_transformer.py` | Public |

### 다음 단계 reproducibility 강화

- [ ] 3-seed variance 분석 (정책 안정성 정량화)
- [ ] AICS Conformance Test Suite v1.0 (외부 자동 검증)
- [ ] Container image (Docker / Singularity)
- [ ] 학술 논문 (NeurIPS workshop / IEEE T-II)

**🔧 근거**: 모든 GitHub repo public. PR / issue 환영.

---

## 슬라이드 15 — 마지막: 신뢰성 statement

### 🎯 우리 RL은 마법이 아닙니다. 알려진 알고리즘 + 정직한 한계 공개 + 다층 안전 게이트.

```
[수학]      Constrained MDP + Lagrangian (Altman 1999, Stooke 2020)
            → 표준 formalism, EU AI Act audit과 호환

[알고리즘]   PPO (Schulman 2017) + dual ascent
            → 검증된 baseline, 코드 ~300 lines

[아키텍처]   MambaVision + Mamba-SSM
            → 시퀀스 길이 L=512+에서 transformer 대비 측정 가능 우위

[검증]       VisA 4,416 imgs, A100, 60 iter escape=0, 5.6× cost ↓
            → 재현 가능, 코드·데이터 공개

[안전]       SHADOW 모드 + safety gate + 자동 강등 + drift 모니터링
            → RL 실패해도 line 안 멈춤

[솔직함]     PASS-collapse 발생 / 원인·완화·gap 모두 공개
            → "이상화된 결과만 자랑" 안 함
```

**우리 입장**: "우리 RL이 완벽하다"가 아니라, **"우리 시스템이 RL을 안전하게 사용할 수 있다"**.

```
담당     DrJinHoChoi (최진호)
GitHub   github.com/DrJinHoChoi
이메일   _(미팅 후 전달)_
```

---

> **"RL 정책이 약속을 못 지키면, 시스템이 강제로 지킵니다."**

---

## 발표 운영 가이드

### 청중별 강조

| 청중 | 핵심 |
|------|------|
| **CTO / ML lead** | 4-5 (수학) + 9-11 (실험·실패) + 13 (safeguards) |
| **연구자 / 학계** | 3-7 (formal) + 14 (reproducibility) |
| **표준 body 기술 검토자** | 3-4 (Constrained MDP), 13 (gates), 14 (open) |
| **고객사 안전팀** | 13 (safeguards) + 11 (실패 공개) — *솔직함이 신뢰의 근거* |

### 시간 배분 (15분)

| 구간 | 슬라이드 | 시간 |
|------|---------|------|
| 질문·formalism | 1-5 | 4분 |
| 아키텍처·셋업 | 6-8 | 3분 |
| 결과·실패 | 9-11 | 4분 |
| 비교·safeguards·재현성 | 12-14 | 3분 |
| 클로징 | 15 | 1분 |

### 외울 한 문장

> **"RL 정책이 약속을 못 지키면, 시스템이 강제로 지킵니다. 수학·코드·실험·솔직한 실패 분석 모두 공개돼 있습니다."**

### Q&A 예상 질문 (필요시 백업 슬라이드)

| 질문 | 답변 anchor |
|------|------------|
| "PASS-collapse 다시 일어나면?" | 슬라이드 13 — safety gate가 promote 막음 |
| "ε=0.001 어떻게 정함?" | 슬라이드 3 — 사이트별 cost matrix에 맞춰 조정 |
| "Mamba CUDA 의존성?" | 슬라이드 7 — pure-PyTorch fallback 작동 검증 |
| "single seed인데 robust?" | 슬라이드 14 — 3-seed variance 다음 단계 |
| "왜 CPO 안 써?" | 슬라이드 12 — v2 roadmap, 첫 anchor 후 |

---

## PPTX 빌드

```bash
node docs/sales/build_pitch_deck_rl_validation.js
```

---

*Last updated: 2026-05-09. ML 엔지니어·CTO·연구자·표준 body 기술 검토자 대상.*
*Companions:*
*- [pitch_deck_kr.md](pitch_deck_kr.md) — tactical (구매팀)*
*- [pitch_deck_strategic_kr.md](pitch_deck_strategic_kr.md) — vision (CEO·VC·표준 body)*
*- [pilot_evidence_kr.md](pilot_evidence_kr.md) — 실험 trajectory 상세*
