# Manufacturing Standards FOR AI

> **"AI for manufacturing standards"는 데이터 layer이고, "manufacturing standards FOR AI"는 메타 layer다. 후자가 진짜 ₩100B 회사의 결정적 moat다.**

기존 [표준화 방법론](standardization_methodology.md)이 **검사 결과를 어떻게 표현하느냐**(AICS / CCS — 데이터 layer)를 다룬다면, 이 문서는 **그 위에서 동작하는 AI 자체를 어떻게 검증·비교·인증할 것인가** — 메타 layer를 다룬다.

읽는 시점: AICS / CCS 메인 표준화 진척이 안정 단계 (Year 2 Q1+) 진입 시.

---

## 1. 두 단계 표준화 — 우리가 어디 있고 어디로 가는가

```
[현재 — Layer 1: AI for manufacturing standards]
   AICS (검사 데이터 표준), CCS (가공 데이터 표준)
   "어떻게 vendor 무관하게 결과를 표현하나"
                              ↓
[다음 — Layer 0: Manufacturing standards FOR AI]
   AIAS, AISC, MMC — AI 자체의 audit·비교·인증 표준
   "어떻게 vendor 무관하게 AI 자체를 검증·비교·인증하나"
```

**이름이 Layer 0인 이유**: 데이터 layer 아래에 있는 게 아니라, **모든 AI에 적용되는 메타 표준**. AICS는 검사 결과만, AIAS는 AOI/CNC/로봇/자율주행 모든 제조 AI에 동일 적용.

---

## 2. 왜 지금 — 규제 inevitability

EU·미국·한국 모두 2024-2027 사이 제조 AI를 **high-risk AI**로 분류·규제 시작. **표준이 없으면 규제 audit이 불가능**해서, 누군가는 표준을 정의해야 한다.

| 규제 / 표준 | 발효 | 우리에게 의미 |
|------------|-----|--------------|
| **EU AI Act** | 2024 발효, 2026-2027 단계적 시행 | 제조 AI = high-risk 분류, 의무 audit |
| **ISO/IEC 42001** | 2023 published | AI Management Systems 표준 — 인증 가능 |
| **ISO/IEC 23894** | 2023 | AI Risk Management |
| **NIST AI RMF** | 2023 (US) | 미국 정부·방산 조달 요건화 진행 중 |
| **한국 AI 기본법** | 2025 통과, 2026 시행 | 시행규칙에 분야별 인증 절차 |
| **IEC 61508** + ML 확장 | 진행 중 | functional safety에 ML 모델 포함 |

**Window of opportunity**: 2026-2028. 그 안에 우리가 표준을 정의하고 채택을 만들면 — 2030년대 규제는 **우리 표준을 인용**한다. 그게 ₩100B 회사 진입 지점.

---

## 3. 8가지 표준 후보 — 우리가 정의해야 할 것

### Layer 0.1 — AI 결정 자체의 audit (가장 시급)

#### 1. **AIAS** — AI Audit Schema

AI가 내린 모든 결정에 첨부되는 메타데이터 표준.

```yaml
# 예시: AOI 결정 1건의 AIAS 메타
aias_version: "0.1"
decision_id: "uuid"
timestamp: "2026-05-09T08:14:22+09:00"

model:
  name: "aoi-sentinel-classifier"
  version: "v1.2.3"
  weights_hash: "sha256:..."
  training_data_hash: "sha256:..."
  framework: "PyTorch 2.10"

context:
  cost_matrix: { c_escape: 50, c_false_call: 1, c_operator: 2 }
  safety_constraint: { metric: "escape_rate", limit: 0.001 }
  operating_mode: "ASSIST"

decision:
  action: "PASS"
  confidence: 0.94
  rationale_class: "calibrated_high_confidence"

human_override:
  occurred: false
  operator_id: null
```

**정의자**: 우리. **채택 모멘텀**: 첫 anchor 사이트에서 자동 기록 → 케이스 스터디 → 다른 vendor 따라하게.

#### 2. **AISC** — AI Safety Contract

AI가 약속하는 안전 보장을 머신 readable로.

```yaml
contract_version: "0.1"
model: "aoi-sentinel-classifier@v1.2.3"

guarantees:
  - metric: "escape_rate"
    operator: "≤"
    value: 0.001
    confidence: 0.95
    domain: "automotive_smt_pcb"

  - metric: "false_call_reduction"
    operator: "≥"
    value: 0.70
    measured_after_days: 180

invalidation_conditions:
  - "training_data_drift > 0.3 KL"
  - "image_distribution_shift > threshold X"
  - "operator_label_disagreement > 0.05"

audit_endpoint: "/api/v1/audit/{decision_id}"
```

**채택 모멘텀**: OEM 조달팀이 AISC 형태로 vendor에게 요구 → vendor 모두 따라야 함.

#### 3. **MMC** — Manufacturing Model Card

HuggingFace model card의 제조 vertical 변형.

표준 필수 필드:
- 학습 데이터 분포 (defect 비율, 라인 다양성, vendor 다양성)
- 검증된 KPI (escape, false call, recall, precision)
- 실패 모드 (어떤 distribution shift에서 깨지는가)
- 적합 도메인 / 부적합 도메인
- 라이선스 / 데이터 출처
- Carbon footprint (학습 시 kWh)
- Last validated date

**채택 모멘텀**: 학술계 + ESG 보고서에 인용 → 산업으로 흘러옴.

### Layer 0.2 — AI 운영의 표준

#### 4. **Drift Detection Schema**

배포된 모델의 "아직 동작하나" 모니터링 표준.

표준 metric:
- KL divergence (입력 분포 변화)
- Confidence calibration drift
- Human override rate change
- Safety constraint violation count

#### 5. **Cross-vendor Benchmark Protocol**

Mycronic AI vs Koh Young AI vs 우리 — 동일 평가 프로토콜.

표준 정의:
- 공개 벤치마크 데이터셋 (VisA, DeepPCB 같은 것의 manufacturing 버전)
- 표준 지표 (cost-curve, escape rate at fixed FC budget, AURC)
- 통계적 유의성 테스트 방법
- Reproducibility 요건 (seed, config, hardware)

#### 6. **Federated Manufacturing Learning Protocol**

여러 고객사 데이터를 공유 없이 모델 학습하는 표준.

기존 (FedAvg, FedProx) 위에:
- 제조 vertical 특화 (cost matrix 공유 없음, 사이트별 distribution)
- Privacy budget (differential privacy ε)
- Drift coordination protocol

### Layer 0.3 — AI lifecycle 표준

#### 7. **AI Lineage Tracking**

데이터 → 모델 → 배포 → 결정 → 결과 전체 추적.

W3C PROV 표준 위에 manufacturing 확장 — "이 결정이 잘못됐다면 어느 학습 batch까지 책임을 물을 수 있나" 답변 가능.

#### 8. **AI Decommissioning**

더 이상 안전하지 않은 모델의 자동 retire 기준.

Trigger 표준:
- Drift score 임계값 초과 N일 연속
- Safety constraint violation 1회 이상
- 외부 audit 결과 fail
- 학습 데이터 라이선스 만료

---

## 4. 기존 표준과의 매핑

우리는 처음부터 만드는 게 아니라 **기존 AI governance 표준 위에 manufacturing extension**:

| 기존 표준 | 우리 확장 |
|----------|----------|
| **ISO/IEC 42001** (AI Management Systems) | manufacturing vertical Annex (검사·가공·로봇) |
| **ISO/IEC 23894** (AI Risk Management) | 제조 risk catalog (escape, false-call 카테고리) |
| **NIST AI RMF** | manufacturing profile |
| **EU AI Act Annex III** (high-risk AI) | manufacturing AI 정확한 정의 |
| **IEC 61508** (functional safety) | ML 모델 functional safety 확장 |
| **ISO/TS 16949** (자동차 QMS) | AI 검증 프로세스 추가 |
| **W3C PROV** (lineage) | manufacturing decision provenance |
| **Hugging Face Model Cards** | manufacturing-specific 필드 |

→ 각 매핑은 RFC v0.2-v1.0 작업 시 부속서로 published. **"기존 표준과 호환된다"는 메시지가 표준 채택의 핵심 신호**.

---

## 5. 왜 이게 결정적 moat인가

```
일반 AI vendor:
  AI 모델 1개 만듦
  "우리 모델 좋아요" — 자랑
       ↓
  벤치마크 표준 없음 → 비교 불가
       ↓
  OEM이 "그럼 알아서 골라" 하다 큰 vendor가 채택됨
       ↓
  AIAS 의무화 후 → vendor가 매년 ₩X억 audit 비용 + 우리한테 사용료 지불

aoi-sentinel (우리):
  AI 모델 + AIAS 표준 정의
  "이 표준으로 검증 가능합니다 + 다른 vendor도 통과해야 비교됩니다"
       ↓
  OEM이 "이 표준이 정답이다" 인식
       ↓
  AI vendor 모두 우리 표준 통과해야 OEM 납품 가능
       ↓
  우리 = 표준 권력
       + 우리 모델 = reference impl (가장 자연스럽게 통과)
       + audit 서비스 / 인증 비용 추가 매출
```

**Buffett 렌즈**: 모델 카피 18개월. 표준 정의·거버넌스·신뢰 카피 18년.

---

## 6. 채택 timeline — 6년 plan

| 시점 | 단계 | 핵심 액션 |
|-----|------|---------|
| **Year 1** (현재) | RFC 초안 | AIAS RFC v0.1 published, 우리 reference 구현 운영 |
| **Year 2** | 첫 외부 채택 | 1-2 외부 vendor 또는 customer가 AIAS 형태로 데이터 출력 |
| **Year 3** | 표준 body 접촉 | ISO/IEC 42001 working group 합류, NIST AI RMF profile 제안 |
| **Year 4** | 한국 정부 인용 | 한국 AI 기본법 시행규칙에 AIAS-style 요건 등장 |
| **Year 5** | 국제 표준 등록 | ISO/IEC TC SC 42 working group 정식 RFC |
| **Year 6** | 의무화 시작 | EU AI Act high-risk audit framework로 인용 채택 |
| **Year 7+** | 의무 표준 | AIAS 통과 못 하면 OEM 납품 불가능 |

이 시점 = **₩100B 회사 진입 임계점**.

---

## 7. 즉시 실행 — 이번 분기

1. **`aias-spec`** 별도 repo placeholder 생성 (도메인 점유)
2. **`aoi-common-spec`에 AIAS 메타데이터 부속 RFC** 초안 — 검사 결과에 AI 결정 lineage 첨부 가능하게
3. **첫 MMC (Model Card) 작성** — 우리 stage 1 classifier 모델 / Mamba RL 모델 대상, 표준 reference로 사용
4. **NIST AI RMF / ISO 42001 매핑 문서** — 영업 자료 부록
5. **다음 anchor 미팅에 한 슬라이드 추가** — "5년 후 의무화될 표준을 저희가 정의 중입니다"

---

## 8. 영업 메시지 갱신

기존:
> "저희는 vendor-neutral AOI 솔루션입니다."

신규:
> **"저희는 AI 검사 솔루션을 팝니다. 그리고 그 솔루션이 따라야 할 audit 표준 자체도 정의하고 있습니다. 5년 후에는 이 표준을 통과하지 못하는 AI는 OEM 납품이 불가능해질 겁니다 — EU AI Act, ISO 42001, 한국 AI 기본법 흐름이 그쪽입니다. 저희와 시작하시면 그 시점에 자동으로 통과 상태입니다."**

이 한 마디로 **시간 차원의 차별화**가 만들어짐 — 다른 vendor는 "오늘 좋은 솔루션"이지만 우리는 **"5년 후에도 살아있는 솔루션"**.

---

## 9. 한 줄 만트라

> **"AI 모델은 commodity, AI audit 표준은 권력."**

> **"우리는 AI를 파는 게 아니라, AI가 따라야 할 표준을 정의한다."**

---

## 10. 우리 strategic brief에서의 위치

기존 §7.1 (B+ ₩100B path)에 추가 트리거:

```
[AIAS adoption — meta-layer standard]
├─ 12개월: AIAS RFC v0.1 published
├─ 24개월: 첫 AIAS-conformant 외부 모델 등장
├─ 36개월: ISO/IEC 42001 working group 합류
├─ 48개월: 한국 AI 기본법 시행규칙에 AIAS 인용
└─ 60개월: EU AI Act high-risk audit framework로 채택
        → 그 시점, 모든 제조 AI는 AIAS 통과해야 함
        → 우리 = audit body + reference impl
        → ₩100B 회사 진입
```

---

## 참고 — 비슷한 메타 표준의 역사

| 메타 표준 | 정의자 | 결과 |
|----------|--------|------|
| **HACCP** (식품 안전) | 1960년대 NASA + Pillsbury | 전 세계 식품 산업 의무 |
| **Six Sigma** (품질 관리) | Motorola 1986 | 모든 산업 표준 어휘 |
| **OWASP Top 10** (웹 보안) | OWASP 2003 | 웹 보안 표준 어휘 |
| **MLflow Model Registry** | Databricks 2018 | ML 운영 표준 도구 |
| **HuggingFace Model Cards** | 2018 | AI 모델 메타데이터 표준 |

공통 패턴:
1. 작은 조직이 표준 정의
2. 무료 reference 구현
3. 큰 조직이 채택 시작
4. 정부·규제가 인용
5. 의무화

**우리 timeline은 정확히 이 패턴**. 인내심 + 실행력.

---

*Last updated: 2026-05-09*
*Companion: [standardization_methodology.md](standardization_methodology.md), [strategic_brief.md](strategic_brief.md)*
