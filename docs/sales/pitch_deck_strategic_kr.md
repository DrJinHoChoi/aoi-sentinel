# aoi-sentinel — 전략 발표자료 (비전 / 표준화)

> **15분 / 15장 / 임원·투자자·표준 body 미팅 표준.**
>
> 기존 [pilot 발표자료](pitch_deck_kr.md)가 **"6개월 무료 박스 빌려쓰세요"** (구매팀 대상)라면, 이건 **"5년 후 의무화될 AI 표준을 우리가 정의 중"** (CEO·CTO·VC·표준 body 대상).

각 슬라이드 구조: 🎯 비전 메시지 + 3개 전략 포인트 + 🔧 증거.

---

## 슬라이드 1 — 표지

### 🎯 5년 후, 모든 제조 AI는 누군가가 정의한 표준을 통과해야 합니다. 그게 누가 될지 정해지는 게 지금입니다.

```
              aoi-sentinel
        제조 AI 표준의 reference 구현
            DrJinHoChoi · 2026
```

**🔧 근거**: EU AI Act (2024 발효), ISO/IEC 42001 (2023 published), 한국 AI 기본법 (2025-2026 시행) — 모두 2027-2030 의무 audit 수렴.

---

## 슬라이드 2 — 숨은 문제

### 🎯 모든 AOI 제조사가 AI를 자랑하는데, 누가 더 좋은지 비교할 방법이 없습니다

- Mycronic AI vs Koh Young AI vs 인하우스 — **벤치마크 표준 0개**
- OEM이 vendor 채택할 때 **객관적 비교 불가능** → 결국 큰 vendor가 채택됨
- AI 결정의 **lineage·audit·인증 표준 없음** → 사고 발생 시 책임 추적 불가

→ AI는 도처에 있지만 **거버넌스는 비어있음**.

**🔧 근거**: 현재 어느 vendor도 AI 모델의 escape rate / training data lineage / safety contract를 표준 포맷으로 공개하지 않음.

---

## 슬라이드 3 — 규제는 inevitability

### 🎯 2027년부터 OEM은 AI 솔루션 audit을 의무화해야 합니다. 그 audit 표준이 아직 없습니다.

```
2024  EU AI Act 발효  ─────────────────────────┐
2025  한국 AI 기본법 통과                       │
2026  ISO/IEC 42001 채택 가속                  ├─→ 의무 audit 도래
2027  EU AI Act high-risk AI 단계적 시행        │
2028  미국 NIST AI RMF 정부조달 의무화           │
2029  한국 AI 기본법 시행규칙 분야별 인증절차    │
2030  글로벌 산업 AI audit 의무화 ──────────────┘
```

→ **"누가 audit 표준을 만드는가"** 가 2026년에 정해짐.

**🔧 근거**: EU AI Act Annex III "high-risk AI" 분류에 manufacturing inspection 포함됨.

---

## 슬라이드 4 — 두 Layer 게임

### 🎯 데이터 표준은 substrate. 진짜 moat는 그 위 메타 표준입니다.

```
[Layer 0]  Manufacturing standards FOR AI         ← 진짜 ₩100B moat
            AIAS · AISC · MMC
            "AI 자체를 audit·비교·인증하는 표준"
            적용: 모든 제조 AI

                  ↑↑↑ 그 위에 얹힘 ↑↑↑

[Layer 1]  AI for manufacturing standards         ← 우리 현재 위치
            AICS (검사) · CCS (가공)
            "검사 결과를 vendor 무관하게 표현"
```

→ 우리는 **두 layer 모두 정의** 중. 다른 vendor는 layer 1조차 없음.

**🔧 근거**: aoi-common-spec RFC v0.1 (BSD-3, GitHub published) + AIAS RFC 작성 중.

---

## 슬라이드 5 — Buffett 렌즈

### 🎯 AI 모델은 commodity. AI audit 표준은 권력.

| 자산 | 카피 시간 |
|------|----------|
| AI 모델 가중치 | **18개월** |
| 어댑터 코드 | **18개월** |
| 학술 논문 | **24개월** |
| **공개 표준 + 거버넌스** | **18년** |
| **다중 vendor 누적 라벨** | **시간만이 만듦** |

대형 vendor는 우리 모델 18개월 안에 따라잡습니다. 표준은 못 따라잡습니다.

**🔧 근거**: Bluetooth (1998 → 2010 default, 12년), HTTPS (1995 → 2018 강제, 23년), HACCP (1960s → 글로벌 의무, 30년).

---

## 슬라이드 6 — 우리가 정의할 8가지 표준

### 🎯 AI 결정의 audit · 운영 · lifecycle 전체 layer를 표준화합니다

| Layer | 표준 | 역할 |
|-------|------|------|
| **Audit** | AIAS, AISC, MMC | 결정 lineage, 안전 contract, model card |
| **운영** | Drift, Benchmark, Federated | 배포 후 monitoring, vendor 비교, 데이터 공유 없는 학습 |
| **Lifecycle** | Lineage, Decommissioning | 데이터→배포→retire 전체 추적 |

기존 ISO/IEC 42001 (AI Mgmt), NIST AI RMF, IEC 61508 (functional safety) 위에 **manufacturing extension**.

**🔧 근거**: 8개 표준 후보 상세는 [docs/ai_manufacturing_standards.md](../ai_manufacturing_standards.md).

---

## 슬라이드 7 — AIAS (AI Audit Schema) 상세

### 🎯 모든 AI 결정에 6개 메타데이터를 강제로 첨부 — 5년 후 OEM 조달 의무 요건

```yaml
aias_version: "0.1"
decision_id: "uuid-..."
model:
  version: "v1.2.3"
  weights_hash: "sha256:..."
  training_data_hash: "sha256:..."
context:
  cost_matrix: { c_escape: 50, c_false_call: 1 }
  safety_constraint: { metric: "escape_rate", limit: 0.001 }
decision:
  action: "PASS"
  confidence: 0.94
human_override:
  occurred: false
```

→ AI 사고 발생 시 **어느 학습 batch까지 책임 추적 가능**. 보험사·정부 audit 친화적.

**🔧 근거**: W3C PROV (lineage 표준) + HuggingFace Model Cards를 manufacturing vertical로 합성.

---

## 슬라이드 8 — 6년 채택 timeline

### 🎯 표준 정의는 1년. 의무화까지 7년. 지금 시작 안 하면 늦습니다.

```
Year 1   AIAS RFC v0.1 published          ← 우리 (현재)
Year 2   첫 외부 vendor 채택
Year 3   ISO/IEC 42001 working group 합류
Year 4   한국 AI 기본법 시행규칙 인용
Year 5   ISO/IEC TC SC 42 정식 RFC
Year 6   EU AI Act audit framework 채택
Year 7+  의무화 → 우리 표준 통과 못 하면 OEM 납품 X
```

이 시점 = **₩100B 회사 진입 임계점**.

**🔧 근거**: 비슷한 메타 표준 평균 7-10년 (HACCP 30년, OWASP 10년, HuggingFace Model Cards 5년 → de-facto).

---

## 슬라이드 9 — 역사적 패턴 — 왜 작동하나

### 🎯 작은 조직이 정의한 표준이 글로벌 의무가 되는 5단계 패턴

```
1. 작은 조직이 표준 정의                    ← 우리 (Year 1)
2. 무료 reference 구현                       ← aoi-sentinel
3. 큰 조직이 채택 시작                       ← Year 2
4. 정부·규제가 인용                          ← Year 4-5
5. 의무화                                    ← Year 6-7+
```

**검증된 사례**:
- HACCP (식품 안전): NASA + Pillsbury → 글로벌 의무 30년
- OWASP Top 10: 작은 비영리 → 웹 보안 표준 어휘
- HuggingFace Model Cards: 학술 연구 → AI 메타데이터 default

**🔧 근거**: 모든 사례에서 정의자(founding org)가 reference impl + 거버넌스 owner.

---

## 슬라이드 10 — 이미 가진 traction

### 🎯 추상적 비전이 아닙니다 — 표준 + 코드 + 실험 결과 모두 공개됐습니다

| 자산 | 위치 | 진척 |
|------|------|------|
| **AICS RFC v0.1** | github.com/DrJinHoChoi/aoi-common-spec | ✅ Published, BSD-3 |
| **aoi-sentinel reference 구현** | github.com/DrJinHoChoi/aoi-sentinel | ✅ 50+ tests pass |
| **CCS RFC** | github.com/DrJinHoChoi/cnc-common-spec | ✅ Placeholder repo |
| **Mamba RL 학습 검증** | VisA 4,416장 PCB, A100 80GB | ✅ Escape=0 60 iter |
| **3-vendor 어댑터** | Saki / Koh Young / generic_csv | ✅ 작동 |

**현재 부족한 것 = 첫 anchor 고객 1곳.**

**🔧 근거**: 모든 결과 GitHub 검증 가능, BSD-3 라이선스, RFC v0.1.

---

## 슬라이드 11 — VisA 실험 — 시스템 작동 직접 증거

### 🎯 lab에서 60번 학습 — 놓친 결함 0건, 비용 5.6배 ↓

```
┌──────────────────┬──────────────────┐
│ Escape per iter  │   비용 추세       │
│ 0  0  0  0  0    │  1,280 → 230     │
│ (60 iter 동안 0) │  (5.6× 감소)      │
└──────────────────┴──────────────────┘
┌──────────────────┬──────────────────┐
│ λ (안전 dual)     │  정책 자동 진화   │
│ 자동 조절됨       │  ESCALATE→DEFECT  │
└──────────────────┴──────────────────┘
```

→ **Lagrangian PPO escape contract 수학적 작동 입증**. AISC (AI Safety Contract) 표준의 reference 구현 동작.

**🔧 근거**: NVIDIA A100 80GB, VisA 4,416 imgs, 상세 [pilot_evidence_kr.md](pilot_evidence_kr.md).

---

## 슬라이드 12 — TAM — 시장 규모

### 🎯 우리 시장은 manufacturing AI 전체. 한국·아시아 우선 진입.

| 도메인 | 2024 | 2030E | 우리 적용 |
|--------|------|-------|----------|
| 자동차 전장 SMT (현재 wedge) | $20B | $40B | ✅ 즉시 |
| AI 서버 / HBM 보드 | $80B | $250B | ⏳ Year 2 |
| 휴머노이드 부품 | $2B | $30B | ⏳ Year 2 |
| EV 배터리 BMS | $30B | $80B | ⏳ Year 3 |
| CNC 가공 | $20B | $40B | ⏳ Year 2 |
| **합계 직접 가능** | **~$150B** | **~$440B** | — |

같은 박스, 같은 표준, 다른 도메인 — **하나의 표준이 모든 manufacturing AI를 통일**.

**🔧 근거**: Mordor Intelligence, IDC, 골드만삭스 humanoid 보고서 (2024-2025).

---

## 슬라이드 13 — Stakeholder ecosystem

### 🎯 표준은 혼자 못 만듭니다 — 5종류 이해관계자 동시 engage

| Stakeholder | 한국 | 글로벌 | 우리가 줄 가치 |
|-------------|------|--------|----------------|
| **고객 (OEM/Tier-1)** | 현대모비스, LG이노텍 | Continental, Bosch | ROI 데이터 |
| **벤더 (장비)** | (한국 적음) | Saki, Koh Young, Mycronic | 플랫폼 통합 |
| **표준 body** | KSA, KETI, TTA | IPC, SEMI, IEC, ISO | 잘 만든 RFC |
| **연구계** | 서울대, KAIST | MIT, ETH | 데이터셋 |
| **정부** | 산업부, 중기부 | EU IndustryX, NIST | 정책 부합 |

→ Phase A (1-2년) 고객·벤더 집중. Phase B 연구계·표준 body. Phase C 정부.

**🔧 근거**: 상세 playbook [docs/standardization_methodology.md](../standardization_methodology.md).

---

## 슬라이드 14 — 거버넌스 — 의도적 자기 견제

### 🎯 표준이 우리만의 것이 되면 표준이 아닙니다. Multi-editor 거버넌스로 의도적 자기 견제.

```
v0.x  단독 editor (DrJinHoChoi)                    ← 현재
v1.0+ Multi-editor 위원회 (자동 전환)
       자격: AICS-conformant adapter 구현 또는 production 배포 조직
       의사결정: simple majority + 14일 comment window
       editor 수: 5-7명
v2.0+ 비영리 재단 또는 컨소시엄으로 분리           (Year 3+)
       Bluetooth SIG 모델 — 우리 = founding member 중 하나
```

→ **표준은 Bluetooth가 노키아 것이 아니듯, AIAS도 우리만의 것이 아님**. 그러나 우리가 정의·운영했으므로 reference impl + audit 서비스에서 매출 발생.

**🔧 근거**: AICS RFC-001 §11에 이미 명시 + BSD-3 라이선스로 코드 공유.

---

## 슬라이드 15 — 마지막 메시지

### 🎯 AI 모델은 commodity. AI 표준은 권력.
### 🎯 우리는 AI를 파는 게 아니라, AI가 따라야 할 표준을 정의합니다.

```
지금 시작하는 사람이 5년 후 표준 owner.
                    ↓
                ₩100B 회사
```

**현재 부족한 것**:
- 첫 anchor 고객 1곳 (모든 표준화의 전제)
- Phase A 자본 (12-18개월 운영비)
- 표준 body 컨택 (KSA, IPC Korea, KETI)

**필요하시면 같이 만들 수 있습니다.**

```
담당     DrJinHoChoi (최진호)
GitHub   github.com/DrJinHoChoi
```

---

> **"제조 AI 표준 — 5년 후 누가 정의하는가. 정해지는 게 지금입니다."**

---

## 발표 운영 가이드

### 청중별 강조 슬라이드

| 청중 | 핵심 슬라이드 |
|------|--------------|
| **OEM CEO/CTO** | 1, 3 (규제 inevitability), 6 (8가지 표준), 11 (실험 증거), 15 |
| **VC / 투자자** | 1, 5 (moat), 8 (timeline), 12 (TAM), 15 |
| **표준 body** | 4 (두 layer), 7 (AIAS 상세), 9 (역사적 패턴), 13 (ecosystem), 14 (거버넌스) |
| **정부 / 규제** | 3 (규제 inevitability), 7 (AIAS), 13 (한국 stakeholder), 14 (자기 견제) |

### 시간 배분 (15분)

| 단계 | 슬라이드 | 시간 |
|------|---------|------|
| 비전 (문제 + 게임) | 1-5 | 4분 |
| 어떻게 (8 표준 + 상세) | 6-9 | 4분 |
| 증거 + 시장 | 10-12 | 3분 |
| Ecosystem + 거버넌스 | 13-14 | 2분 |
| 클로징 | 15 | 2분 |

### 외울 한 문장

> **"AI 모델은 commodity, AI 표준은 권력. 5년 후 의무화될 표준을 지금 우리가 정의 중입니다."**

---

## PPTX 빌드

```bash
node docs/sales/build_pitch_deck_strategic.js
```

PowerPoint에서 열어 한국어 폰트 렌더링 확인 후 미팅 사용.

---

*Last updated: 2026-05-09. 임원·투자자·표준 body 대상 비전 deck.*
*Companion: [pitch_deck_kr.md](pitch_deck_kr.md) (구매팀·공장장 대상 tactical deck)*
