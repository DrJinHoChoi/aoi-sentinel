# 제조 표준화 방법론

> **"표준은 RFC를 쓰는 게 아니라, 5개 회사가 동시에 채택하기 시작할 때 만들어진다.
> 우리 일은 그 5개를 만드는 것이다."**

이 문서는 우리가 만들고 있는 기술(AICS, CCS, Lagrangian PPO 안전 제약, vendor 무관 어댑터 SDK)을 **사실상 표준 → 정식 산업 표준 → 의무 채택 표준**으로 단계적으로 격상시키기 위한 실행 가능한 playbook이다.

읽는 시점: 매분기 1회. 다음 분기 액션 결정 전.

---

## 0. 두 단계 — 데이터 표준 vs 메타 표준

이 문서가 다루는 표준화는 **Layer 1 (데이터 표준 — AICS / CCS)**.

그 위에 **Layer 0 (메타 표준 — manufacturing AI 자체의 audit·인증·비교)**가 있다. 둘 다 우리가 정의해야 진짜 moat가 만들어진다.

```
[Layer 0]  Manufacturing standards FOR AI         ← 진짜 ₩100B moat
            AIAS / AISC / MMC — 모든 제조 AI에 적용
            상세: docs/ai_manufacturing_standards.md
[Layer 1]  AI for manufacturing standards         ← 이 문서
            AICS / CCS — 검사·가공 데이터 표준
```

Layer 1 모멘텀이 잡히면 Layer 0 작업 시작 (Year 2 Q1+). 두 layer 모두 같은 거버넌스·라이선스·multi-editor 모델 공유.

---

## 1. 표준화 6-Layer Stack

표준은 한 층이 아니다. 각 layer마다 owner / 타임라인 / 거버넌스가 다르다.

```
┌──────────────────────────────────────────────────────────┐
│ Layer 6  법규 표준 (KS / IEC / ISO 정식 등록)             │  5-10년
├──────────────────────────────────────────────────────────┤
│ Layer 5  인증 표준 (IEC 61508 functional safety,         │  3-5년
│          IEC 62443 cybersec, K-Mark)                     │
├──────────────────────────────────────────────────────────┤
│ Layer 4  산업 표준 (IPC, SEMI 정식 RFC 채택)              │  2-3년
├──────────────────────────────────────────────────────────┤
│ Layer 3  프로세스 표준 (SHADOW→ASSIST→AUTONOMOUS,         │  1-2년
│          mode 전환 trigger, 안전 게이트 KPI)              │
├──────────────────────────────────────────────────────────┤
│ Layer 2  알고리즘 표준 (Constrained MDP, escape ≤ ε       │  1년
│          contract, 감사 가능성)                           │
├──────────────────────────────────────────────────────────┤
│ Layer 1  데이터 표준 (AICS / CCS — RFC v0.1) ← 현재 위치  │  6개월
└──────────────────────────────────────────────────────────┘
```

**현재 위치**: Layer 1 진입. 위로 올라갈수록 시간·정치 비용 ↑, 영향력·moat ↑.

---

## 2. 4-Phase 타임라인

표준은 점진적으로 만들어진다. Bluetooth(1998 발표 → 2010 default), USB(1996 → 2008) 모두 10년+.

### Phase A — De facto adoption (현재~12개월)

법적 효력 0, 시장 권력 ↑. **자발적 채택 모멘텀**이 핵심.

| 액션 | 결과물 |
|------|-------|
| 5-10개 고객사에 작동하는 제품 deploy | 케이스 스터디 |
| 공개 RFC + reference 구현 ✅ | 개발자 신뢰 |
| 첫 외부 contributor 모집 | 거버넌스 진정성 |
| 산업 컨퍼런스 발표 (IPC APEX, SEMI Korea) | 가시성 |
| 벤더 라이브러리 연동 (Saki SDK adapter 등) | 채택 모멘텀 |

### Phase B — Working group + 형식화 (12-30개월)

**한국 주도 + 국제 연동** 두 트랙 동시.

| 트랙 | 액션 |
|------|------|
| 한국 | KSA(한국표준협회) 통해 표준 RFC 비공식 제출 |
| 한국 | KETI(전자기술연구원) 또는 KIMM(기계연구원) 공동연구 |
| 한국 | 자동차산업협회 / 반도체산업협회 워킹그룹 발표 |
| 국제 | IPC-2581 / IPC-CFX 인접 워킹그룹 참여 |
| 국제 | SEMI E142 (장비 통신) 등에 AICS 매핑 |
| 국제 | OPC Foundation (OPC-UA) Companion Spec 검토 |

### Phase C — 정식 표준 등록 (30-60개월)

| 표준 body | 타깃 등록 |
|-----------|----------|
| **KS X** (정보통신) | KS X "AOI 검사 결과 데이터 교환 표준" |
| **IPC** | IPC 신규 standard 또는 IPC-2581 확장 |
| **SEMI** | SEMI E "AOI Inspection Schema" |
| **IEC TC 65** | 산업 자동화 시스템 통신 |
| **ISO 16100** | 제조시스템 통합 시스템 적용 |

### Phase D — 의무 adoption (60-120개월)

**시장 강제력 발동**:
- OEM 조달 요건 ("AICS 호환 솔루션만 입찰 가능")
- ISO/TS 16949 (자동차 QMS)에 embedded
- 정부 스마트팩토리 R&D 과제 요건
- KOSHA 안전 인증 절차에 반영

이 시점 = **표준 owner = ₩100B 회사 path 진입**.

---

## 3. 5-Stakeholder Ecosystem

표준은 혼자 못 만든다. 5종류의 이해관계자 동시 engage.

| Stakeholder | Korea | Global | 우리가 줄 가치 | 그들에게서 받을 것 |
|-------------|-------|--------|----------------|---------------------|
| **고객 (OEM/Tier-1)** | 현대모비스, LG이노텍, 삼성전기 | Continental, Bosch | ROI 데이터, 도입 사례 | 채택, 요구사항 피드백, 추천 |
| **벤더 (장비)** | (한국 vendor 적음) | Saki, Koh Young, Mycronic | 플랫폼 통합 채널 | adapter 구현, export 기본화 |
| **표준 body** | KSA, KETI, KIMM, TTA | IPC, SEMI, IEEE, IEC, ISO | 잘 만든 RFC | 정식 등록, 권위 |
| **연구계** | 서울대 / KAIST / 포스텍 (제조 IT) | MIT, ETH, NUS | 데이터셋, 챌린지 | 논문, 학문적 신뢰성 |
| **정부** | 산업부, 중기부 | EU IndustryX, US NIST | 정책 부합 솔루션 | 보조금, 의무화 |

**우선순위**: Phase A에서는 **고객 + 벤더** 집중. Phase B에서 **연구계 + 표준 body** 추가. Phase C에서 **정부** 마지막.

---

## 4. 분기별 액션 (24개월 plan)

### Q1-Q2 — Phase A 시작

- ✅ AICS RFC v0.1 published
- [ ] 첫 anchor 라인에 박스 deploy
- [ ] 외부 contributor 1명 (vendor 측 엔지니어 또는 customer 통합 엔지니어)
- [ ] AICS 사용 장려 — workshop, SDK 자료
- [ ] **IPC APEX 2027 발표 abstract 제출** (마감 보통 11월)

### Q3-Q4 — Phase A 강화

- [ ] 5개 사이트 deploy → 케이스 스터디 5건
- [ ] Saki / Koh Young 한국지사와 비공식 ML
- [ ] **AICS Conformance Test Suite v1.0** — 외부에서 검증 가능
- [ ] **KSA 표준화 위원회에 RFC 비공식 제출**

### Year 2 Q1-Q2 — Phase B 시작

- [ ] AICS v1.0 — multi-editor 거버넌스 전환 (RFC §11 약속 이행)
- [ ] KETI 또는 KIMM 공동연구 ML
- [ ] 산업체 워킹그룹 발족 (자동차산업협회 산하)
- [ ] **CCS RFC v0.1** (CNC 가공 표준)

### Year 2 Q3-Q4 — Phase B 가속

- [ ] 국제 표준 body 첫 접촉 (IPC, SEMI Korea)
- [ ] 한국 정부 R&D 과제 컨소시엄 참여 (산업부)
- [ ] 학술 논문 1-2편 (NeurIPS / ICCV / IEEE T-II)
- [ ] **KS X 표준 등록 시도**

---

## 5. 거버넌스 — 표준이 "우리 것"이 안 되도록 의도적 설계

**핵심 역설**: 표준은 "우리가 정의했지만 우리만의 것이 아닐 때" 권력을 가진다.

### Multi-editor 모델 (이미 RFC-001 §11에 명시)

```
v0.x:  단독 editor (DrJinHoChoi)
v1.0+: multi-editor 위원회
  자격:        AICS-conformant adapter 구현 또는 production 배포 조직
  의사결정:    simple majority + 14일 comment window for breaking change
  editor 수:   5-7명 (의사결정 가능 + 다양성)
```

### Public artifacts

- **GitHub repo** (public, BSD-3) — 누구나 PR
- **공개 mailing list** — 결정 투명성
- **분기 회의록 published** — 정치적 신뢰
- **Conformance test suite** — 자체 검증 가능

### 조직적 분리 (Year 3+)

장기적으로 표준 거버넌스를 **별도 비영리 재단 또는 컨소시엄**으로 분리. **Bluetooth SIG 모델**:
- 우리 회사 = founding member 중 하나
- 단독 owner X
- 회비/재정은 위원회에서 운영

---

## 6. 인접 표준과의 관계 — "기존 표준 위에 얹기"

새 표준을 처음부터 만들기보다 **기존 표준의 확장 또는 맵핑**이 채택 빠름.

| 기존 표준 | 우리와의 관계 |
|-----------|---------------|
| **IPC-2581** | PCB 디자인 데이터 표준. AICS는 그 검사 결과 layer로 자연스러움 |
| **IPC-CFX** (Connected Factory Exchange) | wire protocol 후보 |
| **OPC-UA Companion Specs** | 장비 통신 표준. AICS = OPC-UA 위 application layer |
| **MTConnect** | 가공 데이터 표준 (CNC). CCS와 직접 매핑 가능 |
| **STEP-NC** | CAM-NC 표준. CCS의 drawing layer 호환 |
| **SEMI E142** | 장비 데이터 통신. SMT 라인 통합 시 |

**액션**: AICS RFC v0.2 작업 시 IPC-CFX 및 OPC-UA 매핑 표 포함 → 채택 모멘텀 ↑.

---

## 7. 위험 + 회피 전략

| 위험 | 회피 방법 |
|------|----------|
| 대형 vendor가 자체 표준 발표 | 우리가 먼저 모멘텀 — Phase A 12개월 내 5+ 채택 |
| 정부 표준이 반대 방향 | 정부 R&D 과제 일찍 합류 (Phase B Q3) |
| 거버넌스가 우리에 갇힘 | 의도적 multi-editor 전환 + 공개 회의 |
| 학술적 신뢰성 부족 | NeurIPS/CVPR/T-II 1-2편 (Year 2) |
| 호환 불완전 | conformance test suite + reference impl 항상 동기화 |
| Korean-only 갇힘 | Phase B에 IPC/SEMI 동시 진입 |

---

## 8. KPI — 표준화 진척 측정

| 지표 | Year 1 목표 | Year 2 목표 | Year 3 목표 |
|------|------------|------------|------------|
| AICS 채택 사이트 수 | 3-5 | 15-20 | 50+ |
| 외부 adapter 구현 | 1 | 3-5 | 10+ |
| Multi-editor 수 | 1 (us) | 3-5 | 5-7 (위원회) |
| 학술 인용 | 0 | 5-10 | 30+ |
| 정부 R&D 과제 | 0 | 1 (참여) | 2-3 (주관 또는 핵심) |
| 표준 body 정식 RFC | 0 | KSA 1건 | IPC/SEMI 1-2건 |
| 정식 KS / IEC 등록 | — | — | 1건 (KS X) |

---

## 9. 인접 도메인 확장 (CCS 등)

AOI 표준 모멘텀이 잡히면 **같은 playbook을 CNC, 3D AOI, ICT, FCT로 복제**:

```
[AOI 표준 (AICS)]    Phase A → Phase B → Phase C → Phase D
                                  ↓ Year 2
                        [CNC 표준 (CCS)]    Phase A → Phase B → ...
                                              ↓ Year 4
                                     [SPI 표준]     Phase A → ...
                                                       ↓ Year 6
                                              [통합 제조 데이터 layer]
```

각 표준은 독립적이지만 **거버넌스 모델, 라이선스, 멀티-에디터 룰은 공유**. 첫 표준의 신뢰가 다음 표준 채택을 가속.

**Year 5 비전**: AICS + CCS + SPI + ... = **"제조 검사·가공 데이터의 통합 표준 layer"** = 우리가 [전략 브리프 §7](strategic_brief.md#7-10년-후--세-갈래-길) B+ path (₩100B 회사) 진입.

---

## 10. 한 줄 만트라

> **"표준은 RFC를 쓰는 게 아니라, 5개 회사가 동시에 채택하기 시작할 때 만들어진다.
> 우리 일은 그 5개를 만드는 것이다."**

---

## 다음 액션 — 이번 분기 끝나기 전에

1. [ ] **첫 anchor LOI 1건** ← 모든 표준화의 전제
2. [ ] **AICS Conformance Test Suite v0.1** (코드 1일, 신뢰 ↑↑)
3. [ ] **IPC APEX 2027 발표 abstract 제출**
4. [ ] **KSA 또는 KETI 비공식 미팅 1건** — RFC 피드백
5. [ ] **첫 외부 contributor 모집 글** GitHub Discussions에 게시

---

## 참고 — 성공한 표준의 역사적 패턴

| 표준 | 발표 | De facto | 정식 표준 | 의무화 |
|------|-----|----------|-----------|-------|
| **TCP/IP** | 1974 | 1985 | RFC (IETF) 1981 | 2000년대 (정부 조달) |
| **HTTP** | 1991 | 1995 | IETF RFC 2616 (1999) | 사실상 default 2005 |
| **Bluetooth** | 1998 | 2002 | IEEE 802.15.1 (2002) | 모바일 표준 2005-2010 |
| **USB** | 1996 | 2002 | USB 2.0 표준 (2000) | PC 표준 2008 |
| **HTTPS / TLS** | 1995 | 2010 | IETF RFC 8446 (2018) | 모든 사이트 강제 (2018, Chrome) |

공통 패턴:
1. **Founding company + open spec** (우리도 그 위치)
2. **Multi-editor 거버넌스 전환** (Year 3-5)
3. **정식 표준 body 등록** (Year 5-10)
4. **시장 / 정부 강제 adoption** (Year 7-15)

**우리 timeline은 industry standard 패턴과 일치** — 인내심 + 실행력만 있으면 됨.

---

*Last updated: 2026-05-09*
*Next quarterly review: Year 1 Q2 끝.*
