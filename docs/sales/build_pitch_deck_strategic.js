// Build pitch_deck_strategic_kr.pptx — vision/standardization deck for
// executives, investors, and standards bodies (separate from the tactical
// 6-month pilot deck).

const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "DrJinHoChoi";
pres.title = "aoi-sentinel — 전략 발표자료 (비전·표준화)";

// Color palette: Midnight Executive (consistent with tactical deck)
const NAVY    = "1E2761";
const ICE     = "CADCFC";
const ACCENT  = "F96167";
const GOLD    = "F9E795";  // accent for vision/strategic moments
const INK     = "1A1A2E";
const MUTED   = "6B7280";
const PAGE_BG = "FFFFFF";
const OK      = "2DA44E";
const NO      = "CF222E";

const FONT_HEADER = "Malgun Gothic";
const FONT_BODY   = "Malgun Gothic";

function addSlideHeader(slide, headline, slideNum) {
  slide.addText(`${slideNum} / 15`, {
    x: 9.0, y: 0.18, w: 0.85, h: 0.3,
    fontFace: FONT_BODY, fontSize: 10, color: MUTED, align: "right", margin: 0,
  });
  slide.addText("🎯 " + headline, {
    x: 0.5, y: 0.4, w: 9.0, h: 1.0,
    fontFace: FONT_HEADER, fontSize: 22, bold: true, color: NAVY,
    align: "left", valign: "top", margin: 0,
  });
}

function addTechNote(slide, note) {
  slide.addText("🔧  " + note, {
    x: 0.5, y: 5.10, w: 9.0, h: 0.35,
    fontFace: FONT_BODY, fontSize: 10, italic: true, color: MUTED,
    align: "left", margin: 0,
  });
}

// ---- Slide 1: Title -----------------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: NAVY };

  slide.addText("제조 AI 표준", {
    x: 0.5, y: 1.0, w: 9.0, h: 1.0,
    fontFace: FONT_HEADER, fontSize: 56, bold: true, color: "FFFFFF",
    align: "center", margin: 0,
  });
  slide.addText("5년 후, 누가 정의하는가?", {
    x: 0.5, y: 2.0, w: 9.0, h: 0.7,
    fontFace: FONT_HEADER, fontSize: 28, color: GOLD, italic: true,
    align: "center", margin: 0,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 3.0, y: 3.2, w: 4.0, h: 0.05, fill: { color: ACCENT }, line: { color: ACCENT },
  });

  slide.addText("aoi-sentinel  —  제조 AI 표준의 reference 구현", {
    x: 0.5, y: 3.5, w: 9.0, h: 0.5,
    fontFace: FONT_BODY, fontSize: 18, color: ICE,
    align: "center", margin: 0,
  });

  slide.addText("DrJinHoChoi · 2026", {
    x: 0.5, y: 5.0, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 12, color: ICE,
    align: "center", margin: 0,
  });
}

// ---- Slide 2: Hidden problem -------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "모든 AOI 제조사가 AI를 자랑하는데, 누가 더 좋은지 비교할 방법이 없습니다", 2);

  // Big visual: vendor logos (placeholder text) with ?
  const vendors = ["Mycronic AI", "Koh Young KSMART", "Saki AI", "기타 vendor"];
  vendors.forEach((v, i) => {
    const x = 0.5 + i * 2.3;
    slide.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.7, w: 2.0, h: 1.0, fill: { color: "F8FAFC" }, line: { color: NAVY, width: 1 },
    });
    slide.addText(v, {
      x, y: 1.7, w: 2.0, h: 1.0,
      fontFace: FONT_HEADER, fontSize: 13, bold: true, color: INK,
      align: "center", valign: "middle", margin: 0,
    });
  });

  // VS in middle
  slide.addText("vs  vs  vs", {
    x: 0.5, y: 2.85, w: 9.0, h: 0.4,
    fontFace: FONT_HEADER, fontSize: 18, italic: true, color: MUTED,
    align: "center", margin: 0,
  });

  // Big question
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 3.45, w: 9.0, h: 1.0, fill: { color: ACCENT }, line: { color: ACCENT },
  });
  slide.addText("벤치마크 표준 없음 → 객관적 비교 불가능 → 결국 큰 vendor 채택", {
    x: 0.5, y: 3.45, w: 9.0, h: 1.0,
    fontFace: FONT_HEADER, fontSize: 18, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });

  addTechNote(slide, "현재 어느 vendor도 model lineage / safety contract / cost matrix를 표준 포맷으로 공개 안 함");
}

// ---- Slide 3: Regulatory inevitability ----------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "2027년부터 OEM은 AI audit 의무화. 그 audit 표준이 아직 없습니다.", 3);

  // Timeline
  const events = [
    { y: "2024", t: "EU AI Act 발효" },
    { y: "2025", t: "한국 AI 기본법 통과" },
    { y: "2026", t: "ISO/IEC 42001 가속" },
    { y: "2027", t: "EU high-risk AI 시행" },
    { y: "2028", t: "NIST AI RMF 정부조달" },
    { y: "2029", t: "한국 시행규칙 분야별 인증" },
    { y: "2030", t: "글로벌 의무 audit" },
  ];

  events.forEach((e, i) => {
    const y = 1.7 + i * 0.45;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 1.2, h: 0.35,
      fill: { color: i >= 4 ? ACCENT : NAVY }, line: { color: i >= 4 ? ACCENT : NAVY },
    });
    slide.addText(e.y, {
      x: 0.5, y, w: 1.2, h: 0.35,
      fontFace: FONT_HEADER, fontSize: 12, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText(e.t, {
      x: 1.85, y, w: 7.5, h: 0.35,
      fontFace: FONT_BODY, fontSize: 13, color: INK,
      align: "left", valign: "middle", margin: 0,
    });
  });

  slide.addText("→ \"누가 audit 표준을 만드는가\"가 2026년에 정해집니다.", {
    x: 0.5, y: 4.95, w: 9.0, h: 0.35,
    fontFace: FONT_HEADER, fontSize: 13, bold: true, italic: true, color: ACCENT,
    align: "center", margin: 0,
  });

  addTechNote(slide, "EU AI Act Annex III \"high-risk AI\" 분류에 manufacturing inspection 포함");
}

// ---- Slide 4: Two-layer game --------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "데이터 표준은 substrate. 진짜 moat는 그 위 메타 표준입니다.", 4);

  // Layer 0 (top, accent)
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.7, w: 9.0, h: 1.4, fill: { color: ACCENT }, line: { color: ACCENT },
  });
  slide.addText("Layer 0 — Manufacturing standards FOR AI", {
    x: 0.7, y: 1.8, w: 8.6, h: 0.4,
    fontFace: FONT_HEADER, fontSize: 16, bold: true, color: "FFFFFF",
    align: "left", margin: 0,
  });
  slide.addText("AIAS · AISC · MMC — \"AI 자체를 audit · 비교 · 인증하는 표준\"", {
    x: 0.7, y: 2.2, w: 8.6, h: 0.4,
    fontFace: FONT_BODY, fontSize: 14, color: "FFFFFF",
    align: "left", margin: 0,
  });
  slide.addText("적용 범위: 모든 제조 AI (검사·가공·로봇·자율주행)", {
    x: 0.7, y: 2.6, w: 8.6, h: 0.4,
    fontFace: FONT_BODY, fontSize: 12, italic: true, color: GOLD,
    align: "left", margin: 0,
  });

  // Arrow
  slide.addText("↑↑↑  그 위에 얹힘  ↑↑↑", {
    x: 0.5, y: 3.2, w: 9.0, h: 0.3,
    fontFace: FONT_HEADER, fontSize: 12, color: MUTED, italic: true,
    align: "center", margin: 0,
  });

  // Layer 1
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 3.55, w: 9.0, h: 1.3, fill: { color: NAVY }, line: { color: NAVY },
  });
  slide.addText("Layer 1 — AI for manufacturing standards", {
    x: 0.7, y: 3.65, w: 8.6, h: 0.4,
    fontFace: FONT_HEADER, fontSize: 16, bold: true, color: "FFFFFF",
    align: "left", margin: 0,
  });
  slide.addText("AICS (검사) · CCS (가공) — \"검사 결과를 vendor 무관하게 표현\"", {
    x: 0.7, y: 4.05, w: 8.6, h: 0.4,
    fontFace: FONT_BODY, fontSize: 14, color: ICE,
    align: "left", margin: 0,
  });
  slide.addText("우리 현재 위치  ✓", {
    x: 0.7, y: 4.45, w: 8.6, h: 0.3,
    fontFace: FONT_BODY, fontSize: 12, italic: true, color: GOLD,
    align: "left", margin: 0,
  });

  addTechNote(slide, "두 layer 모두 우리가 정의 중 — 다른 vendor는 layer 1조차 없음");
}

// ---- Slide 5: Buffett moat --------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "AI 모델은 commodity. AI audit 표준은 권력.", 5);

  slide.addTable([
    [
      { text: "자산", options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
      { text: "카피 시간", options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
    ],
    ["AI 모델 가중치",                     { text: "18 개월", options: { color: NO, bold: true } }],
    ["어댑터 코드 (BSD-3)",                 { text: "18 개월", options: { color: NO, bold: true } }],
    ["학술 논문",                          { text: "24 개월", options: { color: NO, bold: true } }],
    [
      { text: "공개 표준 + 거버넌스", options: { color: INK, bold: true, fill: { color: ICE } } },
      { text: "18 년 (Bluetooth)", options: { color: OK, bold: true, fill: { color: ICE } } },
    ],
    [
      { text: "다중 vendor 누적 라벨", options: { color: INK, bold: true, fill: { color: ICE } } },
      { text: "시간만이 만듦", options: { color: OK, bold: true, fill: { color: ICE } } },
    ],
  ], {
    x: 0.5, y: 1.7, w: 9.0, colW: [5.5, 3.5],
    fontFace: FONT_BODY, fontSize: 14,
    border: { pt: 1, color: "E5E7EB" }, rowH: 0.45,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.6, w: 9.0, h: 0.5, fill: { color: NAVY }, line: { color: NAVY },
  });
  slide.addText("대형 vendor는 우리 모델 18개월 안에 따라잡습니다. 표준은 못 따라잡습니다.", {
    x: 0.6, y: 4.6, w: 8.8, h: 0.5,
    fontFace: FONT_BODY, fontSize: 13, italic: true, color: GOLD, bold: true,
    align: "center", valign: "middle", margin: 0,
  });

  addTechNote(slide, "역사적 비교: Bluetooth 12년, HTTPS 23년, HACCP 30년 — 모두 표준 owner가 우위 유지");
}

// ---- Slide 6: 8 standards overview --------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "AI 결정의 audit · 운영 · lifecycle 전체 layer를 표준화합니다", 6);

  const groups = [
    { title: "Audit (Layer 0.1)",      stds: ["AIAS — 결정 lineage", "AISC — 안전 contract", "MMC — Model Card"] },
    { title: "운영 (Layer 0.2)",       stds: ["Drift Detection", "Cross-vendor Benchmark", "Federated Learning"] },
    { title: "Lifecycle (Layer 0.3)",  stds: ["Lineage Tracking", "Decommissioning", ""] },
  ];

  groups.forEach((g, i) => {
    const x = 0.5 + i * 3.05;
    slide.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.7, w: 2.85, h: 2.7,
      fill: { color: "F8FAFC" }, line: { color: NAVY, width: 1 },
    });
    slide.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.7, w: 2.85, h: 0.45, fill: { color: NAVY }, line: { color: NAVY },
    });
    slide.addText(g.title, {
      x, y: 1.7, w: 2.85, h: 0.45,
      fontFace: FONT_HEADER, fontSize: 13, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    g.stds.filter(Boolean).forEach((s, j) => {
      slide.addText("•  " + s, {
        x: x + 0.15, y: 2.3 + j * 0.5, w: 2.55, h: 0.45,
        fontFace: FONT_BODY, fontSize: 12, color: INK,
        align: "left", valign: "middle", margin: 0,
      });
    });
  });

  slide.addText("기존 ISO/IEC 42001 + NIST AI RMF + IEC 61508 위에 manufacturing extension", {
    x: 0.5, y: 4.6, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 12, italic: true, color: ACCENT, bold: true,
    align: "center", margin: 0,
  });

  addTechNote(slide, "8개 표준 후보 상세: docs/ai_manufacturing_standards.md");
}

// ---- Slide 7: AIAS detail ----------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "AIAS — 모든 AI 결정에 6개 메타데이터 첨부 (5년 후 OEM 의무)", 7);

  // YAML example (left)
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.7, w: 5.5, h: 3.3,
    fill: { color: "0F172A" }, line: { color: "0F172A" },
  });
  slide.addText([
    { text: "aias_version: \"0.1\"\n", options: { color: GOLD, breakLine: true } },
    { text: "decision_id: \"uuid-...\"\n", options: { color: ICE, breakLine: true } },
    { text: "model:\n", options: { color: GOLD, breakLine: true } },
    { text: "  weights_hash: sha256:...\n", options: { color: ICE, breakLine: true } },
    { text: "  training_data_hash: sha256:...\n", options: { color: ICE, breakLine: true } },
    { text: "context:\n", options: { color: GOLD, breakLine: true } },
    { text: "  cost_matrix: { c_escape: 50 }\n", options: { color: ICE, breakLine: true } },
    { text: "  safety: { metric: escape_rate,\n", options: { color: ICE, breakLine: true } },
    { text: "            limit: 0.001 }\n", options: { color: ICE, breakLine: true } },
    { text: "decision:\n", options: { color: GOLD, breakLine: true } },
    { text: "  action: PASS\n", options: { color: ICE, breakLine: true } },
    { text: "  confidence: 0.94\n", options: { color: ICE, breakLine: true } },
    { text: "human_override: false\n", options: { color: ICE } },
  ], {
    x: 0.7, y: 1.85, w: 5.1, h: 3.0,
    fontFace: "Consolas", fontSize: 11, valign: "top", margin: 0,
  });

  // Right: meaning
  slide.addText("의미", {
    x: 6.2, y: 1.7, w: 3.3, h: 0.35,
    fontFace: FONT_HEADER, fontSize: 14, bold: true, color: NAVY,
    align: "left", margin: 0,
  });
  slide.addText([
    { text: "AI 사고 발생 시 어느 학습 batch까지 책임 추적 가능", options: { bullet: true, breakLine: true, color: INK } },
    { text: "보험사·정부 audit 친화적", options: { bullet: true, breakLine: true, color: INK } },
    { text: "vendor 간 객관적 비교 가능", options: { bullet: true, breakLine: true, color: INK } },
    { text: "5년 후 OEM 조달 의무 요건화 예상", options: { bullet: true, color: NO, bold: true } },
  ], {
    x: 6.2, y: 2.1, w: 3.3, h: 2.9,
    fontFace: FONT_BODY, fontSize: 12, paraSpaceAfter: 6, valign: "top",
  });

  addTechNote(slide, "W3C PROV (lineage 표준) + HuggingFace Model Cards를 manufacturing vertical로 합성");
}

// ---- Slide 8: 6-year timeline ------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "표준 정의 1년. 의무화까지 7년. 지금 시작 안 하면 늦습니다.", 8);

  const milestones = [
    { y: "Year 1", e: "AIAS RFC v0.1 published", n: "← 우리 (현재)", color: ACCENT },
    { y: "Year 2", e: "첫 외부 vendor 채택", n: "", color: NAVY },
    { y: "Year 3", e: "ISO/IEC 42001 working group 합류", n: "", color: NAVY },
    { y: "Year 4", e: "한국 AI 기본법 시행규칙 인용", n: "", color: NAVY },
    { y: "Year 5", e: "ISO/IEC TC SC 42 정식 RFC", n: "", color: NAVY },
    { y: "Year 6", e: "EU AI Act audit framework 채택", n: "", color: NAVY },
    { y: "Year 7+", e: "의무화 → 우리 표준 통과 못 하면 OEM 납품 X", n: "₩100B 회사 임계점", color: OK },
  ];

  milestones.forEach((m, i) => {
    const y = 1.65 + i * 0.45;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 1.2, h: 0.35, fill: { color: m.color }, line: { color: m.color },
    });
    slide.addText(m.y, {
      x: 0.5, y, w: 1.2, h: 0.35,
      fontFace: FONT_HEADER, fontSize: 12, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText(m.e, {
      x: 1.85, y, w: 5.0, h: 0.35,
      fontFace: FONT_BODY, fontSize: 13, color: INK,
      align: "left", valign: "middle", margin: 0,
    });
    if (m.n) {
      slide.addText(m.n, {
        x: 6.9, y, w: 2.6, h: 0.35,
        fontFace: FONT_BODY, fontSize: 11, italic: true, bold: true, color: ACCENT,
        align: "right", valign: "middle", margin: 0,
      });
    }
  });

  addTechNote(slide, "비슷한 메타 표준 평균 7-10년: HACCP 30년, OWASP 10년, HuggingFace Model Cards 5년");
}

// ---- Slide 9: Historical pattern ---------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "작은 조직이 정의한 표준이 글로벌 의무가 되는 5단계 패턴", 9);

  const steps = [
    "작은 조직이 표준 정의                ← 우리 (Year 1)",
    "무료 reference 구현                  ← aoi-sentinel",
    "큰 조직이 채택 시작                   ← Year 2",
    "정부·규제가 인용                      ← Year 4-5",
    "의무화                              ← Year 6-7+",
  ];

  steps.forEach((s, i) => {
    const y = 1.7 + i * 0.45;
    slide.addShape(pres.shapes.OVAL, {
      x: 0.5, y: y + 0.04, w: 0.4, h: 0.32,
      fill: { color: NAVY }, line: { color: NAVY },
    });
    slide.addText(String(i + 1), {
      x: 0.5, y: y + 0.04, w: 0.4, h: 0.32,
      fontFace: FONT_HEADER, fontSize: 13, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText(s, {
      x: 1.05, y, w: 8.4, h: 0.4,
      fontFace: FONT_BODY, fontSize: 13, color: INK,
      align: "left", valign: "middle", margin: 0,
    });
  });

  // Verified examples panel
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.05, w: 9.0, h: 0.95,
    fill: { color: ICE }, line: { color: ICE },
  });
  slide.addText([
    { text: "검증된 사례:  ", options: { bold: true, color: NAVY } },
    { text: "HACCP (NASA→글로벌 의무 30년)  ·  ", options: { color: INK } },
    { text: "OWASP Top 10 (비영리→웹 보안 표준)  ·  ", options: { color: INK } },
    { text: "HuggingFace Model Cards (학술→AI 메타데이터 default 5년)", options: { color: INK } },
  ], {
    x: 0.7, y: 4.1, w: 8.6, h: 0.85,
    fontFace: FONT_BODY, fontSize: 12, valign: "middle", margin: 0,
  });

  addTechNote(slide, "공통: 정의자 = reference impl + 거버넌스 owner. 우리도 둘 다 갖춤.");
}

// ---- Slide 10: Existing traction ---------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "추상적 비전이 아닙니다 — 표준 + 코드 + 실험 결과 모두 공개됐습니다", 10);

  slide.addTable([
    [
      { text: "자산",     options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
      { text: "위치",     options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
      { text: "진척",     options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
    ],
    ["AICS RFC v0.1",                "github.com/DrJinHoChoi/aoi-common-spec",  { text: "✅ Published", options: { color: OK, bold: true } }],
    ["aoi-sentinel reference 구현",  "github.com/DrJinHoChoi/aoi-sentinel",     { text: "✅ 50+ tests", options: { color: OK, bold: true } }],
    ["CCS RFC (placeholder)",         "github.com/DrJinHoChoi/cnc-common-spec",  { text: "✅ Reserved",  options: { color: OK, bold: true } }],
    ["Mamba RL 학습 검증",            "VisA 4,416장, A100 80GB",                  { text: "✅ Escape=0",  options: { color: OK, bold: true } }],
    ["3-vendor 어댑터",                "Saki / Koh Young / generic_csv",           { text: "✅ 작동",      options: { color: OK, bold: true } }],
  ], {
    x: 0.5, y: 1.7, w: 9.0, colW: [3.0, 4.5, 1.5],
    fontFace: FONT_BODY, fontSize: 12, color: INK,
    border: { pt: 1, color: "E5E7EB" }, rowH: 0.5,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.6, w: 9.0, h: 0.5, fill: { color: ACCENT }, line: { color: ACCENT },
  });
  slide.addText("현재 부족한 것 = 첫 anchor 고객 1곳", {
    x: 0.6, y: 4.6, w: 8.8, h: 0.5,
    fontFace: FONT_HEADER, fontSize: 14, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });

  addTechNote(slide, "모든 결과 GitHub 검증 가능, BSD-3 라이선스");
}

// ---- Slide 11: VisA proof ----------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "lab에서 60번 학습 — 놓친 결함 0건, 비용 5.6배 ↓", 11);

  const stats = [
    { v: "0",     l: "Escape per iter",     s: "60 iter 동안", c: OK },
    { v: "5.6×",  l: "Cost reduction",       s: "1280 → 230",  c: NAVY },
    { v: "iter 4", l: "정책 자동 전환",       s: "ESCALATE→DEFECT", c: ACCENT },
    { v: "0",     l: "Safety violations",    s: "Hard contract", c: OK },
  ];

  stats.forEach((s, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.5 + col * 4.6;
    const y = 1.7 + row * 1.65;
    slide.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 4.3, h: 1.45,
      fill: { color: "F8FAFC" }, line: { color: "E5E7EB", width: 1 },
    });
    slide.addText(s.v, {
      x, y, w: 4.3, h: 0.85,
      fontFace: FONT_HEADER, fontSize: 40, bold: true, color: s.c,
      align: "center", valign: "bottom", margin: 0,
    });
    slide.addText(s.l, {
      x, y: y + 0.85, w: 4.3, h: 0.3,
      fontFace: FONT_BODY, fontSize: 12, bold: true, color: INK,
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText(s.s, {
      x, y: y + 1.13, w: 4.3, h: 0.3,
      fontFace: FONT_BODY, fontSize: 10, color: MUTED,
      align: "center", valign: "middle", margin: 0,
    });
  });

  slide.addText("→ AISC (AI Safety Contract) 표준의 reference 구현이 동작함을 입증", {
    x: 0.5, y: 5.0, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 12, italic: true, color: ACCENT, bold: true,
    align: "center", margin: 0,
  });

  addTechNote(slide, "NVIDIA A100 80GB, Lagrangian PPO, pure-PyTorch Mamba — pilot_evidence_kr.md");
}

// ---- Slide 12: TAM ------------------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "우리 시장은 manufacturing AI 전체. 한국·아시아 우선 진입.", 12);

  slide.addTable([
    [
      { text: "도메인",       options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
      { text: "2024",          options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
      { text: "2030E",        options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
      { text: "우리 적용",    options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
    ],
    ["자동차 전장 SMT (현재 wedge)", "$20B",  "$40B",   { text: "✅ 즉시", options: { color: OK, bold: true } }],
    ["AI 서버 / HBM 보드",            "$80B",  "$250B",  { text: "⏳ Year 2", options: { color: ACCENT, bold: true } }],
    ["휴머노이드 부품",               "$2B",   "$30B",   { text: "⏳ Year 2", options: { color: ACCENT, bold: true } }],
    ["EV 배터리 BMS",                  "$30B",  "$80B",   { text: "⏳ Year 3", options: { color: ACCENT, bold: true } }],
    ["CNC 가공",                       "$20B",  "$40B",   { text: "⏳ Year 2", options: { color: ACCENT, bold: true } }],
    [
      { text: "합계 직접 가능 시장",       options: { bold: true, color: INK, fill: { color: ICE } } },
      { text: "~$150B",                    options: { bold: true, color: INK, fill: { color: ICE } } },
      { text: "~$440B",                    options: { bold: true, color: NO, fill: { color: ICE } } },
      { text: "—",                          options: { color: INK, fill: { color: ICE } } },
    ],
  ], {
    x: 0.5, y: 1.7, w: 9.0, colW: [3.5, 1.4, 1.6, 2.5],
    fontFace: FONT_BODY, fontSize: 12, color: INK,
    border: { pt: 1, color: "E5E7EB" }, rowH: 0.42,
  });

  slide.addText("같은 박스, 같은 표준, 다른 도메인 — 하나의 표준이 모든 manufacturing AI를 통일", {
    x: 0.5, y: 4.85, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 12, italic: true, color: ACCENT, bold: true,
    align: "center", margin: 0,
  });

  addTechNote(slide, "출처: Mordor Intelligence, IDC, Goldman Sachs humanoid 보고서 (2024-2025)");
}

// ---- Slide 13: Stakeholder ---------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "표준은 혼자 못 만듭니다 — 5종류 이해관계자 동시 engage", 13);

  slide.addTable([
    [
      { text: "Stakeholder",    options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
      { text: "한국",            options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
      { text: "글로벌",          options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
      { text: "우리가 줄 가치",  options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
    ],
    [{ text: "고객 (OEM/Tier-1)", options: { bold: true, color: INK } },  "현대모비스, LG이노텍",   "Continental, Bosch",     "ROI 데이터"],
    [{ text: "벤더 (장비)",        options: { bold: true, color: INK } },  "(한국 적음)",            "Saki, Koh Young, Mycronic", "플랫폼 통합"],
    [{ text: "표준 body",          options: { bold: true, color: INK } },  "KSA, KETI, TTA",        "IPC, SEMI, IEC, ISO",    "잘 만든 RFC"],
    [{ text: "연구계",             options: { bold: true, color: INK } },  "서울대, KAIST",          "MIT, ETH",                "데이터셋"],
    [{ text: "정부",               options: { bold: true, color: INK } },  "산업부, 중기부",         "EU IndustryX, NIST",      "정책 부합 솔루션"],
  ], {
    x: 0.5, y: 1.7, w: 9.0, colW: [2.2, 2.0, 2.5, 2.3],
    fontFace: FONT_BODY, fontSize: 11, color: INK,
    border: { pt: 1, color: "E5E7EB" }, rowH: 0.42,
  });

  slide.addText("Phase A 고객·벤더 집중  →  Phase B 연구계·표준 body  →  Phase C 정부", {
    x: 0.5, y: 4.6, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 12, italic: true, color: ACCENT, bold: true,
    align: "center", margin: 0,
  });

  addTechNote(slide, "상세 24개월 분기별 액션: docs/standardization_methodology.md");
}

// ---- Slide 14: Governance ----------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "표준이 우리만의 것이 되면 표준이 아닙니다 — 의도적 자기 견제", 14);

  const stages = [
    { ver: "v0.x",   desc: "단독 editor (DrJinHoChoi)",                          when: "현재",   c: NAVY },
    { ver: "v1.0+",  desc: "Multi-editor 위원회 (5-7명, simple majority)",       when: "Year 2", c: ACCENT },
    { ver: "v2.0+",  desc: "비영리 재단/컨소시엄 (Bluetooth SIG 모델)",          when: "Year 3+", c: OK },
  ];

  stages.forEach((s, i) => {
    const y = 1.8 + i * 0.85;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 1.4, h: 0.65, fill: { color: s.c }, line: { color: s.c },
    });
    slide.addText(s.ver, {
      x: 0.5, y, w: 1.4, h: 0.65,
      fontFace: FONT_HEADER, fontSize: 16, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText(s.desc, {
      x: 2.0, y, w: 5.5, h: 0.65,
      fontFace: FONT_BODY, fontSize: 13, color: INK,
      align: "left", valign: "middle", margin: 0,
    });
    slide.addText(s.when, {
      x: 7.6, y, w: 1.9, h: 0.65,
      fontFace: FONT_BODY, fontSize: 11, italic: true, color: MUTED,
      align: "right", valign: "middle", margin: 0,
    });
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.55, w: 9.0, h: 0.55, fill: { color: NAVY }, line: { color: NAVY },
  });
  slide.addText("Bluetooth가 노키아 것이 아니듯 — 표준은 우리만의 것이 아님. 그러나 reference impl + audit 서비스 매출은 우리.", {
    x: 0.6, y: 4.55, w: 8.8, h: 0.55,
    fontFace: FONT_BODY, fontSize: 11, italic: true, color: GOLD, bold: true,
    align: "left", valign: "middle", margin: 0,
  });

  addTechNote(slide, "AICS RFC-001 §11에 multi-editor 거버넌스 약속 명시 + BSD-3 라이선스");
}

// ---- Slide 15: Closing manifesto ---------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: NAVY };

  slide.addText("AI 모델은 commodity.", {
    x: 0.5, y: 1.0, w: 9.0, h: 0.7,
    fontFace: FONT_HEADER, fontSize: 36, bold: true, color: "FFFFFF",
    align: "center", margin: 0,
  });
  slide.addText("AI 표준은 권력.", {
    x: 0.5, y: 1.7, w: 9.0, h: 0.7,
    fontFace: FONT_HEADER, fontSize: 36, bold: true, color: ACCENT,
    align: "center", margin: 0,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 3.0, y: 2.7, w: 4.0, h: 0.05, fill: { color: GOLD }, line: { color: GOLD },
  });

  slide.addText("우리는 AI를 파는 게 아니라,\nAI가 따라야 할 표준을 정의합니다.", {
    x: 0.5, y: 3.0, w: 9.0, h: 1.0,
    fontFace: FONT_BODY, fontSize: 18, italic: true, color: ICE,
    align: "center", margin: 0,
  });

  slide.addText("지금 시작하는 사람이 5년 후 표준 owner.", {
    x: 0.5, y: 4.2, w: 9.0, h: 0.5,
    fontFace: FONT_HEADER, fontSize: 16, bold: true, color: GOLD,
    align: "center", margin: 0,
  });

  slide.addText("DrJinHoChoi  ·  github.com/DrJinHoChoi  ·  필요하시면 같이 만들 수 있습니다.", {
    x: 0.5, y: 4.95, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 12, color: ICE,
    align: "center", margin: 0,
  });
}

// ---- Save ----------------------------------------------------------------
const outPath = "C:/Users/jinho/source/repos/DrJinHoChoi/aoi-sentinel/docs/sales/pitch_deck_strategic_kr.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log(`✓ wrote ${outPath}`);
});
