// Build pitch_deck_kr.pptx from the markdown source.
// Hybrid layout: each slide has a big executive headline, 3 plain-Korean bullets,
// and a small technical note for engineers.

const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";  // 10" × 5.625"
pres.author = "DrJinHoChoi";
pres.title = "aoi-sentinel — Pitch Deck (KR)";

// ---- Color palette: Midnight Executive (industrial / serious / Korean B2B)
const NAVY    = "1E2761";  // headers, title slides
const ICE     = "CADCFC";  // subtle backgrounds
const ACCENT  = "F96167";  // coral red — only for highlights
const INK     = "1A1A2E";  // body text on light backgrounds
const MUTED   = "6B7280";  // tech notes
const PAGE_BG = "FFFFFF";  // body slide backgrounds (white)
const OK      = "2DA44E";  // green checkmarks
const NO      = "CF222E";  // red X

const FONT_HEADER = "Malgun Gothic";
const FONT_BODY   = "Malgun Gothic";

// ---- Slide-helper functions ----------------------------------------------

function addSlideHeader(slide, headline, slideNum) {
  // Slide number in upper-right
  slide.addText(`${slideNum} / 15`, {
    x: 9.0, y: 0.18, w: 0.85, h: 0.3,
    fontFace: FONT_BODY, fontSize: 10, color: MUTED, align: "right", margin: 0,
  });

  // Headline
  slide.addText("🎯 " + headline, {
    x: 0.5, y: 0.4, w: 9.0, h: 1.0,
    fontFace: FONT_HEADER, fontSize: 24, bold: true, color: NAVY,
    align: "left", valign: "top", margin: 0,
  });
}

function addTechNote(slide, note) {
  // Bottom-of-slide technical credibility line
  slide.addText("🔧  " + note, {
    x: 0.5, y: 5.10, w: 9.0, h: 0.35,
    fontFace: FONT_BODY, fontSize: 10, italic: true, color: MUTED,
    align: "left", margin: 0,
  });
}

// ---- Slide 1: Title ------------------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: NAVY };

  slide.addText("aoi-sentinel", {
    x: 0.5, y: 1.5, w: 9.0, h: 1.2,
    fontFace: FONT_HEADER, fontSize: 60, bold: true, color: "FFFFFF",
    align: "center", margin: 0,
  });

  slide.addText("AI가 SMT 검사 후 70% '가짜 불량'을 자동으로 골라드립니다", {
    x: 0.5, y: 2.8, w: 9.0, h: 0.6,
    fontFace: FONT_BODY, fontSize: 20, color: ICE,
    align: "center", margin: 0,
  });

  slide.addText("— 6개월 무료 파일럿 —", {
    x: 0.5, y: 3.5, w: 9.0, h: 0.5,
    fontFace: FONT_BODY, fontSize: 18, color: ACCENT, bold: true,
    align: "center", margin: 0,
  });

  slide.addText("DrJinHoChoi · 2026", {
    x: 0.5, y: 5.1, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 12, color: ICE,
    align: "center", margin: 0,
  });
}

// ---- Slide 2: Problem ----------------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "작업자분이 매일 보시는 200건 중 140건은 헛짓입니다", 2);

  // Big stat: 70% callout
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.7, w: 3.5, h: 2.8,
    fill: { color: ACCENT }, line: { color: ACCENT },
  });
  slide.addText("70%", {
    x: 0.5, y: 1.8, w: 3.5, h: 1.5,
    fontFace: FONT_HEADER, fontSize: 90, bold: true, color: "FFFFFF",
    align: "center", margin: 0,
  });
  slide.addText("AOI NG 중 가성불량", {
    x: 0.5, y: 3.4, w: 3.5, h: 0.5,
    fontFace: FONT_BODY, fontSize: 16, color: "FFFFFF",
    align: "center", margin: 0,
  });
  slide.addText("매일 시간 낭비", {
    x: 0.5, y: 3.95, w: 3.5, h: 0.4,
    fontFace: FONT_BODY, fontSize: 14, color: "FFFFFF",
    align: "center", margin: 0,
  });

  // Bullets
  slide.addText([
    { text: "100건 중 30건만 진짜 결함, 70건은 가성불량", options: { bullet: true, breakLine: true, color: INK } },
    { text: "운영자 1-2명이 매일 전담 (라인 처리량 ↓, 야근 발생)", options: { bullet: true, breakLine: true, color: INK } },
    { text: "데이터 = 운영자 머릿속 (휘발 = 자산화 안 됨)", options: { bullet: true, color: INK } },
  ], {
    x: 4.4, y: 1.8, w: 5.2, h: 2.8,
    fontFace: FONT_BODY, fontSize: 16, paraSpaceAfter: 14, valign: "top",
  });

  addTechNote(slide, "자동차 전장 SMT 라인 평균 false-call rate 30% (산업 측정값)");
}

// ---- Slide 3: Limits of existing solutions -------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "다른 회사 솔루션은 '자기네 검사기에서만' 동작합니다", 3);

  slide.addTable([
    [
      { text: "솔루션", options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "left" } },
      { text: "한계", options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "left" } },
    ],
    ["Mycronic DeepReview",          "Mycronic 검사기에서만 작동"],
    ["Koh Young KSMART AI",          "Koh Young 검사기에서만 + 클라우드 의존"],
    ["Siemens Opcenter AOI-FCR",     "룰 수동 튜닝, 자동학습 없음"],
  ], {
    x: 0.5, y: 1.7, w: 9.0, h: 2.0, colW: [3.5, 5.5],
    fontFace: FONT_BODY, fontSize: 14, color: INK,
    border: { pt: 1, color: "E5E7EB" }, rowH: 0.5,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.0, w: 9.0, h: 0.9,
    fill: { color: ICE }, line: { color: ICE },
  });
  slide.addText("→ 라인에 검사기 두 종류 섞여있으면 두 개 따로 도입.  데이터는 vendor에 갇힘.", {
    x: 0.6, y: 4.05, w: 8.8, h: 0.8,
    fontFace: FONT_BODY, fontSize: 14, color: NAVY, bold: true,
    align: "left", valign: "middle", margin: 0,
  });

  addTechNote(slide, "위 솔루션 모두 closed-source schema, 데이터 휴대 불가, escape rate 보장 없음");
}

// ---- Slide 4: Our answer -------------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "어떤 검사기든 OK. 데이터는 공장 밖에 절대 안 나갑니다.", 4);

  const checks = [
    "벤더 무관 — Saki / Koh Young / Mycronic / TRI 다 지원",
    "온프레 only — 데이터 외부 export 0",
    "자가학습 — 운영자 클릭 = 학습 라벨",
    "Escape 0건 수학적 보장 — 결함 누출 사고 안 남",
    "공개 산업 표준 — 5년 후에도 데이터 휴대 가능",
  ];
  checks.forEach((t, i) => {
    const y = 1.7 + i * 0.6;
    slide.addShape(pres.shapes.OVAL, {
      x: 0.55, y: y + 0.05, w: 0.4, h: 0.4,
      fill: { color: OK }, line: { color: OK },
    });
    slide.addText("✓", {
      x: 0.55, y: y + 0.05, w: 0.4, h: 0.4,
      fontFace: FONT_HEADER, fontSize: 18, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText(t, {
      x: 1.1, y: y, w: 8.4, h: 0.5,
      fontFace: FONT_BODY, fontSize: 16, color: INK,
      align: "left", valign: "middle", margin: 0,
    });
  });

  addTechNote(slide, "AICS (AOI Common Inspection Schema) RFC v0.1 — github.com/DrJinHoChoi/aoi-common-spec, BSD-3");
}

// ---- Slide 5: How it works ----------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "검사기 옆에 USB로 박스 하나 꽂으면 끝", 5);

  // Flow row 1: AOI → Box
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.7, w: 2.0, h: 0.9, fill: { color: ICE }, line: { color: NAVY, width: 1 },
  });
  slide.addText("AOI 검사기", {
    x: 0.5, y: 1.7, w: 2.0, h: 0.9,
    fontFace: FONT_HEADER, fontSize: 16, bold: true, color: NAVY,
    align: "center", valign: "middle", margin: 0,
  });

  slide.addText("→", {
    x: 2.55, y: 1.75, w: 0.5, h: 0.8,
    fontFace: FONT_HEADER, fontSize: 28, color: NAVY, bold: true,
    align: "center", valign: "middle", margin: 0,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 3.2, y: 1.5, w: 2.4, h: 1.3, fill: { color: NAVY }, line: { color: NAVY },
  });
  slide.addText("aoi-sentinel\n박스", {
    x: 3.2, y: 1.5, w: 2.4, h: 1.3,
    fontFace: FONT_HEADER, fontSize: 16, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });

  slide.addText("→", {
    x: 5.7, y: 1.75, w: 0.5, h: 0.8,
    fontFace: FONT_HEADER, fontSize: 28, color: NAVY, bold: true,
    align: "center", valign: "middle", margin: 0,
  });

  // Three decisions
  const decisions = [
    { label: "DEFECT (자동 rework)", color: NO },
    { label: "PASS (자동 통과)", color: OK },
    { label: "ESCALATE (운영자)", color: "F59E0B" },
  ];
  decisions.forEach((d, i) => {
    const y = 1.5 + i * 0.45;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 6.4, y, w: 3.1, h: 0.4, fill: { color: d.color }, line: { color: d.color },
    });
    slide.addText(d.label, {
      x: 6.4, y, w: 3.1, h: 0.4,
      fontFace: FONT_BODY, fontSize: 12, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
  });

  // Below: feedback loop
  slide.addText("운영자가 ESCALATE 결정 → 학습 라벨 → 모델 자동 개선", {
    x: 0.5, y: 3.7, w: 9.0, h: 0.4,
    fontFace: FONT_HEADER, fontSize: 16, bold: true, color: ACCENT,
    align: "center", valign: "middle", margin: 0,
  });

  slide.addText([
    { text: "라인 운영자가 평소대로 일하면 박스가 알아서 학습", options: { bullet: true, breakLine: true, color: INK } },
    { text: "라인 운영 변경 / SCADA 통합 / MES 통합 모두 불필요", options: { bullet: true, breakLine: true, color: INK } },
    { text: "셋업 = SMB 폴더 마운트 + 박스 부팅 + 운영자 UI 5분 교육", options: { bullet: true, color: INK } },
  ], {
    x: 0.5, y: 4.2, w: 9.0, h: 0.9,
    fontFace: FONT_BODY, fontSize: 13, paraSpaceAfter: 4, valign: "top",
  });

  addTechNote(slide, "ConvNeXt 이미지 + Mamba 시퀀스 인코더 + Lagrangian PPO 정책, 3-action {DEFECT, PASS, ESCALATE}");
}

// ---- Slide 6: 3-mode rollout --------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "처음 한 달은 박스가 일 안 합니다. 그 다음에 천천히 자동.", 6);

  slide.addTable([
    [
      { text: "단계",     options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
      { text: "기간",     options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
      { text: "운영자 부담", options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
      { text: "박스 역할",   options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
    ],
    ["SHADOW",     "Day 1-30",     "100% (평소대로)",       "자기 결정 표시만"],
    ["ASSIST",     "Day 31-180",   "30% → 5% 점진 감소",    "자신 있는 케이스 자동"],
    ["AUTONOMOUS", "Day 181+ (옵션)", "KPI 모니터링만",      "MES 직접 push"],
  ], {
    x: 0.5, y: 1.7, w: 9.0, h: 2.4, colW: [1.5, 1.7, 2.5, 3.3],
    fontFace: FONT_BODY, fontSize: 13, color: INK,
    border: { pt: 1, color: "E5E7EB" }, rowH: 0.5,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.3, w: 9.0, h: 0.6,
    fill: { color: ACCENT }, line: { color: ACCENT },
  });
  slide.addText("Escape 한 건이라도 발생 시 자동으로 보수적 모드로 강등 — 사고 안 남.", {
    x: 0.6, y: 4.3, w: 8.8, h: 0.6,
    fontFace: FONT_BODY, fontSize: 14, bold: true, color: "FFFFFF",
    align: "left", valign: "middle", margin: 0,
  });

  addTechNote(slide, "모드 전환은 hold-out replay safety gate 통과 시 자동 격상 (runtime/safety_gate.py)");
}

// ---- Slide 7: VisA experiment proof -------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "4,400장 PCB로 60번 학습 — 놓친 결함 0건, 비용 5.6배 ↓", 7);

  // 4 stat cards
  const stats = [
    { label: "Escape per iter",  value: "0",       sub: "60 iter 동안",  color: OK },
    { label: "Cost reduction",   value: "5.6×",    sub: "1280 → 230",   color: NAVY },
    { label: "Auto policy switch", value: "iter 4", sub: "ESCALATE→DEFECT", color: ACCENT },
    { label: "Safety violations", value: "0",      sub: "Hard contract", color: OK },
  ];
  stats.forEach((s, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.5 + col * 4.6;
    const y = 1.7 + row * 1.65;
    slide.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 4.3, h: 1.45,
      fill: { color: "F8FAFC" }, line: { color: "E5E7EB", width: 1 },
    });
    slide.addText(s.value, {
      x, y, w: 4.3, h: 0.85,
      fontFace: FONT_HEADER, fontSize: 44, bold: true, color: s.color,
      align: "center", valign: "bottom", margin: 0,
    });
    slide.addText(s.label, {
      x, y: y + 0.85, w: 4.3, h: 0.3,
      fontFace: FONT_BODY, fontSize: 12, bold: true, color: INK,
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText(s.sub, {
      x, y: y + 1.13, w: 4.3, h: 0.3,
      fontFace: FONT_BODY, fontSize: 10, color: MUTED,
      align: "center", valign: "middle", margin: 0,
    });
  });

  slide.addText("이건 lab 결과 — 실제 라인에선 같은 일이 day 단위로 일어납니다.", {
    x: 0.5, y: 5.0, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 12, italic: true, color: MUTED,
    align: "center", margin: 0,
  });

  addTechNote(slide, "NVIDIA A100 80GB, VisA 4416 images (defect 0.091), pure-PyTorch Mamba — pilot_evidence_kr.md");
}

// ---- Slide 8: Moat -------------------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "AI 모델은 카피됩니다. 표준은 카피 못 합니다.", 8);

  slide.addTable([
    [
      { text: "자산",                       options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
      { text: "18개월 후 카피 가능?",         options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
    ],
    [
      { text: "AI 모델 가중치",                options: { color: INK } },
      { text: "✅ 가능",                       options: { color: NO, bold: true } },
    ],
    [
      { text: "어댑터 코드 (BSD-3 공개)",       options: { color: INK } },
      { text: "✅ 가능",                       options: { color: NO, bold: true } },
    ],
    [
      { text: "공개 표준 (AICS) — 우리가 정의", options: { color: INK, bold: true, fill: { color: ICE } } },
      { text: "❌ 불가능",                      options: { color: OK, bold: true, fill: { color: ICE } } },
    ],
    [
      { text: "다중 vendor 누적 라벨 데이터",   options: { color: INK, bold: true, fill: { color: ICE } } },
      { text: "❌ 불가능 (시간만이 만듦)",        options: { color: OK, bold: true, fill: { color: ICE } } },
    ],
  ], {
    x: 0.5, y: 1.7, w: 9.0, colW: [5.5, 3.5],
    fontFace: FONT_BODY, fontSize: 14,
    border: { pt: 1, color: "E5E7EB" }, rowH: 0.5,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.5, w: 9.0, h: 0.6, fill: { color: NAVY }, line: { color: NAVY },
  });
  slide.addText("Bluetooth가 노키아 것이 아니듯 — 표준은 우리만의 것이 아닙니다. 하지만 우리가 정의했고 운영합니다.", {
    x: 0.6, y: 4.5, w: 8.8, h: 0.6,
    fontFace: FONT_BODY, fontSize: 12, italic: true, color: ICE, bold: true,
    align: "left", valign: "middle", margin: 0,
  });

  addTechNote(slide, "Buffett 경제적 해자 — 표준 거버넌스 + 데이터 플라이휠 + 다중 vendor 어댑터 ecosystem");
}

// ---- Slide 9: Competition table -----------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "6가지 항목 다 ✅인 솔루션은 우리뿐", 9);

  const yes = (txt) => ({ text: txt, options: { color: OK, bold: true, align: "center" } });
  const no  = (txt) => ({ text: txt, options: { color: NO, bold: true, align: "center" } });
  const mid = (txt) => ({ text: txt, options: { color: "F59E0B", bold: true, align: "center" } });
  const head = (txt) => ({ text: txt, options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } });

  slide.addTable([
    [head(""), head("Mycronic"), head("Koh Young"), head("Siemens"),
      { text: "aoi-sentinel", options: { bold: true, color: "FFFFFF", fill: { color: ACCENT }, align: "center" } }],
    [{ text: "벤더 무관", options: { color: INK, bold: true } }, no("❌"), no("❌"), mid("△"), yes("✅")],
    [{ text: "온프레 only", options: { color: INK, bold: true } }, mid("△"), no("❌"), mid("△"), yes("✅")],
    [{ text: "자가학습", options: { color: INK, bold: true } }, mid("△"), yes("✅"), no("❌"), yes("✅")],
    [{ text: "Escape 보장", options: { color: INK, bold: true } }, no("❌"), no("❌"), no("❌"), yes("✅")],
    [{ text: "공개 표준", options: { color: INK, bold: true } }, no("❌"), no("❌"), no("❌"), yes("✅")],
    [{ text: "무료 파일럿", options: { color: INK, bold: true } }, no("❌"), no("❌"), no("❌"), yes("✅")],
  ], {
    x: 0.5, y: 1.7, w: 9.0, colW: [2.4, 1.5, 1.5, 1.5, 2.1],
    fontFace: FONT_BODY, fontSize: 13,
    border: { pt: 1, color: "E5E7EB" }, rowH: 0.42,
  });

  addTechNote(slide, "평가 근거: vendor 공식 자료 + Mycronic DeepReview / Koh Young KSMART 보도자료 (2024-2026)");
}

// ---- Slide 10: Pilot terms ----------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "비용 ₩0. 효과 미달 시 비용 0. 잃을 게 없습니다.", 10);

  // Big "₩0"
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.6, w: 3.5, h: 2.0,
    fill: { color: NAVY }, line: { color: NAVY },
  });
  slide.addText("₩0", {
    x: 0.5, y: 1.6, w: 3.5, h: 1.4,
    fontFace: FONT_HEADER, fontSize: 80, bold: true, color: "FFFFFF",
    align: "center", valign: "bottom", margin: 0,
  });
  slide.addText("HW + SW + 셋업 + 6개월 지원", {
    x: 0.5, y: 3.0, w: 3.5, h: 0.5,
    fontFace: FONT_BODY, fontSize: 12, color: ICE,
    align: "center", margin: 0,
  });

  // Right: 2 column - 우리 / 귀사
  slide.addText("저희 제공", {
    x: 4.4, y: 1.6, w: 5.2, h: 0.4,
    fontFace: FONT_HEADER, fontSize: 14, bold: true, color: NAVY,
    align: "left", margin: 0,
  });
  slide.addText([
    { text: "Edge box (Jetson Orin Nano)", options: { bullet: true, breakLine: true, color: INK } },
    { text: "운영자 UI (모바일 친화)",         options: { bullet: true, breakLine: true, color: INK } },
    { text: "셋업·교육·24/7 원격 지원",       options: { bullet: true, color: INK } },
  ], {
    x: 4.4, y: 2.0, w: 5.2, h: 1.2,
    fontFace: FONT_BODY, fontSize: 13, paraSpaceAfter: 3, valign: "top",
  });

  slide.addText("귀사 제공", {
    x: 4.4, y: 3.3, w: 5.2, h: 0.4,
    fontFace: FONT_HEADER, fontSize: 14, bold: true, color: NAVY,
    align: "left", margin: 0,
  });
  slide.addText([
    { text: "AOI 결과 폴더 read-only 권한",   options: { bullet: true, breakLine: true, color: INK } },
    { text: "운영자 30분/일 검토 시간",          options: { bullet: true, breakLine: true, color: INK } },
    { text: "6개월 운영 데이터 사용권 + 케이스 스터디 동의", options: { bullet: true, color: INK } },
  ], {
    x: 4.4, y: 3.7, w: 5.2, h: 1.2,
    fontFace: FONT_BODY, fontSize: 13, paraSpaceAfter: 3, valign: "top",
  });

  addTechNote(slide, "박스 = read-only 폴더 watch만. SCADA / MES 통합 0. 셋업 반나절.");
}

// ---- Slide 11: Success criteria -----------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "가성불량 70% 안 줄면 박스 회수, 비용 0원", 11);

  slide.addTable([
    [
      { text: "지표",   options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
      { text: "목표",   options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
      { text: "측정 방법", options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
    ],
    ["가성불량 감소율",  { text: "70% ↑",   options: { bold: true, color: NO } },           "첫 30일 vs 종료 30일 비교"],
    ["Escape 건수",       { text: "0건",      options: { bold: true, color: NO } },           "운영자 재검 ground truth"],
    ["운영자 부담 감소",  { text: "월 200hr ↑", options: { bold: true, color: NO } },         "2차 검사 시간 측정"],
  ], {
    x: 0.5, y: 1.7, w: 9.0, colW: [3.0, 1.8, 4.2],
    fontFace: FONT_BODY, fontSize: 14, color: INK,
    border: { pt: 1, color: "E5E7EB" }, rowH: 0.55,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.2, w: 9.0, h: 0.7,
    fill: { color: ACCENT }, line: { color: ACCENT },
  });
  slide.addText("미달 시 → 자동 종료, 박스 회수, 데이터 파기 옵션, 비용 0", {
    x: 0.6, y: 4.2, w: 8.8, h: 0.7,
    fontFace: FONT_BODY, fontSize: 16, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });

  addTechNote(slide, "측정 = 동일 라인·동일 제품·동일 기간, 통계적 유의성 95% 신뢰구간");
}

// ---- Slide 12: Pricing --------------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "라인당 연 절감 ₩10-25억. 2-4개월 만에 본전.", 12);

  // 3 pricing options as cards
  const opts = [
    { title: "A. 영구 라이선스",   price: "₩100-200M",  sub: "사이트당, 라인 무제한", note: "+ 연 지원료 15%" },
    { title: "B. 연구독",           price: "₩30-50M",     sub: "라인당 / 년",            note: "HW 임대 + SW + 지원" },
    { title: "C. 추가 무료 파일럿", price: "₩0",          sub: "다른 라인·다른 vendor",  note: "동일 조건 무상" },
  ];
  opts.forEach((o, i) => {
    const x = 0.5 + i * 3.1;
    slide.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.7, w: 2.9, h: 2.3,
      fill: { color: i === 1 ? NAVY : "F8FAFC" }, line: { color: i === 1 ? NAVY : "E5E7EB" },
    });
    const fg = i === 1 ? "FFFFFF" : INK;
    const accentCol = i === 1 ? ACCENT : NAVY;
    slide.addText(o.title, {
      x: x + 0.1, y: 1.8, w: 2.7, h: 0.4,
      fontFace: FONT_HEADER, fontSize: 14, bold: true, color: fg,
      align: "left", margin: 0,
    });
    slide.addText(o.price, {
      x: x + 0.1, y: 2.2, w: 2.7, h: 0.85,
      fontFace: FONT_HEADER, fontSize: 28, bold: true, color: accentCol,
      align: "left", valign: "middle", margin: 0,
    });
    slide.addText(o.sub, {
      x: x + 0.1, y: 3.05, w: 2.7, h: 0.35,
      fontFace: FONT_BODY, fontSize: 12, color: fg,
      align: "left", margin: 0,
    });
    slide.addText(o.note, {
      x: x + 0.1, y: 3.45, w: 2.7, h: 0.4,
      fontFace: FONT_BODY, fontSize: 10, color: i === 1 ? ICE : MUTED, italic: true,
      align: "left", margin: 0,
    });
  });

  slide.addText("ROI: 라인당 연 ₩10-25억 절감 vs 본 계약가 ₩100-200M  →  회수 2-4개월", {
    x: 0.5, y: 4.3, w: 9.0, h: 0.5,
    fontFace: FONT_HEADER, fontSize: 14, bold: true, color: ACCENT,
    align: "center", margin: 0,
  });

  addTechNote(slide, "ROI 추정: 노무비 ₩2-5만/건 × 일 200건 false-call 절감 가정. 사이트별 측정 후 보정.");
}

// ---- Slide 13: Two tracks -----------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "박스 안에 '안전한 기본 AI' + '자가학습 고급 AI' 둘 다", 13);

  // Production card
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.7, w: 4.3, h: 3.0,
    fill: { color: "F8FAFC" }, line: { color: NAVY, width: 2 },
  });
  slide.addText("기본 — Production v0", {
    x: 0.6, y: 1.8, w: 4.1, h: 0.4,
    fontFace: FONT_HEADER, fontSize: 14, bold: true, color: NAVY,
    align: "left", margin: 0,
  });
  slide.addText("Cost-sensitive Classifier", {
    x: 0.6, y: 2.2, w: 4.1, h: 0.4,
    fontFace: FONT_HEADER, fontSize: 18, bold: true, color: INK,
    align: "left", margin: 0,
  });
  slide.addText([
    { text: "결정론적 (매번 같은 결과)",     options: { bullet: true, breakLine: true, color: INK } },
    { text: "양산 라인 즉시 deploy",            options: { bullet: true, breakLine: true, color: INK } },
    { text: "Escape 위험 0 — 안전 보장",        options: { bullet: true, color: INK } },
  ], {
    x: 0.6, y: 2.7, w: 4.1, h: 1.7,
    fontFace: FONT_BODY, fontSize: 13, paraSpaceAfter: 3, valign: "top",
  });

  // R&D card
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.7, w: 4.3, h: 3.0,
    fill: { color: NAVY }, line: { color: NAVY },
  });
  slide.addText("고급 — R&D 트랙", {
    x: 5.3, y: 1.8, w: 4.1, h: 0.4,
    fontFace: FONT_HEADER, fontSize: 14, bold: true, color: ACCENT,
    align: "left", margin: 0,
  });
  slide.addText("Mamba RL + Lagrangian PPO", {
    x: 5.3, y: 2.2, w: 4.1, h: 0.4,
    fontFace: FONT_HEADER, fontSize: 18, bold: true, color: "FFFFFF",
    align: "left", margin: 0,
  });
  slide.addText([
    { text: "자가학습 — 시간 갈수록 더 똑똑",         options: { bullet: true, breakLine: true, color: ICE } },
    { text: "VisA 입증 (60 iter, escape=0, 5.6× ↓)",  options: { bullet: true, breakLine: true, color: ICE } },
    { text: "Phase 2 fine-tune + 학술 contribution",  options: { bullet: true, color: ICE } },
  ], {
    x: 5.3, y: 2.7, w: 4.1, h: 1.7,
    fontFace: FONT_BODY, fontSize: 13, paraSpaceAfter: 3, valign: "top",
  });

  slide.addText("Production은 신뢰성, R&D는 차별화 — 같은 박스에서 둘 다 동작", {
    x: 0.5, y: 4.85, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 12, italic: true, color: ACCENT, bold: true,
    align: "center", margin: 0,
  });

  addTechNote(slide, "둘 다 같은 어댑터 SDK + 같은 AICS 스키마 위에서 동작. 모드 전환은 config 한 줄.");
}

// ---- Slide 14: Next steps -----------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "14일 내 박스 셋업. 30일 베이스라인. 6개월 평가.", 14);

  const steps = [
    { day: "오늘",     act: "LOI 1페이지 검토", desc: "법무팀 검토" },
    { day: "14일",     act: "박스 셋업",         desc: "배송 + 폴더 마운트 + 운영자 교육 (반나절)" },
    { day: "Day 1-30", act: "베이스라인 측정",    desc: "SHADOW 모드, 라인 영향 0" },
    { day: "Day 31-180", act: "ASSIST 운영",      desc: "운영자 부담 점진 감소" },
    { day: "180일",    act: "성과 평가",         desc: "성공 기준 충족 시 본 계약 / 미달 시 자동 종료" },
  ];

  steps.forEach((s, i) => {
    const y = 1.65 + i * 0.65;
    // Day pill
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 1.6, h: 0.5,
      fill: { color: NAVY }, line: { color: NAVY },
    });
    slide.addText(s.day, {
      x: 0.5, y, w: 1.6, h: 0.5,
      fontFace: FONT_HEADER, fontSize: 12, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    // Act
    slide.addText(s.act, {
      x: 2.3, y, w: 2.5, h: 0.5,
      fontFace: FONT_HEADER, fontSize: 14, bold: true, color: ACCENT,
      align: "left", valign: "middle", margin: 0,
    });
    // Desc
    slide.addText(s.desc, {
      x: 4.9, y, w: 4.6, h: 0.5,
      fontFace: FONT_BODY, fontSize: 12, color: INK,
      align: "left", valign: "middle", margin: 0,
    });
  });

  addTechNote(slide, "셋업 = SMB 마운트 + 박스 부팅 + 운영자 UI 교육 5분. 추가 IT 자원 불필요.");
}

// ---- Slide 15: Closing --------------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: NAVY };

  slide.addText("감사합니다.", {
    x: 0.5, y: 0.6, w: 9.0, h: 0.9,
    fontFace: FONT_HEADER, fontSize: 44, bold: true, color: "FFFFFF",
    align: "center", margin: 0,
  });

  slide.addText("\"6개월 무료 박스. 가성불량 70% 안 줄면 비용 0원. 잃을 게 없습니다.\"", {
    x: 0.5, y: 1.7, w: 9.0, h: 0.6,
    fontFace: FONT_BODY, fontSize: 18, italic: true, color: ACCENT,
    align: "center", margin: 0,
  });

  // Contact panel
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 1.5, y: 2.6, w: 7.0, h: 1.4,
    fill: { color: "FFFFFF" }, line: { color: ICE, width: 1 },
  });
  slide.addText([
    { text: "담당   ", options: { bold: true, color: NAVY } },
    { text: "DrJinHoChoi (최진호)", options: { color: INK, breakLine: true } },
    { text: "GitHub ", options: { bold: true, color: NAVY } },
    { text: "github.com/DrJinHoChoi", options: { color: INK, breakLine: true } },
    { text: "이메일 ", options: { bold: true, color: NAVY } },
    { text: "_(미팅 후 전달)_", options: { color: MUTED, italic: true } },
  ], {
    x: 1.7, y: 2.7, w: 6.6, h: 1.2,
    fontFace: FONT_BODY, fontSize: 14, paraSpaceAfter: 4, valign: "top", margin: 0,
  });

  // Attachments line
  slide.addText("첨부:  ① 6개월 무료 파일럿 1-pager   ② LOI (의향서)   ③ 기술 증거 자료", {
    x: 0.5, y: 4.3, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 13, color: ICE,
    align: "center", margin: 0,
  });

  slide.addText("\"스펙 공개가 라인 코드보다 무겁다\"", {
    x: 0.5, y: 4.95, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 11, italic: true, color: ICE,
    align: "center", margin: 0,
  });
}

// ---- Save ----------------------------------------------------------------
const outPath = "C:/Users/jinho/source/repos/DrJinHoChoi/aoi-sentinel/docs/sales/pitch_deck_kr.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log(`✓ wrote ${outPath}`);
});
