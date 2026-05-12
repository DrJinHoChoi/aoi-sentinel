// Build pitch_deck_general_vision_kr.pptx — plain-language vision deck
// showing the full scope (AOI inspection + SMT yield + CNC machining)
// for non-technical audiences.

const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "DrJinHoChoi";
pres.title = "aoi-sentinel — 큰 그림 일반인용";

const NAVY    = "1E2761";
const ICE     = "CADCFC";
const ACCENT  = "F96167";
const GOLD    = "F9E795";
const INK     = "1A1A2E";
const MUTED   = "6B7280";
const PAGE_BG = "FFFFFF";
const OK      = "2DA44E";
const NO      = "CF222E";
const BLUE2   = "0EA5E9";   // 두 번째 도메인 accent
const PURPLE  = "8B5CF6";   // 세 번째 도메인 accent

const FONT_HEADER = "Malgun Gothic";
const FONT_BODY   = "Malgun Gothic";

function addSlideHeader(slide, headline, slideNum) {
  slide.addText(`${slideNum} / 15`, {
    x: 9.0, y: 0.18, w: 0.85, h: 0.3,
    fontFace: FONT_BODY, fontSize: 10, color: MUTED, align: "right", margin: 0,
  });
  slide.addText("🎯 " + headline, {
    x: 0.5, y: 0.4, w: 9.0, h: 0.95,
    fontFace: FONT_HEADER, fontSize: 21, bold: true, color: NAVY,
    align: "left", valign: "top", margin: 0,
  });
}

function addAnalogy(slide, text) {
  slide.addText("💡  " + text, {
    x: 0.5, y: 5.10, w: 9.0, h: 0.35,
    fontFace: FONT_BODY, fontSize: 11, italic: true, color: ACCENT, bold: true,
    align: "left", margin: 0,
  });
}

// ---- Slide 1: Title ----
{
  const slide = pres.addSlide();
  slide.background = { color: NAVY };

  slide.addText("공장이 더 똑똑해지도록", {
    x: 0.5, y: 1.0, w: 9.0, h: 0.7,
    fontFace: FONT_HEADER, fontSize: 36, color: ICE,
    align: "center", margin: 0,
  });
  slide.addText("돕는 회사입니다.", {
    x: 0.5, y: 1.7, w: 9.0, h: 0.7,
    fontFace: FONT_HEADER, fontSize: 36, bold: true, color: GOLD,
    align: "center", margin: 0,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 3.0, y: 2.7, w: 4.0, h: 0.05, fill: { color: ACCENT }, line: { color: ACCENT },
  });

  slide.addText("aoi-sentinel", {
    x: 0.5, y: 3.0, w: 9.0, h: 0.7,
    fontFace: FONT_HEADER, fontSize: 36, bold: true, color: "FFFFFF",
    align: "center", margin: 0,
  });
  slide.addText("제조업의 AI 도우미", {
    x: 0.5, y: 3.7, w: 9.0, h: 0.5,
    fontFace: FONT_BODY, fontSize: 16, color: ICE, italic: true,
    align: "center", margin: 0,
  });

  slide.addText("작은 박스 하나가 공장의 검사·생산·가공을 도와드립니다.", {
    x: 0.5, y: 4.7, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 12, color: ICE,
    align: "center", margin: 0,
  });

  slide.addText("DrJinHoChoi · 2026", {
    x: 0.5, y: 5.1, w: 9.0, h: 0.3,
    fontFace: FONT_BODY, fontSize: 11, color: ICE,
    align: "center", margin: 0,
  });
}

// ---- Slide 2: Big problem ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "공장에서는 매일 사람이 같은 일을 수천 번 반복합니다.", 2);

  const jobs = [
    { who: "검사 작업자",  ask: "이거 불량인가? 멀쩡한가?",              n: "매일 1,000번", color: NAVY },
    { who: "SMT 작업자", ask: "땜납이 잘 발렸나? 부품 위치 맞나?",     n: "매일 수천 번",  color: BLUE2 },
    { who: "CNC 작업자",  ask: "공구 갈 때 됐나? 속도 맞나? 정밀도 OK?", n: "매일 끊임없이", color: PURPLE },
  ];

  jobs.forEach((j, i) => {
    const y = 1.6 + i * 0.85;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 1.8, h: 0.75, fill: { color: j.color }, line: { color: j.color },
    });
    slide.addText(j.who, {
      x: 0.5, y, w: 1.8, h: 0.75,
      fontFace: FONT_HEADER, fontSize: 13, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText('"' + j.ask + '"', {
      x: 2.5, y, w: 5.4, h: 0.75,
      fontFace: FONT_BODY, fontSize: 13, italic: true, color: INK,
      align: "left", valign: "middle", margin: 0,
    });
    slide.addText(j.n, {
      x: 8.0, y, w: 1.5, h: 0.75,
      fontFace: FONT_BODY, fontSize: 11, color: j.color, bold: true,
      align: "right", valign: "middle", margin: 0,
    });
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.4, w: 9.0, h: 0.55, fill: { color: ICE }, line: { color: ICE },
  });
  slide.addText("→ 매일 같은 판단을 수천 번 → 피로, 야근, 실수", {
    x: 0.6, y: 4.4, w: 8.8, h: 0.55,
    fontFace: FONT_HEADER, fontSize: 14, bold: true, color: NAVY,
    align: "center", valign: "middle", margin: 0,
  });

  addAnalogy(slide, "도서관 사서가 1만 권을 매일 같은 분류로 정리하는 것과 같습니다. AI가 도우면 진짜 중요한 판단만 합니다.");
}

// ---- Slide 3: 3 things we help with ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "같은 박스가 세 가지 다 합니다. 학습 방식이 같기 때문에.", 3);

  const helpers = [
    { num: "1️⃣", title: "AOI 검사 도우미",    desc: "이 부품 불량인지 멀쩡한지 1차 판단",       benefit: "사람은 헷갈리는 것만",      color: NAVY },
    { num: "2️⃣", title: "SMT 생산 도우미",   desc: "땜납·부품·온도가 잘 되어가나 미리 알림",   benefit: "불량 만들기 전 예방",       color: BLUE2 },
    { num: "3️⃣", title: "CNC 가공 도우미",    desc: "공구 마모·진동·온도 보고 자동 조정",       benefit: "가공 사고·낭비 방지",       color: PURPLE },
  ];

  helpers.forEach((h, i) => {
    const x = 0.5 + i * 3.05;
    slide.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.6, w: 2.85, h: 3.2, fill: { color: "F8FAFC" }, line: { color: h.color, width: 2 },
    });
    slide.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.6, w: 2.85, h: 0.6, fill: { color: h.color }, line: { color: h.color },
    });
    slide.addText(h.num + "  " + h.title, {
      x, y: 1.6, w: 2.85, h: 0.6,
      fontFace: FONT_HEADER, fontSize: 14, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });

    slide.addText('"' + h.desc + '"', {
      x: x + 0.15, y: 2.4, w: 2.55, h: 1.2,
      fontFace: FONT_BODY, fontSize: 12, italic: true, color: INK,
      align: "center", valign: "top", margin: 0,
    });

    slide.addShape(pres.shapes.RECTANGLE, {
      x: x + 0.15, y: 4.1, w: 2.55, h: 0.55, fill: { color: ICE }, line: { color: ICE },
    });
    slide.addText("→ " + h.benefit, {
      x: x + 0.15, y: 4.1, w: 2.55, h: 0.55,
      fontFace: FONT_BODY, fontSize: 11, bold: true, color: NAVY,
      align: "center", valign: "middle", margin: 0,
    });
  });

  addAnalogy(slide, "셋 다 \"사람 옆에서 보고 배우고 도와주는 AI\"라는 점에서 똑같습니다.");
}

// ---- Slide 4: AOI 검사 ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "1️⃣ AOI 검사 도우미 — 100건 중 70건 헛수고를 AI가 골라드림", 4);

  // Flow
  const steps = [
    { who: "검사기",   say: "\"100건 불량이에요!\"",            color: MUTED },
    { who: "AI 박스", say: "\"30건은 진짜, 70건은 멀쩡합니다\"", color: ACCENT },
    { who: "작업자",   say: "\"모르겠는 것만 내가 볼게\"",        color: OK },
  ];

  steps.forEach((s, i) => {
    const y = 1.55 + i * 0.95;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 1.7, h: 0.8, fill: { color: s.color }, line: { color: s.color },
    });
    slide.addText(s.who, {
      x: 0.5, y, w: 1.7, h: 0.8,
      fontFace: FONT_HEADER, fontSize: 15, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText(s.say, {
      x: 2.4, y, w: 7.0, h: 0.8,
      fontFace: FONT_BODY, fontSize: 14, color: INK,
      align: "left", valign: "middle", margin: 0,
    });
    if (i < steps.length - 1) {
      slide.addText("↓", {
        x: 1.0, y: y + 0.85, w: 0.7, h: 0.1,
        fontFace: FONT_HEADER, fontSize: 14, color: MUTED, bold: true,
        align: "center", valign: "top", margin: 0,
      });
    }
  });

  slide.addText("오늘 우리 제품. 자동차 부품 공장에서 매일 작업자 야근의 원인.", {
    x: 0.5, y: 4.55, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 12, italic: true, color: NAVY, bold: true,
    align: "center", margin: 0,
  });

  addAnalogy(slide, "공항 X-ray가 매일 100번 의심 표시. 실제 위험은 30번. 보안요원이 70번 헛고생.");
}

// ---- Slide 5: SMT 수율 ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "2️⃣ SMT 생산 도우미 — 검사는 \"이미 망한 거 분류\". 우리는 \"안 망하게 미리 알림\".", 5);

  // Before
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.5, w: 9.0, h: 1.2, fill: { color: "FEF3C7" }, line: { color: "F59E0B", width: 1 },
  });
  slide.addText("지금까지 — 사후 분류", {
    x: 0.6, y: 1.55, w: 8.8, h: 0.3,
    fontFace: FONT_HEADER, fontSize: 12, bold: true, color: "B45309",
    align: "left", margin: 0,
  });
  slide.addText("공정 → 불량 발생 → 검사 → 폐기  (이미 늦었음)", {
    x: 0.6, y: 1.9, w: 8.8, h: 0.75,
    fontFace: FONT_BODY, fontSize: 13, color: INK,
    align: "center", valign: "middle", margin: 0,
  });

  // After
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 2.85, w: 9.0, h: 1.9, fill: { color: "DCFCE7" }, line: { color: OK, width: 1 },
  });
  slide.addText("확장 — 사전 예방", {
    x: 0.6, y: 2.9, w: 8.8, h: 0.3,
    fontFace: FONT_HEADER, fontSize: 12, bold: true, color: OK,
    align: "left", margin: 0,
  });
  slide.addText([
    { text: "공정 → AI: \"땜납 양 줄고 있어요\"  → 미리 조정 → 불량 안 만듦\n", options: { color: INK, breakLine: true } },
    { text: "       AI: \"오븐 온도 흔들려요\"\n", options: { color: INK, breakLine: true } },
    { text: "       AI: \"부품 위치 비뚤어져요\"", options: { color: INK } },
  ], {
    x: 0.6, y: 3.25, w: 8.8, h: 1.4,
    fontFace: FONT_BODY, fontSize: 12, paraSpaceAfter: 2, valign: "top", margin: 0,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.9, w: 9.0, h: 0.4, fill: { color: ACCENT }, line: { color: ACCENT },
  });
  slide.addText("수율 1% 개선만 해도 라인 1개당 연 수십억 ROI", {
    x: 0.6, y: 4.9, w: 8.8, h: 0.4,
    fontFace: FONT_HEADER, fontSize: 12, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });

  addAnalogy(slide, "의사가 병 걸린 후 치료가 아니라, 미리 건강검진으로 예방하는 것과 같습니다.");
}

// ---- Slide 6: CNC ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "3️⃣ CNC 가공 도우미 — 선반·밀링·머시닝센터, 모든 금속 가공기에 적용", 6);

  // What it sees (left)
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.5, w: 4.3, h: 2.2, fill: { color: "F8FAFC" }, line: { color: PURPLE, width: 2 },
  });
  slide.addText("박스가 보는 것", {
    x: 0.6, y: 1.55, w: 4.1, h: 0.35,
    fontFace: FONT_HEADER, fontSize: 13, bold: true, color: PURPLE,
    align: "left", margin: 0,
  });
  slide.addText([
    { text: "스핀들 부하 (커터에 걸리는 힘)", options: { bullet: true, breakLine: true, color: INK } },
    { text: "진동 (chatter)", options: { bullet: true, breakLine: true, color: INK } },
    { text: "온도 (열변형)", options: { bullet: true, breakLine: true, color: INK } },
    { text: "가공 시간 vs 품질", options: { bullet: true, color: INK } },
  ], {
    x: 0.7, y: 1.95, w: 4.0, h: 1.65,
    fontFace: FONT_BODY, fontSize: 12, paraSpaceAfter: 4, valign: "top",
  });

  // What it does (right)
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.1, y: 1.5, w: 4.4, h: 2.2, fill: { color: "F8FAFC" }, line: { color: OK, width: 2 },
  });
  slide.addText("AI 자동 조정", {
    x: 5.2, y: 1.55, w: 4.2, h: 0.35,
    fontFace: FONT_HEADER, fontSize: 13, bold: true, color: OK,
    align: "left", margin: 0,
  });
  slide.addText([
    { text: '"공구 마모 임박 → 교체 권장"', options: { bullet: true, breakLine: true, color: INK } },
    { text: '"속도 줄이면 표면 매끈"', options: { bullet: true, breakLine: true, color: INK } },
    { text: '"스핀들 열변형 -5µm 보정"', options: { bullet: true, color: INK } },
  ], {
    x: 5.3, y: 1.95, w: 4.1, h: 1.65,
    fontFace: FONT_BODY, fontSize: 12, paraSpaceAfter: 4, valign: "top",
  });

  // Benefits
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 3.9, w: 9.0, h: 1.0, fill: { color: PURPLE }, line: { color: PURPLE },
  });
  slide.addText("공구 비용 30%↓   ·   가공 시간 10-20%↓   ·   불량 사전 방지", {
    x: 0.6, y: 3.9, w: 8.8, h: 1.0,
    fontFace: FONT_HEADER, fontSize: 16, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });

  addAnalogy(slide, "자동차에 ABS·차선 이탈 경보가 운전자 도와주는 것과 같습니다. CNC 가공에 운전자 보조 시스템.");
}

// ---- Slide 7: Same algorithm ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "같은 박스가 셋 다 하는 비결 — 학습 방식이 같습니다.", 7);

  const rows = [
    { domain: "AOI 검사",  decision: "작업자: \"불량/멀쩡\" 결정",   color: NAVY },
    { domain: "SMT 생산", decision: "엔지니어: \"온도 +/-\" 결정", color: BLUE2 },
    { domain: "CNC 가공",  decision: "가공사: \"속도 줄여\" 결정",  color: PURPLE },
  ];

  rows.forEach((r, i) => {
    const y = 1.55 + i * 0.75;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 1.8, h: 0.6, fill: { color: r.color }, line: { color: r.color },
    });
    slide.addText(r.domain, {
      x: 0.5, y, w: 1.8, h: 0.6,
      fontFace: FONT_HEADER, fontSize: 13, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText(r.decision + "  →  AI 학습  →  점점 자동", {
      x: 2.5, y, w: 6.9, h: 0.6,
      fontFace: FONT_BODY, fontSize: 13, color: INK,
      align: "left", valign: "middle", margin: 0,
    });
  });

  // Same algorithm arrow
  slide.addText("↑  세 가지 모두 같은 알고리즘  ↑", {
    x: 0.5, y: 3.85, w: 9.0, h: 0.4,
    fontFace: FONT_HEADER, fontSize: 14, italic: true, color: ACCENT, bold: true,
    align: "center", margin: 0,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.4, w: 9.0, h: 0.55, fill: { color: NAVY }, line: { color: NAVY },
  });
  slide.addText("연구개발 한 번, 적용은 무한대.", {
    x: 0.6, y: 4.4, w: 8.8, h: 0.55,
    fontFace: FONT_HEADER, fontSize: 16, bold: true, color: GOLD,
    align: "center", valign: "middle", margin: 0,
  });

  addAnalogy(slide, "한국어 잘 배운 사람이 한국어로 요리·운전·강의 다 하는 것과 같습니다. 같은 언어, 다른 일.");
}

// ---- Slide 8: How it learns ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "매일 작업자 결정을 보고 배웁니다. 신입사원처럼.", 8);

  const days = [
    { day: "Day 1",   say: "다 모르겠어요", who: "작업자 100% 결정", color: NAVY },
    { day: "Day 30",  say: "쉬운 건 알겠어요", who: "작업자 30%만 결정", color: BLUE2 },
    { day: "Day 180", say: "거의 다 알아요", who: "작업자 5%만 결정",  color: OK },
  ];

  days.forEach((d, i) => {
    const y = 1.6 + i * 1.0;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 1.5, h: 0.85, fill: { color: d.color }, line: { color: d.color },
    });
    slide.addText(d.day, {
      x: 0.5, y, w: 1.5, h: 0.85,
      fontFace: FONT_HEADER, fontSize: 14, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText([
      { text: "박스: \"" + d.say + "\"\n", options: { bold: true, color: d.color, breakLine: true } },
      { text: d.who, options: { color: INK, italic: true } },
    ], {
      x: 2.2, y, w: 7.2, h: 0.85,
      fontFace: FONT_BODY, fontSize: 13, valign: "middle", paraSpaceAfter: 2, margin: 0,
    });
  });

  slide.addText("학습 데이터는 공장 안에서만. 외부로 안 나감.", {
    x: 0.5, y: 4.7, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 13, italic: true, color: NAVY, bold: true,
    align: "center", margin: 0,
  });

  addAnalogy(slide, "좋은 비서가 6개월 후 사장 일정을 알아서 잡는 것과 같습니다.");
}

// ---- Slide 9: Safety ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "박스가 모르겠으면 무조건 사람에게. 사고 0건 수학적 보장.", 9);

  // Main rule
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.5, w: 9.0, h: 1.3, fill: { color: ICE }, line: { color: NAVY, width: 2 },
  });
  slide.addText("박스 \"모르겠어요\"  →  무조건 작업자에게 escalate", {
    x: 0.6, y: 1.55, w: 8.8, h: 0.55,
    fontFace: FONT_HEADER, fontSize: 17, bold: true, color: NAVY,
    align: "center", valign: "middle", margin: 0,
  });
  slide.addText("작업자 = 최종 결정", {
    x: 0.6, y: 2.15, w: 8.8, h: 0.6,
    fontFace: FONT_HEADER, fontSize: 14, color: INK,
    align: "center", valign: "middle", margin: 0,
  });

  // Three zeros
  const zeros = [
    { what: "검사 사고", color: NAVY },
    { what: "생산 사고", color: BLUE2 },
    { what: "가공 사고", color: PURPLE },
  ];
  zeros.forEach((z, i) => {
    const x = 0.5 + i * 3.05;
    slide.addShape(pres.shapes.RECTANGLE, {
      x, y: 3.0, w: 2.85, h: 1.0, fill: { color: z.color }, line: { color: z.color },
    });
    slide.addText("0건", {
      x, y: 3.0, w: 2.85, h: 0.6,
      fontFace: FONT_HEADER, fontSize: 28, bold: true, color: GOLD,
      align: "center", valign: "bottom", margin: 0,
    });
    slide.addText(z.what, {
      x, y: 3.6, w: 2.85, h: 0.35,
      fontFace: FONT_BODY, fontSize: 12, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
  });

  slide.addText("세 가지 적용 모두 같은 안전 framework", {
    x: 0.5, y: 4.15, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 13, italic: true, color: ACCENT, bold: true,
    align: "center", margin: 0,
  });

  slide.addText("만약 단 한 번이라도 사고 발생 시 → 박스 자동으로 보수적 모드 복귀", {
    x: 0.5, y: 4.65, w: 9.0, h: 0.35,
    fontFace: FONT_BODY, fontSize: 11, italic: true, color: NAVY,
    align: "center", margin: 0,
  });

  addAnalogy(slide, "자율주행이 헷갈리면 운전자에게 핸들 넘기는 것과 같습니다. 안전이 최우선.");
}

// ---- Slide 10: Data stays local ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "회사 비밀(부품 설계·가공 노하우)은 공장 밖으로 안 나갑니다.", 10);

  // Inside the factory
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.55, w: 9.0, h: 2.6, fill: { color: ICE }, line: { color: NAVY, width: 2 },
  });
  slide.addText("공장 내부", {
    x: 0.6, y: 1.6, w: 8.8, h: 0.35,
    fontFace: FONT_HEADER, fontSize: 13, bold: true, color: NAVY,
    align: "left", margin: 0,
  });

  // Three data sources flowing to box
  const sources = [
    { label: "검사 데이터", color: NAVY },
    { label: "SMT 데이터", color: BLUE2 },
    { label: "CNC 데이터", color: PURPLE },
  ];
  sources.forEach((s, i) => {
    const y = 2.05 + i * 0.5;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 1.0, y, w: 2.5, h: 0.4, fill: { color: s.color }, line: { color: s.color },
    });
    slide.addText(s.label, {
      x: 1.0, y, w: 2.5, h: 0.4,
      fontFace: FONT_BODY, fontSize: 12, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText("→", {
      x: 3.6, y, w: 0.4, h: 0.4,
      fontFace: FONT_HEADER, fontSize: 18, color: NAVY, bold: true,
      align: "center", valign: "middle", margin: 0,
    });
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 4.2, y: 2.3, w: 2.0, h: 0.95, fill: { color: ACCENT }, line: { color: ACCENT },
  });
  slide.addText("박스\n(AI)", {
    x: 4.2, y: 2.3, w: 2.0, h: 0.95,
    fontFace: FONT_HEADER, fontSize: 14, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });
  slide.addText("→", {
    x: 6.4, y: 2.55, w: 0.4, h: 0.4,
    fontFace: FONT_HEADER, fontSize: 24, color: NAVY, bold: true,
    align: "center", valign: "middle", margin: 0,
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 7.0, y: 2.45, w: 2.0, h: 0.65, fill: { color: OK }, line: { color: OK },
  });
  slide.addText("학습 → 더 똑똑", {
    x: 7.0, y: 2.45, w: 2.0, h: 0.65,
    fontFace: FONT_HEADER, fontSize: 12, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });

  // Big "0 outside"
  slide.addText("외부 전송: 0건", {
    x: 0.6, y: 3.65, w: 8.8, h: 0.45,
    fontFace: FONT_HEADER, fontSize: 18, bold: true, color: NO,
    align: "center", margin: 0,
  });

  slide.addText("자동차 부품 비밀·반도체 IP·정밀가공 노하우 모두 안전.", {
    x: 0.5, y: 4.3, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 12, italic: true, color: NAVY, bold: true,
    align: "center", margin: 0,
  });

  addAnalogy(slide, "가족 비밀 일기를 집 밖으로 안 가져가는 것과 같습니다.");
}

// ---- Slide 11: Korean industry fit ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "한국이 강한 모든 제조 분야에 다 적용됩니다.", 11);

  slide.addTable([
    [
      { text: "산업",          options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
      { text: "우리 박스 역할", options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
    ],
    [{ text: "자동차 전장",  options: { bold: true, color: INK } },  "AOI 검사 + SMT 수율  (현대모비스, LG이노텍)"],
    [{ text: "반도체",       options: { bold: true, color: INK } },  "HBM 보드 검사 + 패키징  (삼성전기, SK하이닉스)"],
    [{ text: "로봇",         options: { bold: true, color: INK } },  "부품 검사 + CNC 가공  (현대로보틱스, 두산로보틱스)"],
    [{ text: "AI 서버",      options: { bold: true, color: INK } },  "NVIDIA·HBM 보드 검사  (Tier-1 OEM)"],
    [{ text: "EV 배터리",    options: { bold: true, color: INK } },  "BMS 보드 + 셀 housing CNC  (LG에너지, 삼성SDI)"],
    [{ text: "정밀가공 SMB", options: { bold: true, color: INK } },  "CNC 적응형 제어  (3만+ 한국 shop)"],
  ], {
    x: 0.5, y: 1.55, w: 9.0, colW: [2.3, 6.7],
    fontFace: FONT_BODY, fontSize: 12, color: INK,
    border: { pt: 1, color: "E5E7EB" }, rowH: 0.45,
  });

  slide.addText("→ 한국 제조업 전체가 잠재 고객", {
    x: 0.5, y: 4.85, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 13, italic: true, color: ACCENT, bold: true,
    align: "center", margin: 0,
  });

  addAnalogy(slide, "한국은 세계 1위 반도체·디스플레이·자동차·조선 + 정밀가공 강국. 같은 박스가 모든 분야에.");
}

// ---- Slide 12: 5-year vision ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "검사 박스 1개 → 공장 전체를 돕는 AI 플랫폼", 12);

  const years = [
    { y: "2026 (현재)", w: "AOI 검사 도우미",          c: NAVY,    note: "시작점" },
    { y: "2027",        w: "+ SMT 생산 도우미",         c: BLUE2,   note: "" },
    { y: "2028",        w: "+ CNC 가공 도우미",         c: PURPLE,  note: "" },
    { y: "2030",        w: "제조업 AI 표준 플랫폼",      c: ACCENT,  note: "글로벌 표준" },
    { y: "2032+",       w: "AI 인증 권한 보유 기관",     c: OK,      note: "권력" },
  ];

  years.forEach((y, i) => {
    const yy = 1.6 + i * 0.6;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: yy, w: 1.8, h: 0.5, fill: { color: y.c }, line: { color: y.c },
    });
    slide.addText(y.y, {
      x: 0.5, y: yy, w: 1.8, h: 0.5,
      fontFace: FONT_HEADER, fontSize: 12, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText(y.w, {
      x: 2.5, y: yy, w: 5.0, h: 0.5,
      fontFace: FONT_BODY, fontSize: 13, color: INK, bold: true,
      align: "left", valign: "middle", margin: 0,
    });
    if (y.note) {
      slide.addText("← " + y.note, {
        x: 7.6, y: yy, w: 1.9, h: 0.5,
        fontFace: FONT_BODY, fontSize: 11, italic: true, color: y.c, bold: true,
        align: "left", valign: "middle", margin: 0,
      });
    }
  });

  slide.addText("같은 박스, 같은 알고리즘, 점점 더 많은 일.", {
    x: 0.5, y: 4.8, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 12, italic: true, color: ACCENT, bold: true,
    align: "center", margin: 0,
  });

  addAnalogy(slide, "카카오톡이 메시지 → 결제 → 택시 → 은행 → 게임으로 확장한 것과 같습니다. 한 platform이 점점 커짐.");
}

// ---- Slide 13: Government mandate ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "5년 후 AI 솔루션은 정부 인증 통과해야 판매 가능. 우리가 그 인증 기관 자리.", 13);

  const t = [
    { y: "2024", w: "EU AI 법 발효",          done: true },
    { y: "2025", w: "한국 AI 기본법 통과",     done: true },
    { y: "2026", w: "ISO 국제 표준 가속",      done: true },
    { y: "2027", w: "AI 의무 인증 시작",       accent: true },
    { y: "2030", w: "글로벌 AI audit 의무화",  accent: true },
  ];

  t.forEach((m, i) => {
    const y = 1.55 + i * 0.55;
    const c = m.accent ? ACCENT : (m.done ? NAVY : MUTED);
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 1.3, h: 0.45, fill: { color: c }, line: { color: c },
    });
    slide.addText(m.y, {
      x: 0.5, y, w: 1.3, h: 0.45,
      fontFace: FONT_HEADER, fontSize: 13, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText(m.w, {
      x: 1.95, y, w: 7.5, h: 0.45,
      fontFace: FONT_BODY, fontSize: 13, color: m.accent ? ACCENT : INK, bold: !!m.accent,
      align: "left", valign: "middle", margin: 0,
    });
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.4, w: 9.0, h: 0.55, fill: { color: NAVY }, line: { color: NAVY },
  });
  slide.addText("모든 제조 AI는 표준 통과 필수. 우리 = 표준 정의자.", {
    x: 0.6, y: 4.4, w: 8.8, h: 0.55,
    fontFace: FONT_BODY, fontSize: 13, italic: true, color: GOLD, bold: true,
    align: "center", valign: "middle", margin: 0,
  });

  addAnalogy(slide, "자동차 안전벨트 의무화처럼, AI도 5년 후엔 인증된 것만 사용 가능. 우리는 그 인증 기관.");
}

// ---- Slide 14: Validation result ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "검사 도우미 컴퓨터 실험 — 60번 학습에 진짜 불량 놓침 0건", 14);

  // Two big stat cards
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.6, w: 4.3, h: 2.6,
    fill: { color: "F8FAFC" }, line: { color: OK, width: 2 },
  });
  slide.addText("0건", {
    x: 0.5, y: 1.7, w: 4.3, h: 1.4,
    fontFace: FONT_HEADER, fontSize: 86, bold: true, color: OK,
    align: "center", valign: "bottom", margin: 0,
  });
  slide.addText("놓친 불량", {
    x: 0.5, y: 3.2, w: 4.3, h: 0.5,
    fontFace: FONT_BODY, fontSize: 16, bold: true, color: INK,
    align: "center", margin: 0,
  });
  slide.addText("60번 학습 동안", {
    x: 0.5, y: 3.7, w: 4.3, h: 0.5,
    fontFace: FONT_BODY, fontSize: 12, color: MUTED,
    align: "center", margin: 0,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.6, w: 4.3, h: 2.6,
    fill: { color: "F8FAFC" }, line: { color: NAVY, width: 2 },
  });
  slide.addText("5배 ↓", {
    x: 5.2, y: 1.7, w: 4.3, h: 1.4,
    fontFace: FONT_HEADER, fontSize: 78, bold: true, color: NAVY,
    align: "center", valign: "bottom", margin: 0,
  });
  slide.addText("작업자 부담", {
    x: 5.2, y: 3.2, w: 4.3, h: 0.5,
    fontFace: FONT_BODY, fontSize: 16, bold: true, color: INK,
    align: "center", margin: 0,
  });
  slide.addText("1280 → 230 / 회", {
    x: 5.2, y: 3.7, w: 4.3, h: 0.5,
    fontFace: FONT_BODY, fontSize: 12, color: MUTED,
    align: "center", margin: 0,
  });

  slide.addText("→ SMT 수율·CNC 가공도 같은 알고리즘이라 비슷한 결과 기대", {
    x: 0.5, y: 4.45, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 12, italic: true, color: ACCENT, bold: true,
    align: "center", margin: 0,
  });

  addAnalogy(slide, "한 분야에서 잘 학습된 AI가 다른 분야에서도 잘 작동하는 게 증명된 셈.");
}

// ---- Slide 15: Closing ----
{
  const slide = pres.addSlide();
  slide.background = { color: NAVY };

  slide.addText("공장이 더 똑똑해지면,", {
    x: 0.5, y: 0.7, w: 9.0, h: 0.55,
    fontFace: FONT_HEADER, fontSize: 28, bold: true, color: "FFFFFF",
    align: "center", margin: 0,
  });
  slide.addText("작업자가 더 가치 있는 일을 합니다.", {
    x: 0.5, y: 1.3, w: 9.0, h: 0.55,
    fontFace: FONT_HEADER, fontSize: 28, bold: true, color: GOLD,
    align: "center", margin: 0,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 3.0, y: 2.15, w: 4.0, h: 0.05, fill: { color: ACCENT }, line: { color: ACCENT },
  });

  // Two columns
  slide.addText("필요한 것:", {
    x: 0.7, y: 2.4, w: 4.0, h: 0.35,
    fontFace: FONT_HEADER, fontSize: 13, bold: true, color: ACCENT,
    align: "left", margin: 0,
  });
  slide.addText([
    { text: "첫 공장 1곳 (검사부터 시작)\n", options: { bullet: true, breakLine: true, color: ICE } },
    { text: "같이 일할 동료\n", options: { bullet: true, breakLine: true, color: ICE } },
    { text: "정부·표준 기관 다리", options: { bullet: true, color: ICE } },
  ], {
    x: 0.7, y: 2.8, w: 4.0, h: 1.6,
    fontFace: FONT_BODY, fontSize: 12, paraSpaceAfter: 4, valign: "top", margin: 0,
  });

  slide.addText("도와주실 분:", {
    x: 5.3, y: 2.4, w: 4.0, h: 0.35,
    fontFace: FONT_HEADER, fontSize: 13, bold: true, color: ACCENT,
    align: "left", margin: 0,
  });
  slide.addText([
    { text: "자동차·반도체·로봇·정밀가공 회사 임원\n", options: { bullet: true, breakLine: true, color: ICE } },
    { text: "AI·표준화 정책 담당자\n", options: { bullet: true, breakLine: true, color: ICE } },
    { text: "투자자\n", options: { bullet: true, breakLine: true, color: ICE } },
    { text: "가공·검사 베테랑 작업자", options: { bullet: true, color: ICE } },
  ], {
    x: 5.3, y: 2.8, w: 4.0, h: 1.6,
    fontFace: FONT_BODY, fontSize: 12, paraSpaceAfter: 4, valign: "top", margin: 0,
  });

  // Contact
  slide.addText("DrJinHoChoi  ·  github.com/DrJinHoChoi", {
    x: 0.5, y: 4.7, w: 9.0, h: 0.35,
    fontFace: FONT_BODY, fontSize: 12, color: ICE,
    align: "center", margin: 0,
  });

  slide.addText("\"같은 박스가 공장 전체를 돕습니다.\"", {
    x: 0.5, y: 5.1, w: 9.0, h: 0.3,
    fontFace: FONT_BODY, fontSize: 11, italic: true, color: ACCENT,
    align: "center", margin: 0,
  });
}

// ---- Save ----
const outPath = "C:/Users/jinho/source/repos/DrJinHoChoi/aoi-sentinel/docs/sales/pitch_deck_general_vision_kr.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log(`✓ wrote ${outPath}`);
});
