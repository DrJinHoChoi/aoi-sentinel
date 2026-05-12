// Build pitch_deck_general_kr.pptx — plain-language deck for non-technical
// audiences (general public, press, government, factory operators, family).

const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "DrJinHoChoi";
pres.title = "aoi-sentinel — 일반인용 발표자료";

const NAVY    = "1E2761";
const ICE     = "CADCFC";
const ACCENT  = "F96167";
const GOLD    = "F9E795";
const INK     = "1A1A2E";
const MUTED   = "6B7280";
const PAGE_BG = "FFFFFF";
const OK      = "2DA44E";
const NO      = "CF222E";
const WARM    = "FFF7ED";  // warm cream for friendly tone

const FONT_HEADER = "Malgun Gothic";
const FONT_BODY   = "Malgun Gothic";

function addSlideHeader(slide, headline, slideNum) {
  slide.addText(`${slideNum} / 15`, {
    x: 9.0, y: 0.18, w: 0.85, h: 0.3,
    fontFace: FONT_BODY, fontSize: 10, color: MUTED, align: "right", margin: 0,
  });
  slide.addText("🎯 " + headline, {
    x: 0.5, y: 0.4, w: 9.0, h: 0.95,
    fontFace: FONT_HEADER, fontSize: 22, bold: true, color: NAVY,
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

  slide.addText("aoi-sentinel", {
    x: 0.5, y: 1.1, w: 9.0, h: 1.0,
    fontFace: FONT_HEADER, fontSize: 56, bold: true, color: "FFFFFF",
    align: "center", margin: 0,
  });

  slide.addText("공장에서 매일 사람이 하는 검사 일,", {
    x: 0.5, y: 2.2, w: 9.0, h: 0.55,
    fontFace: FONT_HEADER, fontSize: 22, color: ICE,
    align: "center", margin: 0,
  });
  slide.addText("AI가 도와드립니다.", {
    x: 0.5, y: 2.8, w: 9.0, h: 0.55,
    fontFace: FONT_HEADER, fontSize: 22, color: GOLD, bold: true,
    align: "center", margin: 0,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 3.0, y: 3.7, w: 4.0, h: 0.05, fill: { color: ACCENT }, line: { color: ACCENT },
  });

  slide.addText("일반인 · 언론 · 정부 · 일반 직원 · 가족 대상", {
    x: 0.5, y: 4.0, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 13, color: ICE,
    align: "center", margin: 0,
  });

  slide.addText("DrJinHoChoi · 2026", {
    x: 0.5, y: 5.0, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 12, color: ICE,
    align: "center", margin: 0,
  });
}

// ---- Slide 2: 공장에서 매일 일어나는 일 ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "자동차 부품 공장에서는 매일 작은 부품 100만 개를 검사합니다", 2);

  // Flow
  const flow = [
    { label: "자동 검사기", sub: "\"이 부품 불량이에요!\" (NG 표시)", color: NAVY },
    { label: "사람 작업자", sub: "\"어디 봅시다... 맞네\" 또는 \"아닌데?\"", color: ACCENT },
    { label: "결정", sub: "폐기 / 재가공 / 통과", color: OK },
  ];

  flow.forEach((f, i) => {
    const y = 1.6 + i * 1.0;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 1.0, y, w: 2.5, h: 0.8, fill: { color: f.color }, line: { color: f.color },
    });
    slide.addText(f.label, {
      x: 1.0, y, w: 2.5, h: 0.8,
      fontFace: FONT_HEADER, fontSize: 15, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText(f.sub, {
      x: 3.8, y, w: 5.2, h: 0.8,
      fontFace: FONT_BODY, fontSize: 13, color: INK,
      align: "left", valign: "middle", margin: 0,
    });
    if (i < flow.length - 1) {
      slide.addText("↓", {
        x: 1.5, y: y + 0.85, w: 1.5, h: 0.15,
        fontFace: FONT_HEADER, fontSize: 14, color: MUTED, bold: true,
        align: "center", valign: "top", margin: 0,
      });
    }
  });

  addAnalogy(slide, "공항에서 X-ray가 가방을 빨갛게 표시하면, 사람이 다시 확인하는 것과 같습니다.");
}

// ---- Slide 3: 문제 ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "검사기가 \"불량\" 한 것 100건 중 진짜 불량은 30건뿐. 나머지 70건은 헛수고.", 3);

  // Big 100 box (left)
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.5, w: 3.5, h: 3.0,
    fill: { color: ICE }, line: { color: NAVY, width: 2 },
  });
  slide.addText("100건", {
    x: 0.5, y: 1.6, w: 3.5, h: 1.0,
    fontFace: FONT_HEADER, fontSize: 56, bold: true, color: NAVY,
    align: "center", valign: "bottom", margin: 0,
  });
  slide.addText("검사기가 \"불량\" 표시", {
    x: 0.5, y: 2.7, w: 3.5, h: 0.5,
    fontFace: FONT_BODY, fontSize: 14, color: INK,
    align: "center", margin: 0,
  });
  slide.addText("매일 반복", {
    x: 0.5, y: 3.3, w: 3.5, h: 0.5,
    fontFace: FONT_BODY, fontSize: 12, italic: true, color: MUTED,
    align: "center", margin: 0,
  });

  // Right split: 30 real + 70 false
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 4.4, y: 1.5, w: 5.0, h: 1.4, fill: { color: NO }, line: { color: NO },
  });
  slide.addText("진짜 불량  30건", {
    x: 4.5, y: 1.5, w: 4.9, h: 0.5,
    fontFace: FONT_HEADER, fontSize: 18, bold: true, color: "FFFFFF",
    align: "left", valign: "middle", margin: 0,
  });
  slide.addText("→ 사람이 확인 후 폐기 / 재가공", {
    x: 4.5, y: 2.0, w: 4.9, h: 0.85,
    fontFace: FONT_BODY, fontSize: 13, color: "FFFFFF",
    align: "left", valign: "middle", margin: 0,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 4.4, y: 3.05, w: 5.0, h: 1.45, fill: { color: ACCENT }, line: { color: ACCENT },
  });
  slide.addText("사실은 멀쩡  70건  ← 헛수고", {
    x: 4.5, y: 3.05, w: 4.9, h: 0.5,
    fontFace: FONT_HEADER, fontSize: 18, bold: true, color: "FFFFFF",
    align: "left", valign: "middle", margin: 0,
  });
  slide.addText("→ 사람이 확인 후 그냥 통과 (시간 낭비)", {
    x: 4.5, y: 3.55, w: 4.9, h: 0.9,
    fontFace: FONT_BODY, fontSize: 13, color: "FFFFFF",
    align: "left", valign: "middle", margin: 0,
  });

  addAnalogy(slide, "매일 야근, 라인 처리량 감소, 작업자 피로 — 모두 이 70건의 헛수고에서 옵니다.");
}

// ---- Slide 4: 우리가 하는 일 ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "검사기 옆에 작은 박스를 놓으면, AI가 작업자처럼 확인을 도와줍니다.", 4);

  // Center: 박스
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.6, w: 2.2, h: 0.9, fill: { color: NAVY }, line: { color: NAVY },
  });
  slide.addText("검사기", {
    x: 0.5, y: 1.6, w: 2.2, h: 0.9,
    fontFace: FONT_HEADER, fontSize: 16, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });
  slide.addText("→", {
    x: 2.8, y: 1.85, w: 0.4, h: 0.4,
    fontFace: FONT_HEADER, fontSize: 24, color: MUTED, bold: true,
    align: "center", valign: "middle", margin: 0,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 3.3, y: 1.5, w: 2.6, h: 1.1, fill: { color: ACCENT }, line: { color: ACCENT },
  });
  slide.addText("aoi-sentinel\n박스 (AI)", {
    x: 3.3, y: 1.5, w: 2.6, h: 1.1,
    fontFace: FONT_HEADER, fontSize: 15, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });

  slide.addText("→", {
    x: 6.0, y: 1.85, w: 0.4, h: 0.4,
    fontFace: FONT_HEADER, fontSize: 24, color: MUTED, bold: true,
    align: "center", valign: "middle", margin: 0,
  });

  // Three outcomes on right
  const outcomes = [
    { icon: "✓", label: "확실히 멀쩡 → 통과 (자동)",   color: OK },
    { icon: "✗", label: "확실히 불량 → 폐기 (자동)",   color: NO },
    { icon: "?", label: "모르겠음 → 작업자에게",        color: "F59E0B" },
  ];
  outcomes.forEach((o, i) => {
    const y = 1.45 + i * 0.45;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 6.5, y, w: 3.0, h: 0.4, fill: { color: o.color }, line: { color: o.color },
    });
    slide.addText(o.icon + "  " + o.label, {
      x: 6.5, y, w: 3.0, h: 0.4,
      fontFace: FONT_BODY, fontSize: 11, bold: true, color: "FFFFFF",
      align: "left", valign: "middle", margin: 0,
    });
  });

  // Below: punchline
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 3.4, w: 9.0, h: 1.0, fill: { color: ICE }, line: { color: ICE },
  });
  slide.addText("작업자는 \"확실하지 않은 것\"만 보면 됩니다.", {
    x: 0.6, y: 3.4, w: 8.8, h: 1.0,
    fontFace: FONT_HEADER, fontSize: 22, bold: true, color: NAVY,
    align: "center", valign: "middle", margin: 0,
  });

  addAnalogy(slide, "보조 선생님이 시험지 1차 채점하고, 헷갈리는 것만 본 선생님에게 가져오는 것과 같습니다.");
}

// ---- Slide 5: 박스가 어떻게 똑똑해지나 ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "매일 작업자가 결정하는 걸 보고, 박스가 스스로 배워나갑니다.", 5);

  const days = [
    { day: "Day 1",   ask: "100건",   user: "100건 다 확인",  desc: "음... 잘 모르겠어요"          },
    { day: "Day 30",  ask: "30건",    user: "30건만 확인",     desc: "이런 건 알겠어요!"             },
    { day: "Day 180", ask: "5건",     user: "5건만 확인",       desc: "이제 거의 다 알아요"          },
  ];

  days.forEach((d, i) => {
    const y = 1.6 + i * 1.0;
    // Day pill
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 1.3, h: 0.85, fill: { color: NAVY }, line: { color: NAVY },
    });
    slide.addText(d.day, {
      x: 0.5, y, w: 1.3, h: 0.85,
      fontFace: FONT_HEADER, fontSize: 16, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    // Description
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 2.0, y, w: 7.5, h: 0.85, fill: { color: i === 0 ? ICE : (i === 1 ? "FEF3C7" : "DCFCE7") }, line: { color: i === 0 ? NAVY : (i === 1 ? "F59E0B" : OK), width: 1 },
    });
    slide.addText([
      { text: "박스: \"" + d.desc + "\"  ", options: { bold: true, color: i === 0 ? NAVY : (i === 1 ? "B45309" : OK) } },
      { text: "(작업자에게 묻는 횟수: " + d.ask + ")\n", options: { color: INK, breakLine: true } },
      { text: "작업자: " + d.user, options: { color: INK, italic: true } },
    ], {
      x: 2.15, y: y + 0.05, w: 7.2, h: 0.75,
      fontFace: FONT_BODY, fontSize: 12, valign: "middle", paraSpaceAfter: 2, margin: 0,
    });
  });

  addAnalogy(slide, "신입사원이 6개월간 사수 옆에서 일 배우는 것과 같습니다. 점점 혼자 할 수 있어집니다.");
}

// ---- Slide 6: 안전 약속 ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "진짜 불량을 놓치는 일은 절대 없습니다. 수학으로 보장합니다.", 6);

  // Top: rule
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.5, w: 9.0, h: 1.4, fill: { color: ICE }, line: { color: NAVY, width: 2 },
  });
  slide.addText("박스가 \"모르겠어요\" 하면  →  무조건 작업자에게 물어봄", {
    x: 0.6, y: 1.55, w: 8.8, h: 0.5,
    fontFace: FONT_HEADER, fontSize: 16, bold: true, color: NAVY,
    align: "center", valign: "middle", margin: 0,
  });
  slide.addText("그 결정은 사람이 함  →  사고 0건", {
    x: 0.6, y: 2.1, w: 8.8, h: 0.7,
    fontFace: FONT_HEADER, fontSize: 14, color: INK,
    align: "center", valign: "middle", margin: 0,
  });

  // Equation
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 3.1, w: 9.0, h: 0.8, fill: { color: NAVY }, line: { color: NAVY },
  });
  slide.addText("작업자 + 박스 = 둘 중 하나는 반드시 확인  →  둘 다 못 본 사고 = 0건", {
    x: 0.6, y: 3.1, w: 8.8, h: 0.8,
    fontFace: FONT_HEADER, fontSize: 14, bold: true, color: GOLD,
    align: "center", valign: "middle", margin: 0,
  });

  // Bottom: failsafe
  slide.addText("만약 단 한 번이라도 사고 발생 시  →  박스 자동으로 보수적 모드로 복귀", {
    x: 0.5, y: 4.2, w: 9.0, h: 0.5,
    fontFace: FONT_BODY, fontSize: 13, italic: true, color: ACCENT, bold: true,
    align: "center", margin: 0,
  });

  addAnalogy(slide, "자율주행이 헷갈리면 운전자에게 핸들 넘기는 것과 같습니다. 안전이 최우선.");
}

// ---- Slide 7: 3단계 ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "처음 한 달은 박스가 일 안 합니다. 그 다음에 천천히 자동.", 7);

  const stages = [
    { num: "1단계", dur: "한 달",         act: "박스는 의견만 표시",        user: "작업자 100% 결정",      color: NAVY },
    { num: "2단계", dur: "5-6개월",       act: "박스가 확실한 것만 자동",   user: "모르면 작업자에게",     color: ACCENT },
    { num: "3단계", dur: "이후 (옵션)",   act: "박스가 거의 다 결정",       user: "작업자는 KPI만 모니터", color: OK },
  ];

  stages.forEach((s, i) => {
    const y = 1.55 + i * 1.05;
    // num + duration
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 1.8, h: 0.9, fill: { color: s.color }, line: { color: s.color },
    });
    slide.addText(s.num, {
      x: 0.5, y, w: 1.8, h: 0.45,
      fontFace: FONT_HEADER, fontSize: 16, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText(s.dur, {
      x: 0.5, y: y + 0.45, w: 1.8, h: 0.45,
      fontFace: FONT_BODY, fontSize: 11, color: GOLD,
      align: "center", valign: "middle", margin: 0,
    });

    // Act + user
    slide.addText("박스: " + s.act, {
      x: 2.5, y: y + 0.05, w: 6.9, h: 0.4,
      fontFace: FONT_HEADER, fontSize: 14, bold: true, color: s.color,
      align: "left", valign: "middle", margin: 0,
    });
    slide.addText("작업자: " + s.user, {
      x: 2.5, y: y + 0.5, w: 6.9, h: 0.4,
      fontFace: FONT_BODY, fontSize: 12, color: INK,
      align: "left", valign: "middle", margin: 0,
    });
  });

  addAnalogy(slide, "새 직원에게 처음엔 보게 하고, 익숙해지면 쉬운 일부터 맡기고, 마지막엔 알아서 일하게 하는 것과 같습니다.");
}

// ---- Slide 8: 6개월 무료 ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "6개월 무료로 써보세요. 효과 미달이면 비용 0원, 박스 회수.", 8);

  // Big ₩0
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.5, w: 3.5, h: 2.5, fill: { color: NAVY }, line: { color: NAVY },
  });
  slide.addText("₩0", {
    x: 0.5, y: 1.6, w: 3.5, h: 1.7,
    fontFace: FONT_HEADER, fontSize: 92, bold: true, color: "FFFFFF",
    align: "center", valign: "bottom", margin: 0,
  });
  slide.addText("박스 + 셋업 + 6개월 지원", {
    x: 0.5, y: 3.35, w: 3.5, h: 0.5,
    fontFace: FONT_BODY, fontSize: 12, color: ICE,
    align: "center", margin: 0,
  });

  // Right table
  const items = [
    { k: "비용",   v: "0원 (모두 무료)" },
    { k: "기간",   v: "6개월" },
    { k: "약속",   v: "가성불량 70% 줄지 않으면 자동 종료" },
    { k: "위험",   v: "0 — 라인 운영에 영향 없음" },
  ];
  items.forEach((it, i) => {
    const y = 1.55 + i * 0.55;
    slide.addText(it.k, {
      x: 4.4, y, w: 1.3, h: 0.5,
      fontFace: FONT_HEADER, fontSize: 14, bold: true, color: NAVY,
      align: "left", valign: "middle", margin: 0,
    });
    slide.addText(it.v, {
      x: 5.8, y, w: 3.7, h: 0.5,
      fontFace: FONT_BODY, fontSize: 13, color: INK,
      align: "left", valign: "middle", margin: 0,
    });
  });

  // Bottom emphasis
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 4.4, y: 4.0, w: 5.1, h: 0.6, fill: { color: ACCENT }, line: { color: ACCENT },
  });
  slide.addText("마트 시식 코너 같은 거예요", {
    x: 4.4, y: 4.0, w: 5.1, h: 0.6,
    fontFace: FONT_BODY, fontSize: 13, bold: true, italic: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });

  addAnalogy(slide, "맘에 안 들면 그냥 두고 가시면 됩니다.");
}

// ---- Slide 9: 어떤 검사기든 OK ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "일본 Saki든, 한국 Koh Young이든, 어느 회사 검사기든 다 됩니다.", 9);

  // 4 vendor boxes on left
  const vendors = ["Saki (일본)", "Koh Young (한국)", "Mycronic (스웨덴)", "기타 vendor"];
  vendors.forEach((v, i) => {
    const y = 1.55 + i * 0.7;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 3.5, h: 0.55, fill: { color: ICE }, line: { color: NAVY, width: 1 },
    });
    slide.addText(v, {
      x: 0.6, y, w: 3.4, h: 0.55,
      fontFace: FONT_HEADER, fontSize: 13, color: INK,
      align: "left", valign: "middle", margin: 0,
    });
  });

  // Arrow → box
  slide.addText("→", {
    x: 4.1, y: 2.55, w: 0.6, h: 0.6,
    fontFace: FONT_HEADER, fontSize: 32, color: MUTED, bold: true,
    align: "center", valign: "middle", margin: 0,
  });

  // Our box
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 4.9, y: 2.0, w: 4.5, h: 1.7, fill: { color: ACCENT }, line: { color: ACCENT },
  });
  slide.addText("aoi-sentinel\n박스 하나가 다 받아들임", {
    x: 4.9, y: 2.0, w: 4.5, h: 1.7,
    fontFace: FONT_HEADER, fontSize: 18, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });

  // Below
  slide.addText("라인에 두세 종류 검사기 섞여있어도 박스 하나로 통합", {
    x: 0.5, y: 4.4, w: 9.0, h: 0.5,
    fontFace: FONT_BODY, fontSize: 14, bold: true, italic: true, color: NAVY,
    align: "center", margin: 0,
  });

  addAnalogy(slide, "어떤 회사 휴대폰이든 USB-C 케이블 하나로 충전되는 것과 같습니다.");
}

// ---- Slide 10: 데이터 보안 ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "회사의 검사 데이터는 공장 안에서만. 외부 회사 모릅니다.", 10);

  // Inside the factory
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.55, w: 9.0, h: 2.7,
    fill: { color: ICE }, line: { color: NAVY, width: 2 },
  });
  slide.addText("공장 내부", {
    x: 0.6, y: 1.6, w: 8.8, h: 0.4,
    fontFace: FONT_HEADER, fontSize: 13, bold: true, color: NAVY,
    align: "left", margin: 0,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 1.0, y: 2.2, w: 2.2, h: 0.8, fill: { color: NAVY }, line: { color: NAVY },
  });
  slide.addText("검사 데이터", {
    x: 1.0, y: 2.2, w: 2.2, h: 0.8,
    fontFace: FONT_HEADER, fontSize: 13, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });
  slide.addText("→", {
    x: 3.3, y: 2.4, w: 0.4, h: 0.4,
    fontFace: FONT_HEADER, fontSize: 24, color: NAVY, bold: true,
    align: "center", valign: "middle", margin: 0,
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 3.8, y: 2.2, w: 2.2, h: 0.8, fill: { color: ACCENT }, line: { color: ACCENT },
  });
  slide.addText("박스 (AI)", {
    x: 3.8, y: 2.2, w: 2.2, h: 0.8,
    fontFace: FONT_HEADER, fontSize: 13, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });
  slide.addText("→", {
    x: 6.1, y: 2.4, w: 0.4, h: 0.4,
    fontFace: FONT_HEADER, fontSize: 24, color: NAVY, bold: true,
    align: "center", valign: "middle", margin: 0,
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 6.6, y: 2.2, w: 2.4, h: 0.8, fill: { color: OK }, line: { color: OK },
  });
  slide.addText("학습 → 똑똑해짐", {
    x: 6.6, y: 2.2, w: 2.4, h: 0.8,
    fontFace: FONT_HEADER, fontSize: 12, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });

  slide.addText("외부로 전송: 0건", {
    x: 0.6, y: 3.45, w: 8.8, h: 0.5,
    fontFace: FONT_HEADER, fontSize: 16, bold: true, color: NO,
    align: "center", margin: 0,
  });

  // Outside
  slide.addText("→ 자동차 부품·반도체·기술 비밀이 새지 않습니다.", {
    x: 0.5, y: 4.45, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 13, italic: true, color: NAVY, bold: true,
    align: "center", margin: 0,
  });

  addAnalogy(slide, "가족 비밀 일기를 집 밖으로 안 가져가는 것과 같습니다.");
}

// ---- Slide 11: 표준 ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "다른 회사가 AI 모델 따라할 수는 있지만, \"표준\"은 못 만듭니다.", 11);

  const rows = [
    { who: "다른 회사",      what: "\"우리도 AI 만들었어요\"",                       when: "1-2년이면 따라옴", color: MUTED },
    { who: "우리",            what: "\"AI들이 따라야 할 표준 자체를 정의 중\"",        when: "표준은 카피 불가",  color: ACCENT },
    { who: "5년 후",          what: "정부가 의무화 (EU·한국 AI법)",                    when: "표준 통과 못 하면 판매 불가",  color: NAVY },
  ];

  rows.forEach((r, i) => {
    const y = 1.55 + i * 1.05;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 1.7, h: 0.9, fill: { color: r.color }, line: { color: r.color },
    });
    slide.addText(r.who, {
      x: 0.5, y, w: 1.7, h: 0.9,
      fontFace: FONT_HEADER, fontSize: 14, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText(r.what, {
      x: 2.4, y, w: 4.5, h: 0.9,
      fontFace: FONT_BODY, fontSize: 13, color: INK,
      align: "left", valign: "middle", margin: 0,
    });
    slide.addText(r.when, {
      x: 7.1, y, w: 2.4, h: 0.9,
      fontFace: FONT_BODY, fontSize: 11, italic: true, color: r.color, bold: true,
      align: "left", valign: "middle", margin: 0,
    });
  });

  addAnalogy(slide, "Bluetooth 만든 회사가 있어요. 다른 회사들이 따라할 수는 있지만, 'Bluetooth라 부를 권한'은 처음 정의한 곳에 있습니다.");
}

// ---- Slide 12: 더 큰 그림 ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "같은 박스가 SMT 검사뿐 아니라 자동차·로봇·AI 서버 다 적용됩니다.", 12);

  const years = [
    { when: "오늘",    what: "자동차 부품 검사",      color: NAVY },
    { when: "1년 후",  what: "AI 서버 보드 (NVIDIA, HBM)",  color: NAVY },
    { when: "2년 후",  what: "휴머노이드 부품 (Tesla, Figure)", color: ACCENT },
    { when: "3년 후",  what: "EV 배터리 검사",          color: ACCENT },
    { when: "5년 후",  what: "글로벌 표준 layer",       color: OK },
  ];

  years.forEach((y, i) => {
    const yy = 1.55 + i * 0.65;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: yy, w: 1.7, h: 0.5, fill: { color: y.color }, line: { color: y.color },
    });
    slide.addText(y.when, {
      x: 0.5, y: yy, w: 1.7, h: 0.5,
      fontFace: FONT_HEADER, fontSize: 14, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText(y.what, {
      x: 2.4, y: yy, w: 7.0, h: 0.5,
      fontFace: FONT_BODY, fontSize: 14, color: INK,
      align: "left", valign: "middle", margin: 0,
    });
  });

  addAnalogy(slide, "우리가 잘 만든 박스 하나가 모든 제조업의 미래에 들어갑니다.");
}

// ---- Slide 13: 정부 의무화 ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "EU·한국·미국 모두 2027-2030년에 \"AI 검사 의무 인증\" 도입.", 13);

  const t = [
    { y: "2024",  what: "EU AI 법 발효",          done: true },
    { y: "2025",  what: "한국 AI 기본법 통과",    done: true },
    { y: "2026",  what: "ISO 국제 표준 가속",     done: true },
    { y: "2027",  what: "의무화 시작",             done: false, accent: true },
    { y: "2030",  what: "글로벌 AI audit 의무화",  done: false, accent: true },
  ];

  t.forEach((m, i) => {
    const y = 1.55 + i * 0.55;
    const c = m.accent ? ACCENT : (m.done ? NAVY : MUTED);
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 1.2, h: 0.45, fill: { color: c }, line: { color: c },
    });
    slide.addText(m.y, {
      x: 0.5, y, w: 1.2, h: 0.45,
      fontFace: FONT_HEADER, fontSize: 13, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText(m.what, {
      x: 1.85, y, w: 7.6, h: 0.45,
      fontFace: FONT_BODY, fontSize: 13, color: m.accent ? ACCENT : INK, bold: !!m.accent,
      align: "left", valign: "middle", margin: 0,
    });
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.4, w: 9.0, h: 0.55, fill: { color: NAVY }, line: { color: NAVY },
  });
  slide.addText("그 시점에 AI 솔루션은 표준 통과해야 판매 가능. 우리 = 표준 정의자.", {
    x: 0.6, y: 4.4, w: 8.8, h: 0.55,
    fontFace: FONT_BODY, fontSize: 13, italic: true, color: GOLD, bold: true,
    align: "center", valign: "middle", margin: 0,
  });

  addAnalogy(slide, "자동차에 안전벨트 의무화된 것처럼, AI도 5년 후엔 인증된 것만 사용 가능. 우리는 그 인증 기관.");
}

// ---- Slide 14: 실제 결과 ----
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "컴퓨터에서 60번 학습 → 진짜 불량 놓침 0건, 작업자 부담 5배 ↓", 14);

  // Two big stat cards
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.6, w: 4.3, h: 2.8,
    fill: { color: "F8FAFC" }, line: { color: OK, width: 2 },
  });
  slide.addText("0건", {
    x: 0.5, y: 1.7, w: 4.3, h: 1.5,
    fontFace: FONT_HEADER, fontSize: 92, bold: true, color: OK,
    align: "center", valign: "bottom", margin: 0,
  });
  slide.addText("놓친 불량", {
    x: 0.5, y: 3.3, w: 4.3, h: 0.5,
    fontFace: FONT_BODY, fontSize: 16, bold: true, color: INK,
    align: "center", margin: 0,
  });
  slide.addText("60번 학습 동안", {
    x: 0.5, y: 3.8, w: 4.3, h: 0.5,
    fontFace: FONT_BODY, fontSize: 12, color: MUTED,
    align: "center", margin: 0,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.6, w: 4.3, h: 2.8,
    fill: { color: "F8FAFC" }, line: { color: NAVY, width: 2 },
  });
  slide.addText("5배 ↓", {
    x: 5.2, y: 1.7, w: 4.3, h: 1.5,
    fontFace: FONT_HEADER, fontSize: 84, bold: true, color: NAVY,
    align: "center", valign: "bottom", margin: 0,
  });
  slide.addText("작업자 부담", {
    x: 5.2, y: 3.3, w: 4.3, h: 0.5,
    fontFace: FONT_BODY, fontSize: 16, bold: true, color: INK,
    align: "center", margin: 0,
  });
  slide.addText("1280 → 230 / 회", {
    x: 5.2, y: 3.8, w: 4.3, h: 0.5,
    fontFace: FONT_BODY, fontSize: 12, color: MUTED,
    align: "center", margin: 0,
  });

  slide.addText("컴퓨터 실험 → 실제 공장에선 같은 일이 6개월 단위로 일어남.", {
    x: 0.5, y: 4.55, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 13, italic: true, color: ACCENT, bold: true,
    align: "center", margin: 0,
  });

  addAnalogy(slide, "박스가 60번 시험 보면서 단 한 번도 진짜 불량을 놓치지 않았습니다.");
}

// ---- Slide 15: 같이 만들어요 ----
{
  const slide = pres.addSlide();
  slide.background = { color: NAVY };

  slide.addText("작업자의 일을 덜고,", {
    x: 0.5, y: 0.7, w: 9.0, h: 0.6,
    fontFace: FONT_HEADER, fontSize: 28, bold: true, color: "FFFFFF",
    align: "center", margin: 0,
  });
  slide.addText("진짜 불량은 0건.", {
    x: 0.5, y: 1.3, w: 9.0, h: 0.6,
    fontFace: FONT_HEADER, fontSize: 28, bold: true, color: GOLD,
    align: "center", margin: 0,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 3.0, y: 2.1, w: 4.0, h: 0.05, fill: { color: ACCENT }, line: { color: ACCENT },
  });

  // Needs panel
  slide.addText("같이 만들어주실 분 찾습니다:", {
    x: 0.5, y: 2.35, w: 9.0, h: 0.4,
    fontFace: FONT_HEADER, fontSize: 14, bold: true, color: GOLD,
    align: "center", margin: 0,
  });
  slide.addText([
    { text: "✓  6개월 무료 박스를 받아주실 첫 공장 1곳\n", options: { color: ICE, breakLine: true } },
    { text: "✓  AI 표준 만드는 데 참여할 동료\n", options: { color: ICE, breakLine: true } },
    { text: "✓  정부·표준 기관과의 다리", options: { color: ICE } },
  ], {
    x: 2.5, y: 2.85, w: 5.0, h: 1.5,
    fontFace: FONT_BODY, fontSize: 14, paraSpaceAfter: 4, valign: "top", margin: 0,
  });

  // Contact
  slide.addText("DrJinHoChoi  ·  github.com/DrJinHoChoi", {
    x: 0.5, y: 4.7, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 13, color: ICE,
    align: "center", margin: 0,
  });

  slide.addText("\"작업자의 일을 덜고, 사고는 0건. 그게 우리가 만드는 AI입니다.\"", {
    x: 0.5, y: 5.15, w: 9.0, h: 0.3,
    fontFace: FONT_BODY, fontSize: 11, italic: true, color: ACCENT,
    align: "center", margin: 0,
  });
}

// ---- Save ----
const outPath = "C:/Users/jinho/source/repos/DrJinHoChoi/aoi-sentinel/docs/sales/pitch_deck_general_kr.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log(`✓ wrote ${outPath}`);
});
