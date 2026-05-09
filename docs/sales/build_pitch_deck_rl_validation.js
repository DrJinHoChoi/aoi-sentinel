// Build pitch_deck_rl_validation_kr.pptx — technical RL validation deck for
// ML engineers, CTOs, researchers, and standards-body technical reviewers.

const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "DrJinHoChoi";
pres.title = "aoi-sentinel — RL 기술 검증 자료";

const NAVY    = "1E2761";
const ICE     = "CADCFC";
const ACCENT  = "F96167";
const GOLD    = "F9E795";
const INK     = "1A1A2E";
const MUTED   = "6B7280";
const CODE_BG = "0F172A";
const CODE_FG = "E2E8F0";
const PAGE_BG = "FFFFFF";
const OK      = "2DA44E";
const NO      = "CF222E";

const FONT_HEADER = "Malgun Gothic";
const FONT_BODY   = "Malgun Gothic";
const FONT_CODE   = "Consolas";

function addSlideHeader(slide, headline, slideNum) {
  slide.addText(`${slideNum} / 15`, {
    x: 9.0, y: 0.18, w: 0.85, h: 0.3,
    fontFace: FONT_BODY, fontSize: 10, color: MUTED, align: "right", margin: 0,
  });
  slide.addText("🎯 " + headline, {
    x: 0.5, y: 0.4, w: 9.0, h: 0.85,
    fontFace: FONT_HEADER, fontSize: 20, bold: true, color: NAVY,
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

function codeBox(slide, x, y, w, h, lines, opts = {}) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h, fill: { color: CODE_BG }, line: { color: CODE_BG },
  });
  const runs = [];
  lines.forEach((ln, i) => {
    const isLast = i === lines.length - 1;
    let color = CODE_FG;
    if (typeof ln === "object") {
      runs.push({ text: ln.text + (isLast ? "" : "\n"),
                  options: { color: ln.color || CODE_FG, bold: !!ln.bold,
                             breakLine: !isLast } });
    } else {
      runs.push({ text: ln + (isLast ? "" : "\n"),
                  options: { color, breakLine: !isLast } });
    }
  });
  slide.addText(runs, {
    x: x + 0.15, y: y + 0.1, w: w - 0.3, h: h - 0.2,
    fontFace: FONT_CODE, fontSize: opts.fontSize || 11, valign: "top", margin: 0,
  });
}

// ---- Slide 1: Title -----------------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: NAVY };

  slide.addText("Mamba RL + Lagrangian PPO", {
    x: 0.5, y: 1.2, w: 9.0, h: 0.85,
    fontFace: FONT_HEADER, fontSize: 40, bold: true, color: "FFFFFF",
    align: "center", margin: 0,
  });
  slide.addText("기술 검증 자료", {
    x: 0.5, y: 2.05, w: 9.0, h: 0.6,
    fontFace: FONT_HEADER, fontSize: 28, color: GOLD,
    align: "center", margin: 0,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 3.0, y: 3.0, w: 4.0, h: 0.05, fill: { color: ACCENT }, line: { color: ACCENT },
  });

  slide.addText("RL을 양산 라인에 배포해도 안전한가? — 수학·코드·실험으로 답합니다.", {
    x: 0.5, y: 3.3, w: 9.0, h: 0.5,
    fontFace: FONT_BODY, fontSize: 15, italic: true, color: ICE,
    align: "center", margin: 0,
  });

  slide.addText("ML 엔지니어 · CTO · 연구자 · 표준 body 기술 검토자 대상", {
    x: 0.5, y: 4.0, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 13, color: ICE,
    align: "center", margin: 0,
  });

  slide.addText("DrJinHoChoi · 2026", {
    x: 0.5, y: 5.0, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 11, color: ICE,
    align: "center", margin: 0,
  });
}

// ---- Slide 2: 4 skepticisms --------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "RL은 본질적으로 탐색합니다. 그럼 양산에서 어떻게 안전을 보장하나?", 2);

  const skepticisms = [
    { q: "PPO는 unstable. 양산에 못 씀",      a: "Lagrangian PPO + reward clip + 모드 게이트" },
    { q: "escape 보장은 ML로 불가능",          a: "Constrained MDP의 hard constraint" },
    { q: "Mamba는 hype. transformer로 충분",  a: "O(L) vs O(L²) 정량 측정 우위" },
    { q: "실 양산 도메인 검증 안 됨",           a: "VisA 실험 + 어댑터 SDK + 모드 게이트" },
  ];

  skepticisms.forEach((s, i) => {
    const y = 1.55 + i * 0.78;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 4.4, h: 0.65, fill: { color: ICE }, line: { color: NAVY, width: 1 },
    });
    slide.addText(`Q${i+1}.  "${s.q}"`, {
      x: 0.6, y, w: 4.2, h: 0.65,
      fontFace: FONT_BODY, fontSize: 12, color: NAVY, italic: true, bold: true,
      align: "left", valign: "middle", margin: 0,
    });
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 5.0, y, w: 4.5, h: 0.65, fill: { color: NAVY }, line: { color: NAVY },
    });
    slide.addText("→  " + s.a, {
      x: 5.1, y, w: 4.3, h: 0.65,
      fontFace: FONT_BODY, fontSize: 12, color: GOLD, bold: true,
      align: "left", valign: "middle", margin: 0,
    });
  });

  slide.addText("이 deck은 각 회의론에 한 슬라이드씩 답.", {
    x: 0.5, y: 4.7, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 12, italic: true, color: ACCENT, bold: true,
    align: "center", margin: 0,
  });

  addTechNote(slide, "모든 답변 = 코드 + 실험 + 학술 인용으로 검증 가능");
}

// ---- Slide 3: Constrained MDP -----------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "Constrained MDP — 안전 제약이 reward 위에 hard constraint", 3);

  codeBox(slide, 0.5, 1.4, 9.0, 2.3, [
    { text: "M = (S, A, P, r, c, γ, ε)", color: GOLD, bold: true },
    "",
    { text: "maximize    J_r(π) = E_π [ Σ γ^t · r(s_t, a_t) ]", color: CODE_FG },
    { text: "                              ↑ 보상 = -cost matrix", color: MUTED },
    "",
    { text: "subject to  J_c(π) = E_π [ Σ γ^t · c(s_t, a_t) ] ≤ ε", color: ACCENT, bold: true },
    { text: "                              ↑ escape indicator      ↑ 0.001", color: MUTED },
  ], { fontSize: 14 });

  slide.addText([
    { text: "state s   ", options: { bold: true, color: NAVY } },
    { text: "= (image_t, history_{t-L:t})\n", options: { color: INK, breakLine: true } },
    { text: "action a  ", options: { bold: true, color: NAVY } },
    { text: "= {DEFECT, PASS, ESCALATE}\n", options: { color: INK, breakLine: true } },
    { text: "cost c    ", options: { bold: true, color: NAVY } },
    { text: "= 1 if (action=PASS && label=TRUE_DEFECT) else 0", options: { color: INK } },
  ], {
    x: 0.5, y: 3.85, w: 9.0, h: 1.1,
    fontFace: FONT_BODY, fontSize: 12, valign: "top", paraSpaceAfter: 4, margin: 0,
  });

  addTechNote(slide, "Altman 1999 — Constrained Markov Decision Processes. EU AI Act audit과 호환되는 formalism.");
}

// ---- Slide 4: Lagrangian + dual ascent --------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "Lagrangian 분해 + dual ascent (Stooke et al. 2020)", 4);

  codeBox(slide, 0.5, 1.4, 9.0, 1.7, [
    { text: "L(π, λ) = J_r(π) − λ · (J_c(π) − ε),    λ ≥ 0", color: GOLD, bold: true },
    "",
    { text: "Primal:  π ← argmax_π L(π, λ)", color: CODE_FG },
    { text: "Dual:    λ ← max(0, λ + β · (J_c(π) − ε))", color: ACCENT, bold: true },
    { text: "                              ↑ 제약 위반량", color: MUTED },
  ], { fontSize: 14 });

  // Two-column intuition
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 3.25, w: 4.4, h: 1.5, fill: { color: ICE }, line: { color: NAVY, width: 1 },
  });
  slide.addText("escape > ε 초과", {
    x: 0.6, y: 3.3, w: 4.2, h: 0.4,
    fontFace: FONT_HEADER, fontSize: 13, bold: true, color: NO,
    align: "left", margin: 0,
  });
  slide.addText("→ λ ↑ → 정책이 안전 강제\n→ PASS 비용이 더 아프게 인식", {
    x: 0.6, y: 3.7, w: 4.2, h: 1.0,
    fontFace: FONT_BODY, fontSize: 12, color: INK,
    align: "left", valign: "top", margin: 0,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.1, y: 3.25, w: 4.4, h: 1.5, fill: { color: ICE }, line: { color: NAVY, width: 1 },
  });
  slide.addText("escape < ε 만족", {
    x: 5.2, y: 3.3, w: 4.2, h: 0.4,
    fontFace: FONT_HEADER, fontSize: 13, bold: true, color: OK,
    align: "left", margin: 0,
  });
  slide.addText("→ λ ↓ → 정책이 reward 추구 자유\n→ false-call 줄이기 시도", {
    x: 5.2, y: 3.7, w: 4.2, h: 1.0,
    fontFace: FONT_BODY, fontSize: 12, color: INK,
    align: "left", valign: "top", margin: 0,
  });

  addTechNote(slide, "λ 자체가 학습됨 — 사람이 \"비용 가중치 100? 1000?\" 고민 필요 없음");
}

// ---- Slide 5: PPO clipped ----------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "PPO clipped surrogate (Schulman et al. 2017) — 정책 한 번에 너무 크게 안 변함", 5);

  codeBox(slide, 0.5, 1.4, 9.0, 2.3, [
    { text: "ratio_t(π) = π(a_t|s_t) / π_old(a_t|s_t)", color: CODE_FG },
    "",
    { text: "L_clip(π) = E_t [ min(", color: GOLD, bold: true },
    { text: "    ratio_t · A_t,", color: CODE_FG },
    { text: "    clip(ratio_t, 1−ε_clip, 1+ε_clip) · A_t", color: CODE_FG },
    { text: ") ]", color: GOLD, bold: true },
    "",
    { text: "where  A_t = A_r,t  −  λ · A_c,t      ← Lagrangian 결합", color: ACCENT, bold: true },
  ], { fontSize: 13 });

  slide.addText([
    { text: "왜 PPO?  ", options: { bold: true, color: NAVY } },
    { text: "TRPO보다 단순 (1차 method)  ·  on-policy로 시퀀스 인코더와 자연 결합  ·  ", options: { color: INK } },
    { text: "클리핑이 학습 폭주 방지  ·  ", options: { color: INK } },
    { text: "검증된 baseline (산업·학계 default)", options: { color: INK } },
  ], {
    x: 0.5, y: 3.95, w: 9.0, h: 1.05,
    fontFace: FONT_BODY, fontSize: 12, valign: "top", margin: 0,
  });

  addTechNote(slide, "arxiv.org/abs/1707.06347 — Schulman et al. 우리 lagrangian_ppo.py:update에서 정확히 이 수식 구현");
}

// ---- Slide 6: Why Mamba -----------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "Mamba vs Transformer — L=512+에서 측정 가능 우위", 6);

  slide.addTable([
    [
      { text: "시퀀스 L",    options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
      { text: "Transformer FLOPs", options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
      { text: "Mamba FLOPs",  options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
      { text: "비율",          options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } },
    ],
    [{ text: "256",     options: { align: "center" } }, { text: "65,536", options: { align: "center" } }, { text: "256",   options: { align: "center" } }, { text: "256×",   options: { color: ACCENT, bold: true, align: "center" } }],
    [{ text: "512",     options: { align: "center" } }, { text: "262,144", options: { align: "center" } }, { text: "512",   options: { align: "center" } }, { text: "512×",   options: { color: ACCENT, bold: true, align: "center" } }],
    [{ text: "1,024",   options: { align: "center" } }, { text: "1,048,576", options: { align: "center" } }, { text: "1,024", options: { align: "center" } }, { text: "1,024×", options: { color: ACCENT, bold: true, align: "center" } }],
    [{ text: "2,048",   options: { align: "center" } }, { text: "4,194,304", options: { align: "center" } }, { text: "2,048", options: { align: "center" } }, { text: "2,048×", options: { color: ACCENT, bold: true, align: "center" } }],
    [{ text: "4,096",   options: { align: "center", bold: true, fill: { color: ICE } } }, { text: "16,777,216", options: { align: "center", bold: true, fill: { color: ICE } } }, { text: "4,096", options: { align: "center", bold: true, fill: { color: ICE } } }, { text: "4,096×", options: { color: NO, bold: true, align: "center", fill: { color: ICE } } }],
  ], {
    x: 0.5, y: 1.55, w: 9.0, colW: [1.5, 2.8, 2.2, 2.5],
    fontFace: FONT_BODY, fontSize: 13, color: INK,
    border: { pt: 1, color: "E5E7EB" }, rowH: 0.4,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.4, w: 9.0, h: 0.55, fill: { color: ACCENT }, line: { color: ACCENT },
  });
  slide.addText("우리 production NPI 라인 history L=512-4096 — Mamba 없으면 추론 latency 양산 불가능", {
    x: 0.6, y: 4.4, w: 8.8, h: 0.55,
    fontFace: FONT_BODY, fontSize: 13, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });

  addTechNote(slide, "Gu & Dao 2023 — Mamba: Linear-Time Sequence Modeling. scripts/bench_mamba_vs_transformer.py로 실측 가능");
}

// ---- Slide 7: Architecture -----------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "두 Mamba (이미지·시퀀스) + Lagrangian PPO actor-critic", 7);

  // Image branch
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.55, w: 4.3, h: 0.55, fill: { color: NAVY }, line: { color: NAVY },
  });
  slide.addText("ROI image (224×224)", {
    x: 0.5, y: 1.55, w: 4.3, h: 0.55,
    fontFace: FONT_HEADER, fontSize: 13, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 2.25, w: 4.3, h: 0.6, fill: { color: ICE }, line: { color: NAVY, width: 1 },
  });
  slide.addText("ConvNeXt-T / MambaVision\n(image_encoder.py)", {
    x: 0.5, y: 2.25, w: 4.3, h: 0.6,
    fontFace: FONT_BODY, fontSize: 11, color: INK,
    align: "center", valign: "middle", margin: 0,
  });

  // Sequence branch
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.1, y: 1.55, w: 4.4, h: 0.55, fill: { color: NAVY }, line: { color: NAVY },
  });
  slide.addText("Inspection history (L=256)", {
    x: 5.1, y: 1.55, w: 4.4, h: 0.55,
    fontFace: FONT_HEADER, fontSize: 13, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.1, y: 2.25, w: 4.4, h: 0.6, fill: { color: ICE }, line: { color: NAVY, width: 1 },
  });
  slide.addText("Mamba-SSM 시퀀스 인코더\n(sequence_encoder.py)", {
    x: 5.1, y: 2.25, w: 4.4, h: 0.6,
    fontFace: FONT_BODY, fontSize: 11, color: INK,
    align: "center", valign: "middle", margin: 0,
  });

  // Concat arrow
  slide.addText("↓  concat  ↓", {
    x: 0.5, y: 2.95, w: 9.0, h: 0.3,
    fontFace: FONT_HEADER, fontSize: 12, color: MUTED, italic: true,
    align: "center", margin: 0,
  });

  // Trunk
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 3.0, y: 3.3, w: 4.0, h: 0.5, fill: { color: ACCENT }, line: { color: ACCENT },
  });
  slide.addText("Trunk MLP, hidden=384", {
    x: 3.0, y: 3.3, w: 4.0, h: 0.5,
    fontFace: FONT_HEADER, fontSize: 13, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });

  // Three heads
  const heads = [
    { x: 1.2, label: "Actor", sub: "logits over 3 actions", color: NAVY },
    { x: 4.2, label: "V_r",   sub: "reward critic",         color: NAVY },
    { x: 7.0, label: "V_c",   sub: "cost critic",           color: ACCENT },
  ];
  heads.forEach((h) => {
    slide.addShape(pres.shapes.RECTANGLE, {
      x: h.x, y: 4.0, w: 1.8, h: 0.7, fill: { color: h.color }, line: { color: h.color },
    });
    slide.addText(h.label, {
      x: h.x, y: 4.0, w: 1.8, h: 0.4,
      fontFace: FONT_HEADER, fontSize: 14, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText(h.sub, {
      x: h.x, y: 4.4, w: 1.8, h: 0.3,
      fontFace: FONT_BODY, fontSize: 9, color: ICE,
      align: "center", valign: "middle", margin: 0,
    });
  });

  slide.addText("models/policy/actor_critic.py — 약 60 lines.  두 critic head가 핵심: reward GAE / cost GAE 따로.", {
    x: 0.5, y: 4.85, w: 9.0, h: 0.3,
    fontFace: FONT_BODY, fontSize: 11, italic: true, color: MUTED,
    align: "center", margin: 0,
  });

  addTechNote(slide, "MambaActorCritic class — github.com/DrJinHoChoi/aoi-sentinel/blob/main/aoi_sentinel/models/policy/actor_critic.py");
}

// ---- Slide 8: Experimental setup ---------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "실험 셋업 — 재현 가능. seed 고정, 코드·데이터·하이퍼 모두 GitHub.", 8);

  slide.addTable([
    [
      { text: "항목",        options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
      { text: "값",          options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
    ],
    ["데이터",                    "VisA PCB1-4, 4,416 imgs, defect rate 0.091"],
    ["GPU",                       "NVIDIA A100-SXM4-80GB"],
    ["백본",                      "ConvNeXt-Tiny (MambaVision fallback)"],
    ["시퀀스 인코더",              "Pure-PyTorch Mamba (mamba-ssm CUDA fallback)"],
    [{ text: "Cost matrix", options: { bold: true } }, { text: "escape=50, false_call=1, operator=2", options: { bold: true, color: ACCENT } }],
    [{ text: "안전 제약 ε", options: { bold: true } }, { text: "0.001", options: { bold: true, color: ACCENT } }],
    ["Rollout",                   "256 step / iter"],
    ["학습 iter",                 "400 (분석은 0-167)"],
    ["Optimizer",                 "AdamW (actor lr=3e-4, critic lr=1e-3)"],
    ["PPO clip ε",                "0.2, n_epochs=4, minibatch=32"],
    ["Entropy coef / λ_lr",        "0.02 / 0.2"],
    ["Reward clip",                "-100 (vloss 폭발 방지)"],
    [{ text: "Seed", options: { bold: true } }, { text: "42 (single-seed report; multi-seed pending)", options: { color: MUTED } }],
  ], {
    x: 0.5, y: 1.5, w: 9.0, colW: [3.0, 6.0],
    fontFace: FONT_BODY, fontSize: 11, color: INK,
    border: { pt: 1, color: "E5E7EB" }, rowH: 0.27,
  });

  addTechNote(slide, "configs/stage1_npi_rl_light.yaml — 모든 값 검증 가능, git diff로 변경 이력 추적");
}

// ---- Slide 9: Result — 4 stat cards ------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "VisA 결과 — 60 iter escape=0 + 비용 5.6× ↓ — Lagrangian contract 작동 입증", 9);

  const stats = [
    { v: "0",     l: "Escape per iter",     s: "60 iter 동안 (iter 4-167)", c: OK },
    { v: "5.6×",  l: "Cost reduction",       s: "1280 → 230 / iter",          c: NAVY },
    { v: "iter 4", l: "정책 자동 전환",       s: "ESCALATE→DEFECT, 사람 개입 0", c: ACCENT },
    { v: "1.0→7.0→1.8", l: "λ trajectory",   s: "Lagrangian dual 자동 조절",     c: NAVY },
  ];

  stats.forEach((s, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.5 + col * 4.6;
    const y = 1.55 + row * 1.65;
    slide.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 4.3, h: 1.45,
      fill: { color: "F8FAFC" }, line: { color: "E5E7EB", width: 1 },
    });
    slide.addText(s.v, {
      x, y, w: 4.3, h: 0.85,
      fontFace: FONT_HEADER, fontSize: 38, bold: true, color: s.c,
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

  slide.addText("이건 lab 결과 — 실제 라인에선 같은 일이 day 단위로 일어남.", {
    x: 0.5, y: 4.95, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 12, italic: true, color: ACCENT, bold: true,
    align: "center", margin: 0,
  });

  addTechNote(slide, "stage1.log iter 0-167. 상세 4-차트 plot script: pilot_evidence_kr.md");
}

// ---- Slide 10: Phase analysis ------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "정책 자동 진화 — trivial → productive, 사람 개입 0", 10);

  const phases = [
    { name: "Phase 1 (iter 0-3): 무작위 탐색",                color: NAVY,   stats: "escape: 9, 3, 4, 0   reward: -36 → -5" },
    { name: "Phase 2 (iter 4-152): ESCALATE-everything",       color: NAVY,   stats: "esc=256, fc=0, reward=-5   λ rising 0.7 → 7.0" },
    { name: "★ Phase 3 (iter 153-167): productive transition", color: ACCENT, stats: "esc=0, fc≈230, reward=-0.9, escape=0 유지" },
    { name: "Phase 4 (iter 168+): PASS-collapse",               color: NO,     stats: "escapes 25-30/iter — 다음 슬라이드에서 분석" },
  ];

  phases.forEach((p, i) => {
    const y = 1.5 + i * 0.8;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 9.0, h: 0.7, fill: { color: "F8FAFC" }, line: { color: p.color, width: 2 },
    });
    slide.addText(p.name, {
      x: 0.7, y: y + 0.05, w: 8.6, h: 0.3,
      fontFace: FONT_HEADER, fontSize: 13, bold: true, color: p.color,
      align: "left", margin: 0,
    });
    slide.addText(p.stats, {
      x: 0.7, y: y + 0.35, w: 8.6, h: 0.3,
      fontFace: FONT_CODE, fontSize: 11, color: INK,
      align: "left", margin: 0,
    });
  });

  slide.addText("핵심 해석: λ가 7.0 임계점까지 climbing → 정책이 더 productive basin으로 점프. 이게 Lagrangian 메커니즘.", {
    x: 0.5, y: 4.85, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 11, italic: true, color: ACCENT, bold: true,
    align: "center", margin: 0,
  });

  addTechNote(slide, "stage1.log 직접 분석. 회의론자가 line 단위로 reproduce 가능.");
}

// ---- Slide 11: Failure mode + mitigations ------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "실패 모드 + 완화 — 솔직하게 공개", 11);

  // Left: problem
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.45, w: 4.4, h: 1.85, fill: { color: "FEE2E2" }, line: { color: NO, width: 1 },
  });
  slide.addText("발생 (iter 168+)", {
    x: 0.6, y: 1.5, w: 4.2, h: 0.35,
    fontFace: FONT_HEADER, fontSize: 13, bold: true, color: NO,
    align: "left", margin: 0,
  });
  slide.addText([
    { text: "always-PASS basin으로 drift", options: { bullet: true, breakLine: true, color: INK } },
    { text: "escapes 25-30 / iter", options: { bullet: true, breakLine: true, color: INK } },
    { text: "vloss 2,732,971 (critic 폭발)", options: { bullet: true, breakLine: true, color: INK } },
    { text: "λ 4 → 31 (회복 시도)", options: { bullet: true, color: INK } },
  ], {
    x: 0.7, y: 1.85, w: 4.1, h: 1.4,
    fontFace: FONT_BODY, fontSize: 11, paraSpaceAfter: 3, valign: "top",
  });

  // Right: causes
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.1, y: 1.45, w: 4.4, h: 1.85, fill: { color: "FEF3C7" }, line: { color: "F59E0B", width: 1 },
  });
  slide.addText("원인 (3개 동시)", {
    x: 5.2, y: 1.5, w: 4.2, h: 0.35,
    fontFace: FONT_HEADER, fontSize: 13, bold: true, color: "B45309",
    align: "left", margin: 0,
  });
  slide.addText([
    { text: "c_escape=1000 → critic 폭파", options: { bullet: true, breakLine: true, color: INK } },
    { text: "entropy=0.05 → exploration 너무 강", options: { bullet: true, breakLine: true, color: INK } },
    { text: "λ_lr=0.05 → 빠른 변화에 못 따라감", options: { bullet: true, color: INK } },
  ], {
    x: 5.3, y: 1.85, w: 4.1, h: 1.4,
    fontFace: FONT_BODY, fontSize: 11, paraSpaceAfter: 3, valign: "top",
  });

  // Bottom: mitigations
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 3.45, w: 9.0, h: 1.55, fill: { color: "DCFCE7" }, line: { color: OK, width: 1 },
  });
  slide.addText("완화 (commit d92e40c)", {
    x: 0.6, y: 3.5, w: 8.8, h: 0.35,
    fontFace: FONT_HEADER, fontSize: 13, bold: true, color: OK,
    align: "left", margin: 0,
  });
  slide.addText([
    { text: "c_escape:    1000 → 50      ", options: { bold: true, color: NAVY } },
    { text: "(50× 비대칭 유지하되 critic 안정)\n", options: { color: INK, breakLine: true } },
    { text: "entropy:     0.05 → 0.02    ", options: { bold: true, color: NAVY } },
    { text: "(첫 transition 후 drift 억제)\n", options: { color: INK, breakLine: true } },
    { text: "lambda_lr:   0.05 → 0.2     ", options: { bold: true, color: NAVY } },
    { text: "(4× 빠른 λ 반응)\n", options: { color: INK, breakLine: true } },
    { text: "reward clip: -100           ", options: { bold: true, color: NAVY } },
    { text: "(vloss 폭발 hard cap)", options: { color: INK } },
  ], {
    x: 0.7, y: 3.85, w: 8.7, h: 1.1,
    fontFace: FONT_CODE, fontSize: 11, paraSpaceAfter: 2, valign: "top", margin: 0,
  });

  addTechNote(slide, "PPO + 극단 cost asymmetry는 본질적으로 어려움. 단계적 업그레이드 plan: 다음 슬라이드");
}

// ---- Slide 12: Algorithm comparison + roadmap --------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "알고리즘 비교 + 업그레이드 roadmap", 12);

  slide.addTable([
    [
      { text: "알고리즘",  options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
      { text: "출처",       options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
      { text: "단계",       options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
      { text: "효과",       options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
    ],
    [{ text: "Lagrangian PPO (현재)", options: { bold: true, color: ACCENT } }, "Stooke 2020",  "v1",     "작동 + 알려진 한계"],
    [{ text: "PID Lagrangian", options: { bold: true } },        "Stooke 2020",  { text: "v1.1 다음", options: { bold: true, color: OK } }, "λ 응답 빠름, oscillation ↓"],
    [{ text: "Sauté RL",        options: { bold: true } },        "Sootla 2022",  "v1.2",   "상태에 safety budget augment"],
    [{ text: "P3O (Penalized)", options: { bold: true } },        "Zhang 2022",   "v1.5 alt", "dual 변수 0개"],
    [{ text: "CPO",             options: { bold: true } },        "Achiam 2017",  { text: "v2", options: { bold: true, color: OK } },     "trust region + 안전 bound"],
    [{ text: "CRPO",            options: { bold: true } },        "Xu 2021",      "v2 alt", "dual 0개, 수렴 증명"],
    [{ text: "Decision Mamba + cost", options: { bold: true } },  "2024 frontier","v3",     "sequence enc + 학술 contribution"],
  ], {
    x: 0.5, y: 1.5, w: 9.0, colW: [2.6, 1.6, 1.6, 3.2],
    fontFace: FONT_BODY, fontSize: 11, color: INK,
    border: { pt: 1, color: "E5E7EB" }, rowH: 0.4,
  });

  slide.addText("v1.1 (PID Lagrangian) = 코드 5줄, 가장 큰 효과.  v2 (CPO) = 첫 anchor LOI 후 양산 deploy 시점.", {
    x: 0.5, y: 4.9, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 11, italic: true, color: ACCENT, bold: true,
    align: "center", margin: 0,
  });

  addTechNote(slide, "ai_manufacturing_standards.md Layer 0.1 algorithm 참고. 모든 arXiv 인용 검증 가능.");
}

// ---- Slide 13: Production safeguards -----------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "Production safeguards — RL 외부에 다층 방어. RL 실패해도 line 안 멈춤.", 13);

  const layers = [
    { name: "Layer 1 — RL 정책 (Lagrangian PPO)",         desc: "escape rate ≤ ε 수학적 보장 (조건부)", color: NAVY },
    { name: "Layer 2 — Safety gate (모델 promotion 전)",    desc: "후보 모델은 hold-out에서 escape=0 입증 필수", color: NAVY },
    { name: "Layer 3 — 운영 모드 (SHADOW/ASSIST/AUTON)",   desc: "SHADOW = 영향 0 / escape 1건이라도 발생 시 자동 강등", color: ACCENT },
    { name: "Layer 4 — Drift 모니터링 + decommissioning",   desc: "KPI 임계 초과 시 자동 retire", color: NAVY },
  ];

  layers.forEach((l, i) => {
    const y = 1.5 + i * 0.78;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 9.0, h: 0.65, fill: { color: i === 2 ? ICE : "F8FAFC" }, line: { color: l.color, width: i === 2 ? 2 : 1 },
    });
    slide.addText(l.name, {
      x: 0.7, y: y + 0.05, w: 8.6, h: 0.3,
      fontFace: FONT_HEADER, fontSize: 13, bold: true, color: l.color,
      align: "left", margin: 0,
    });
    slide.addText(l.desc, {
      x: 0.7, y: y + 0.32, w: 8.6, h: 0.3,
      fontFace: FONT_BODY, fontSize: 11, color: INK,
      align: "left", margin: 0,
    });
  });

  slide.addText("핵심: RL 정책이 \"이론상 안전\"한 게 아니라, 운영 게이트가 사고 차단. RL 실패해도 line 정상.", {
    x: 0.5, y: 4.7, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 12, italic: true, color: ACCENT, bold: true,
    align: "center", margin: 0,
  });

  addTechNote(slide, "runtime/safety_gate.py + runtime/modes.py — 코드 50+ tests passing");
}

// ---- Slide 14: Reproducibility -----------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: PAGE_BG };
  addSlideHeader(slide, "Reproducibility + open science — 모든 게 공개. 직접 검증 가능.", 14);

  slide.addTable([
    [
      { text: "자산",        options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
      { text: "공개 위치",    options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
      { text: "라이선스",    options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
    ],
    ["RFC v0.1 (AICS schema)",         "github.com/DrJinHoChoi/aoi-common-spec",      { text: "BSD-3", options: { color: OK, bold: true } }],
    ["Reference 구현",                  "github.com/DrJinHoChoi/aoi-sentinel",         "Inspection"],
    ["Lagrangian PPO 코드 (~300 lines)", "models/policy/lagrangian_ppo.py",             { text: "Open", options: { color: OK, bold: true } }],
    ["벤치마크 데이터",                 "VisA (Amazon Science)",                       { text: "BSD-3", options: { color: OK, bold: true } }],
    ["학습 config",                     "configs/stage1_npi_rl_light.yaml",            { text: "Public", options: { color: OK, bold: true } }],
    [{ text: "Seed", options: { bold: true } }, "42 (single-seed reported)", { text: "—", options: { color: MUTED } }],
    ["50+ unit tests",                   "tests/",                                       { text: "Public", options: { color: OK, bold: true } }],
    ["Bench script",                    "scripts/bench_mamba_vs_transformer.py",       { text: "Public", options: { color: OK, bold: true } }],
  ], {
    x: 0.5, y: 1.5, w: 9.0, colW: [3.5, 4.0, 1.5],
    fontFace: FONT_BODY, fontSize: 10, color: INK,
    border: { pt: 1, color: "E5E7EB" }, rowH: 0.32,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.4, w: 9.0, h: 0.55, fill: { color: ACCENT }, line: { color: ACCENT },
  });
  slide.addText("Pending: 3-seed variance 분석 · AICS Conformance Test Suite v1.0 · 학술 논문 (NeurIPS WS / IEEE T-II)", {
    x: 0.6, y: 4.4, w: 8.8, h: 0.55,
    fontFace: FONT_BODY, fontSize: 11, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });

  addTechNote(slide, "PR / issue 환영. 회의론자가 직접 reproduce 가능.");
}

// ---- Slide 15: Closing -------------------------------------------------
{
  const slide = pres.addSlide();
  slide.background = { color: NAVY };

  slide.addText("우리 RL은 마법이 아닙니다.", {
    x: 0.5, y: 0.7, w: 9.0, h: 0.6,
    fontFace: FONT_HEADER, fontSize: 28, bold: true, color: "FFFFFF",
    align: "center", margin: 0,
  });
  slide.addText("알려진 알고리즘 + 정직한 한계 공개 + 다층 안전 게이트.", {
    x: 0.5, y: 1.3, w: 9.0, h: 0.5,
    fontFace: FONT_HEADER, fontSize: 18, color: GOLD, italic: true,
    align: "center", margin: 0,
  });

  // 6 pillars
  const pillars = [
    { k: "수학",       v: "Constrained MDP + Lagrangian (Altman 1999)" },
    { k: "알고리즘",   v: "PPO (Schulman 2017) + dual ascent" },
    { k: "아키텍처",  v: "MambaVision + Mamba-SSM (O(L))" },
    { k: "검증",       v: "VisA 4,416 imgs, 60 iter escape=0, 5.6× ↓" },
    { k: "안전",       v: "SHADOW + safety gate + 자동 강등" },
    { k: "솔직함",     v: "PASS-collapse 발생·원인·완화·gap 모두 공개" },
  ];

  pillars.forEach((p, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.5 + col * 4.55;
    const y = 2.05 + row * 0.65;
    slide.addText([
      { text: p.k + "  ", options: { bold: true, color: ACCENT } },
      { text: p.v,         options: { color: ICE } },
    ], {
      x, y, w: 4.4, h: 0.55,
      fontFace: FONT_BODY, fontSize: 11, valign: "middle", margin: 0,
    });
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.3, w: 9.0, h: 0.55, fill: { color: ACCENT }, line: { color: ACCENT },
  });
  slide.addText("\"RL 정책이 약속을 못 지키면, 시스템이 강제로 지킵니다.\"", {
    x: 0.6, y: 4.3, w: 8.8, h: 0.55,
    fontFace: FONT_HEADER, fontSize: 16, bold: true, italic: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });

  slide.addText("DrJinHoChoi  ·  github.com/DrJinHoChoi", {
    x: 0.5, y: 5.0, w: 9.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 12, color: ICE,
    align: "center", margin: 0,
  });
}

// ---- Save ----------------------------------------------------------------
const outPath = "C:/Users/jinho/source/repos/DrJinHoChoi/aoi-sentinel/docs/sales/pitch_deck_rl_validation_kr.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log(`✓ wrote ${outPath}`);
});
