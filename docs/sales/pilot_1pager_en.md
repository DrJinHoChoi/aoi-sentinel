# aoi-sentinel — 6-Month Free Pilot

> **We will measurably reduce your SMT line's 30% AOI false-call rate within 6 months. Zero deployment cost. If we miss the target, you owe nothing.**

---

## What it is

A vendor-neutral edge box (Saki / Koh Young / Mycronic / TRI / Mirtec / Omron) that connects via USB to your secondary inspection station, learns from your operators' decisions, and reduces false calls automatically — on-prem, with a hard escape-rate guarantee.

## The problem

Roughly **30% of AOI defect calls in automotive electronics SMT are false positives** (NG flagged but actually OK). The cost of bridging that gap — operator labor, line throughput, fatigue — is paid every shift, every day.

Existing AI options (Mycronic DeepReview, Koh Young KSMART AI, Siemens AOI-FCR) are **vendor-locked**. Mixed-vendor lines mean N integrations, and your data lives inside their cloud.

## How we are different

| | Vendor solutions | aoi-sentinel |
|---|---|---|
| Vendor coverage | One vendor at a time | **Any vendor** |
| Data location | Vendor cloud | **On-prem only — never leaves the facility** |
| How it improves | Manual rule tuning | **Learns automatically from operator decisions** |
| Escape guarantee | None | **Hard constraint — auto-demotes on any miss** |
| Expansion path | Single inspector | **Secondary station → line → site orchestration** |

## The 6-month pilot

| Item | Terms |
|------|-------|
| **Cost** | **$0** — hardware, software, and setup all included |
| **Duration** | 6 months, 1 line |
| **We provide** | One edge box (Jetson Orin Nano), operator UI, setup, training, 24/7 remote support |
| **You provide** | Read-only access to AOI result share, ~30 min/day of operator review time, 6-month operational data usage rights, case-study consent |
| **Operating mode** | SHADOW (assistive only — no automated decisions) |

**Zero risk to operations**: the box does not interrupt your line. Existing workflow continues unchanged; we sit alongside the operator as an aid.

## Success criteria (measured at 6 months)

| Metric | Target | Measurement |
|--------|--------|-------------|
| False-call reduction | **≥ 70%** | First 30 days vs. last 30 days, like-for-like |
| Escapes (true defects through) | **0** | Operator re-inspection ground truth |
| Operator workload reduction | **≥ 200 hr/month/line** | Direct timing of secondary inspection |

**If we miss any target, the contract auto-terminates at zero cost.** Box is collected; data deleted on request.

## After 6 months — full contract pricing

If targets are met, choose one:

| Option | Terms |
|--------|-------|
| **A. Perpetual license** | $80–160k per site (unlimited lines) + 15% annual support |
| **B. Annual subscription** | $25–40k per line/year (HW lease + SW + support included) |
| **C. Additional pilot** | Free pilot on a different line or different vendor — same terms |

**ROI reference** (single line):
- 70% false-call reduction ≈ ~200 operator hours/month saved
- $20–50 per false call × ~200/day reduced = **$1.5–3.5M/year/line saved**
- Full contract at $80–160k = **2–4 month payback**

## Why now

- Free pilot offer limited to first 3 customer sites (subsequent pilots: $25–40k introductory)
- We are publishing the open standard [aoi-common-spec](https://github.com/DrJinHoChoi/aoi-common-spec) — early adopters shape the schema
- Automotive industry margin pressure is making operating-cost reduction a board-level priority

## Technical foundation

- **Mamba-based vision + reinforcement learning policy** — Constrained MDP guaranteeing the escape constraint
- **Open standard [aoi-common-spec](https://github.com/DrJinHoChoi/aoi-common-spec)** — we publish RFC v0.1, leading vendor-neutral data standardisation
- **On-prem-only architecture** — zero customer-data egress (auto-OEM QA-friendly)
- Open-source reference implementation: [aoi-sentinel](https://github.com/DrJinHoChoi/aoi-sentinel)

## Next steps

1. **30-minute video meeting** — line walk-through, pilot fit assessment
2. **2-week setup** — edge box delivery, AOI result share mount
3. **30-day baseline measurement** — quantify current false-call rate
4. **6 months SHADOW operation** — labels accumulate, model auto-improves
5. **Day 180 review** — go / no-go on full contract

---

**Contact**

| | |
|---|---|
| Lead | Jin Ho Choi (DrJinHoChoi) |
| Email | _(coming soon)_ |
| GitHub | https://github.com/DrJinHoChoi |
| Reply | Use the channel this page reached you on |

> **If false calls aren't down 70% in 6 months, you pay nothing. There is no risk in starting.**
