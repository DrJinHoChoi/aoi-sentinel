# Jetson Nano POC Demo Box

> **Purpose**: a portable, briefcase-sized demo of the "USB plug-and-play" concept to bring into anchor-customer meetings.
> **Not for production.** The 2019 Jetson Nano (4GB, Maxwell) is too constrained for the Mamba RL stack — it runs a lightweight MobileNetV3-Small encoder and the operator UI only. Production lines use Jetson Orin Nano 8GB (see `../jetson_orin_nano/`).

## What this demonstrates in 5 minutes

1. **Plug power, wait 30s** → box boots, operator UI auto-launches on attached HDMI display, IP and QR code shown.
2. **Smartphone scans QR** → same UI opens on the phone (responsive, mobile-first).
3. **Demonstrator drops a sample CSV bundle into a watched folder** (or runs `aoi-demo replay`).
4. **UI shows ROI cards live** — pulling images out of the bundle through our generic_csv adapter.
5. **Demonstrator clicks PASS / DEFECT / 모름** on each — KPI tiles update in real time, label queue grows.
6. **After 100 clicks**, dashboard shows the simulated false-call rate dropping → demonstrates the self-improvement loop visually.

The whole demo fits in a foam-lined briefcase. Total weight ~2.5 kg. Battery-optional.

## Hardware bill of materials

| Item | Spec | Source | Approx ₩ |
|------|------|--------|----------|
| Jetson Nano Dev Kit (4GB) | original 2019 Maxwell — **already owned** | NVIDIA | — |
| microSD card | 64GB UHS-I A2 (Samsung EVO Plus) | Coupang | 15,000 |
| Power supply | 5V 4A barrel jack (not micro-USB!) | Adafruit / Devicemart | 12,000 |
| Cooling fan | 4-pin PWM 40mm noctua-style | Devicemart | 18,000 |
| 7" HDMI display | Waveshare 1024×600 IPS w/ stand | Aliexpress | 80,000 |
| HDMI mini-A → A cable | 30cm | any | 5,000 |
| Wi-Fi USB dongle | Intel AX210 USB or RTL8821CU | any | 25,000 |
| USB-C battery (optional) | 20,000mAh PD output (lets you carry the box around live) | Anker | 60,000 |
| Briefcase foam-lined | "pelican-style" 350×280×130mm hard case | any | 35,000 |
| **Total** | | | **~₩250,000** |

The original Jetson Nano is a developer kit — the carrier board exposes 40-pin GPIO and an M.2 Key-E slot for the Wi-Fi card. A USB Wi-Fi dongle is simpler.

## Build steps (30 minutes)

### 1. Flash JetPack 4.6.x to the SD card

JetPack 4.6.x is the latest for the original Nano (Maxwell). It is EOL but stable.

```bash
# On a host PC with NVIDIA SDK Manager, OR use balenaEtcher with the L4T image.
# Download: https://developer.nvidia.com/embedded/jetpack-archive
# Image:    nv-jetson-nano-sd-card-image-r32.7.x.zip
```

Boot, complete first-time setup (user `jinho`, password your choice, hostname `aoi-demo`).

### 2. Run the bootstrap script

On the Nano:

```bash
git clone https://github.com/DrJinHoChoi/aoi-sentinel.git ~/aoi-sentinel
cd ~/aoi-sentinel
bash deploy/jetson_nano_poc/setup.sh
```

The script:
- Installs system packages (git, python3.8 via PPA, Nginx)
- Sets up a Python 3.8 venv
- Installs PyTorch 1.10 ARM wheel (NVIDIA-built for JetPack 4.6)
- Installs aoi-sentinel without train/RL extras (Mamba kernels won't compile here, intentionally)
- Generates a self-signed cert for the UI
- Enables the systemd service for auto-start

### 3. Generate the demo bundle

Pre-loaded synthetic data so the demo works without a live AOI:

```bash
python deploy/jetson_nano_poc/scripts/make_demo_bundle.py \
       --out ~/demo_bundle \
       --n_boards 50
```

Drops 50 fake AOI inspection results (~500 ROI components total) into `~/demo_bundle/` in the AICS-conformant generic_csv layout. The first 30 are obvious false calls, the next 15 are obvious true defects, the last 5 are subtle — designed to make the model "appear to learn" on stage.

### 4. Verify on the Nano itself

```bash
sudo systemctl start aoi-demo
xdg-open http://localhost:8080
```

You should see the operator UI. Drop the demo bundle:

```bash
ln -s ~/demo_bundle /tmp/aoi-watch    # the demo systemd unit watches /tmp/aoi-watch
```

Cards should start appearing on the UI within 5 seconds.

### 5. Pack the briefcase

- Bottom layer foam: Jetson Nano + carrier board + fan
- Cutout for: power supply, microSD spare, HDMI cable
- Top layer foam: 7" display in cradle, optional battery
- Side pocket: laser-printed 1-pager + business card + QR sticker pre-printed

## Demo flow at the meeting

See [`docs/demo_playbook_kr.md`](../../docs/demo_playbook_kr.md) for the talking points — what to say at each second.

Short version:

```
T+0:00   Power plug-in. "We connect this anywhere with USB power and Ethernet."
T+0:30   UI shows IP + QR.  "Your operator scans this — same UI on their phone."
T+0:45   Demonstrator scans QR on their own phone, holds it up.
T+1:00   Drop demo bundle.  "This is a Saki AOI export — generic CSV format, but
         we have native Saki and Koh Young adapters too."
T+1:10   Cards appear on the UI.  "Each card is one component the AOI flagged."
T+1:30   Click PASS on the first false call.  "This is what the operator does
         today, just in front of a tablet instead of fumbling through a vendor
         interface."
T+2:30   After 5-6 clicks, point at the KPI tile.  "False-call rate is already
         down 8% — and this is just synthetic demo data. In production with
         real labels, we target 70% reduction in 6 months."
T+3:30   Show the spec.  "We've open-sourced the data schema — your data
         doesn't get locked in. Standards win."
T+5:00   "If this matches what you'd want on your line, here's the 6-month
         free pilot agreement."  Hand over the printed LOI.
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Nano power-cycles under load | Using micro-USB power, current limit | Switch to 5V/4A barrel jack, set jumper J48 |
| Throttling on inference | Stock heatsink only | Mount the 40mm fan; `sudo nvpmodel -m 0` for max perf |
| `pip install mamba-ssm` fails | Maxwell GPU + JetPack 4.6 = unsupported | **Don't install it.** The POC uses MobileNetV3 only |
| UI is slow over Wi-Fi | Wi-Fi USB dongle on USB 2.0 port | Use Ethernet for the demo, or move dongle to USB 3.0 |
| Browser console errors | JetPack 4.6 ships old Chromium | Use the operator's phone instead — Chrome 120+ on Android works fine |

## Why we keep this around even after Orin Nano ships

Even when production deployments use Jetson Orin Nano, the original 2019 Nano stays as the **demo box** because:

- It's the cheapest possible "look — it works on a 5-year-old chip" proof
- Customer rep sees a literal handheld box doing real-time inference
- Sets the expectation that on-prem really means on-prem (no cloud roundtrip)
- The "USB plug-and-play" message is most credible from the smallest possible hardware
