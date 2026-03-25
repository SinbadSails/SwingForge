# SwingForge AI

**Drop a video. Mirror a legend. Fix your swing.**

Tennis swing analysis powered by computer vision and biomechanics. Real-time webcam analysis + video upload + pro player comparison — all running locally on your machine.

## Features

- **Upload & Analyze** — Drop a swing video, get pose-annotated output with SwingScore (0-100) and coaching feedback
- **Live Webcam** — Real-time pose estimation and joint angle readouts as you swing
- **Pro Compare** — Overlay your skeleton against Djokovic, Alcaraz, Federer, Nadal, or Medvedev at the contact frame
- **Swing Plane** — Visualize your swing arc angle (flat vs topspin vs heavy topspin)
- **Ball Flight** — Simulate ball trajectory with drag and Magnus force, Monte Carlo landing distribution
- **Shadow Mode** — Mirror a pro's ghost skeleton in real-time via OpenCV (keyboard controlled)

## Tech Stack

| Layer | Tool |
|-------|------|
| Pose Estimation | MediaPipe Pose (33 keypoints) |
| Player Detection | YOLOv8 |
| Ball Tracking | Fine-tuned YOLOv5 |
| Court Detection | ResNet50 (14 keypoints) |
| Physics | NumPy + SciPy (projectile + drag + Magnus) |
| UI | Gradio (web) + OpenCV (live shadow mode) |
| Video I/O | OpenCV |

## Quick Start

```bash
git clone https://github.com/SinbadSails/SwingForge.git
cd SwingForge
pip install -r requirements.txt
python main.py
```

Open http://localhost:7860 in your browser.

### Other modes

```bash
# CLI analysis
python main.py --analyze path/to/your_swing.mp4

# Live shadow mode (OpenCV webcam)
python main.py --live

# Left-handed player
python main.py --hand left
```

## Shadow Mode Controls

| Key | Action |
|-----|--------|
| SPACE | Start/stop ghost playback |
| P | Cycle pros |
| S | Slow ghost (25% increments) |
| F | Speed up ghost |
| R | Record session to .mp4 |
| Q | Quit |

## Project Structure

```
SwingForge/
├── core/
│   ├── pose_engine.py         # MediaPipe pose extraction + joint angles
│   ├── swing_classifier.py    # Swing phase detection + stroke classification
│   ├── coaching.py            # Scoring engine + coaching text generation
│   └── physics.py             # Ball trajectory simulation
├── modules/
│   └── live_shadow.py         # OpenCV shadow swing mode
├── trackers/                  # YOLOv8 player + ball tracking (from base)
├── court_line_detector/       # ResNet50 court keypoint detection (from base)
├── mini_court/                # Mini court visualization (from base)
├── utils/                     # Video I/O, bbox utils, conversions
├── data/pros/                 # Pro player reference keypoint data
├── ui/app.py                  # Gradio web interface
├── pipeline.py                # Original tennis_analysis pipeline
└── main.py                    # Entry point
```

## Requirements

- Python 3.10+
- macOS (Apple Silicon M1/M2/M3) or Linux/Windows
- Webcam (for live modes)
- No GPU required (CPU inference, M1 GPU accelerated where available)
- No internet required at runtime

## Credits

Built on top of [abdullahtarek/tennis_analysis](https://github.com/abdullahtarek/tennis_analysis) — YOLO player/ball detection, court keypoint detection, and video analysis pipeline.

## License

MIT
