# Boxing Bop It - AI-Powered Boxing Training Game

An AI-powered boxing game that uses computer vision to detect your punches in real-time. Like the classic Bop It toy, but for boxing - voice commands tell you which punch to throw, and an AI judges if you got it right.

![Demo GIF](demo.gif) <!-- Add a demo GIF if you have one -->

## 🎮 Features

- **Real-time punch detection** using Google's MediaPipe pose estimation
- **6 punch types**: Jab, Cross, Lead Hook, Rear Hook, Lead Uppercut, Rear Uppercut
- **Voice commands** tell you which punch to throw
- **Adaptive difficulty** - gets faster as you improve
- **89% accuracy** on trained data
- **30+ FPS** on standard laptops
- **No GPU required** - runs on CPU

## 🎥 Watch the Full Video

[YouTube Link - INSERT YOUR VIDEO LINK]

## 🚀 Quick Start

### System Requirements

- Python 3.8-3.11 (3.13 not yet supported)
- Webcam (720p recommended)
- 4GB RAM minimum
- Windows, macOS, or Linux
- No GPU required

### Installation (20 minutes)

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/boxing-bop-it.git
cd boxing-bop-it
```

2. **Create virtual environment**
```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Generate voice commands**

Go to [ElevenLabs](https://elevenlabs.io) (free account) and generate these 6 audio files:
- "JAB!" → save as `sounds/jab.mp3`
- "CROSS!" → save as `sounds/cross.mp3`
- "LEAD HOOK!" → save as `sounds/lead_hook.mp3`
- "REAR HOOK!" → save as `sounds/rear_hook.mp3`
- "LEAD UPPERCUT!" → save as `sounds/lead_uppercut.mp3`
- "REAR UPPERCUT!" → save as `sounds/rear_uppercut.mp3`

**Alternative:** Use any text-to-speech service, or record your own voice!

5. **Collect training data**

Record yourself throwing each punch type:
```bash
python data_collector.py
```

**Controls:**
- SPACE - Start/stop recording
- N - Next punch class
- P - Previous punch class
- S - Save session
- Q - Quit

**Tips:**
- Record 50-100 examples of each punch type
- Vary your speed, angle, and position
- Record "No Action" by just standing/bouncing (important!)
- Do 3-5 complete sessions for best results

6. **Train the AI model**
```bash
python train_model.py
```

This takes 5-15 minutes on CPU. Target accuracy: 85%+

7. **Play the game!**
```bash
python boxing_game.py
```

**Game Controls:**
- SPACE - Start/pause
- R - Reset score
- Q - Quit

## 📁 Project Structure
```
boxing-bop-it/
├── boxing_game.py          # Main game (Bop It mode)
├── real_time_detector.py   # Simple punch detector
├── data_collector.py       # Record training data
├── train_model.py          # Train the AI model
├── config.py              # Settings & configuration
├── utils.py               # Helper functions
├── requirements.txt       # Python dependencies
├── sounds/                # Voice command audio files
├── data/
│   └── raw/              # Your recorded training sessions
├── models/
│   └── boxing_classifier.h5   # Trained AI model
└── README.md
```

## ⚙️ Configuration

Edit `config.py` to customize:
```python
# Detection speed vs accuracy
WINDOW_SIZE = 15           # Frames to analyze (lower = faster)
CONFIDENCE_THRESHOLD = 0.65  # Min confidence to detect
DEBOUNCE_FRAMES = 5        # Stability filter

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
```

## 🐛 Troubleshooting

### Low FPS (<20)
- Close background applications
- Reduce `CAMERA_WIDTH` and `CAMERA_HEIGHT` in config.py
- Disable skeleton drawing in utils.py

### Low Accuracy (<70%)
- Record more training data (100+ examples per punch)
- Ensure good lighting
- Include variety in "No Action" class
- Increase `EPOCHS` in config.py to 100

### Sound Not Playing
- Check file names are exactly: `jab.mp3`, `cross.mp3`, etc.
- Ensure files are in `sounds/` folder
- Try `pip install pygame==2.5.2` if using newer version

### "No module named 'mediapipe'"
- Activate virtual environment: `venv\Scripts\activate`
- Reinstall: `pip install mediapipe`

## 🎯 How It Works

1. **MediaPipe Pose Detection** - Extracts 33 body landmarks from webcam
2. **Normalize Coordinates** - Makes detection position-independent
3. **Temporal Windowing** - Collects 15 frames (0.5 seconds) of movement
4. **LSTM Neural Network** - Classifies the movement sequence
5. **Debounce Filter** - Smooths predictions to prevent flickering

## 🚀 Future Roadmap

- [ ] Combo detection (Jab-Cross-Hook sequences)
- [ ] Form analysis & technique feedback
- [ ] Multiplayer mode
- [ ] Mobile app (TensorFlow Lite)
- [ ] VR integration
- [ ] Multi-sport versions (basketball, tennis, etc.)

## 🤝 Contributing

Contributions are welcome! Ideas:

- Build a basketball shooting trainer
- Add combo detection
- Improve accuracy
- Port to mobile
- Create VR version
- Add multiplayer

Fork the repo and submit a PR!

## 🙏 Credits

Built with:
- [MediaPipe](https://google.github.io/mediapipe/) - Pose estimation
- [TensorFlow](https://www.tensorflow.org/) - Deep learning
- [OpenCV](https://opencv.org/) - Computer vision
- [Pygame](https://www.pygame.org/) - Audio playback

Inspired by the classic Bop It toy and the need for accessible sports training technology.

Questions? Suggestions? Want to collaborate?

Built with heart by Henry Tasker
