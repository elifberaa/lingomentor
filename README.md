# LingoMentor 🎓🤖

LingoMentor is an interactive language learning platform equipped with an intelligent, AI-powered **Zen Mode** that tracks real-time user fatigue using the computer's webcam.

## Features ✨

- **Language Modules:** Comprehensive topics ranging from beginner to advanced vocabulary, reading exercises, and grammar.
- **AI Zen Mode (Fatigue Detection):** 
  - Utilizes **TensorFlow.js** directly in the browser to run a custom-trained **MobileNetV2** Keras 3 model.
  - Monitors the user's eye state (open/closed) in real time (at 60 FPS) via the webcam.
  - If the user's eyes are closed continuously for more than 3 seconds, a "Zen Mode" notification triggers, encouraging the user to take a break and rest their eyes.
  - Extremely privacy-centric: All video processing and inference happens 100% locally in the browser. No images or videos are sent to any server.
- **Modern UI:** Clean, aesthetic, and responsive HTML/CSS design.

## How it Works (Zen Mode) 👁️

1. The user grants webcam permissions securely via the browser.
2. The hidden `<video>` element captures the user's face.
3. The custom `tfjs` layers-model processes the frames and predicts `Class 0` (Active/Eyes Open) vs `Class 1` (Fatigued/Eyes Closed).
4. A highly optimized time-based detection algorithm calculates continuous fatigue duration to prevent false positives from natural eye blinking.

## Setup & Running Locally 💻

Since the project uses ES6 modules and fetches a local `model.json` file, it must be served via a local web server (due to CORS policies).

1. Clone the repository:
   ```bash
   git clone https://github.com/elifberaa/lingomentor.git
   cd lingomentor
   ```
2. Start a local server. You can use VS Code's **Live Server** extension, or use Python:
   ```bash
   python3 -m http.server 5500
   ```
3. Open your browser and navigate to:
   ```
   http://127.0.0.1:5500/giris.html
   ```

## Tech Stack 🛠️
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **AI/ML:** TensorFlow.js (@tensorflow/tfjs)
- **Model:** MobileNetV2 (Exported from Keras 3 and polyfilled for web compatibility)

---
*Created for an enhanced and healthy learning experience.*
