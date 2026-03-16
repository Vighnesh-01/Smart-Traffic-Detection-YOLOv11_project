🚦 Intelligent Traffic System: AI-Powered Violation Detection

A high-performance, real-time traffic monitoring system built with **YOLOv11**, **ByteTrack**, and **EasyOCR**. This system automates traffic enforcement by detecting speed violations, wrong-way driving, and performing automated license plate recognition (ALPR).

---

🌟 Key Features

* **Real-Time Object Detection:** Leverages YOLOv11 for state-of-the-art vehicle detection (cars, trucks, motorcycles, buses).
* **Speed Estimation:** Mathematically calculates vehicle speed based on pixel-per-meter calibration and frame-time analysis.
* **Wrong-Way Detection:** Monitors vehicle trajectory and flags vehicles moving against the designated lane flow.
* **Tier 3 ALPR Pipeline:**
* **Stage 1:** Vehicle detection & tracking.
* **Stage 2:** License plate localization.
* **Stage 3:** Text extraction using EasyOCR.


* **Automated Telegram Alerts:** Instantly sends violation snapshots, plate numbers, and speed data to a dedicated Telegram bot for authorities.
* **Interactive Dashboard:** A Streamlit dashboard to visualize violation logs, system health, and historical data.

---

## 🛠 Tech Stack

| Component | Technology |
| **Core Language** | Python 3.10+ |
| **AI Model** | YOLOv11 (Ultralytics) |
| **Computer Vision** | OpenCV |
| **OCR Engine** | EasyOCR |
| **Tracking** | ByteTrack |
| **Dashboard** | Streamlit |
| **Messaging API** | Telegram Bot API |

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/vighnesh-b-shriyan/smart-traffic-system.git
cd smart-traffic-system

```

### 2. Environment Setup

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Activate it (Mac/Linux)
# source venv/bin/activate

# Install requirements
pip install -r requirements.txt

```

### 3. Secure Configuration

The system uses a `config.yaml` file to store sensitive API tokens.

1. Locate `config_template.yaml` in the root directory.
2. Duplicate it and rename the copy to `config.yaml`.
3. Fill in your **Telegram Bot Token** and **Chat ID**.

> **Note:** `config.yaml` is automatically ignored by Git to keep your credentials secure.

---

## 🚀 Execution

### Start the Detection System

To begin real-time video processing and violation detection:

```bash
python main.py

```

### Launch the Analytics Dashboard

To view logs and violation statistics:

```bash
streamlit run dashboard.py

```

---

## 🏗 System Architecture

1. **Ingestion:** Processes live RTSP streams or local video files.
2. **Inference:** YOLOv11 identifies vehicles; a specialized crop-model detects license plates.
3. **Analysis:** Mathematical logic checks for speed ($v = \frac{d}{t}$) and direction (Centroid Tracking).
4. **Reporting:** Violations are logged to a CSV/Database and pushed to the Telegram Bot via asynchronous requests.

---

## 🗺 Roadmap & Future Scope

* [ ] **Phase 2:** Integrate Helmet Detection for two-wheelers.
* [ ] **Phase 3:** Triple-line lane crossing detection.
* [ ] **Phase 4:** Cloud deployment for multi-camera management.

---

