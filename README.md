# 🚗 ANPR — Automatic Number Plate Recognition System

Real-time license plate detection and OCR using OpenCV + EasyOCR.

---

## ⚙️ Tech Stack

| Component | Library |
|-----------|---------|
| Detection | OpenCV (Haar Cascade + Contour-based) |
| OCR | EasyOCR |
| Backend | Flask (Python) |
| Frontend | Vanilla HTML/CSS/JS |

---

## 🚀 Setup & Run

### 1. Install Python 3.9+

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Server
```bash
python app.py
```

### 4. Open Browser
```
http://localhost:5000
```

---

## 🎯 Features

- **Image Upload Mode** — Upload any vehicle photo, get annotated output with plate text
- **Live Webcam Mode** — Real-time detection from browser camera (processes every 2 seconds)
- **Detection Log** — Persistent log of all detected plates with timestamps
- **Copy to Clipboard** — One-click copy of detected plate numbers
- **Confidence Score** — Shows OCR confidence for each detected plate

---

## 📁 Project Structure

```
anpr_system/
├── app.py              # Flask backend + detection pipeline
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Frontend UI
├── uploads/            # Temp uploaded images
└── detected/           # Output annotated images
```

---

## 🔧 How It Works

1. **Image Input** — User uploads image or browser captures webcam frame
2. **Preprocessing** — Bilateral filter + Canny edges
3. **Plate Detection** — Haar cascade + contour-based detection
4. **ROI Extraction** — Crop + resize plate region
5. **OCR** — EasyOCR reads characters
6. **Annotation** — Bounding box + plate text drawn on output image

---

## 📌 Tips for Best Results

- Use clear, well-lit vehicle images
- Plates should be at least 60px wide
- Works best with standard rectangular plates
- For webcam: ensure good lighting and plate is visible in frame

---

## 🧪 Test Dataset

Download from Kaggle:
https://www.kaggle.com/datasets/andrewmvd/car-plate-detection

---

## 📝 Notes

- First run will download EasyOCR model (~100MB) automatically
- GPU not required; CPU inference works fine for real-time use
- For production: add YOLO-based plate detection for better accuracy on difficult angles
