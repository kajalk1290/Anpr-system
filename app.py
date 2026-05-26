import os
import cv2
import numpy as np
import re
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DETECTED_FOLDER'] = 'detected'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTED_FOLDER'], exist_ok=True)

detection_log = []

# FIX 1: Lazy EasyOCR - app will NOT crash if easyocr missing
_ocr_reader = None
_ocr_tried = False

def get_ocr_reader():
    global _ocr_reader, _ocr_tried
    if _ocr_tried:
        return _ocr_reader
    _ocr_tried = True
    try:
        import easyocr
        print("[ANPR] Loading EasyOCR...")
        _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("[ANPR] EasyOCR ready.")
    except Exception as e:
        print(f"[ANPR] EasyOCR unavailable: {e}. Using fallback.")
        _ocr_reader = None
    return _ocr_reader

def preprocess_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w < 200:
        scale = 200 / w
        gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def detect_plates_classical(img):
    plates = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    haar_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
    if os.path.exists(haar_path):
        try:
            cascade = cv2.CascadeClassifier(haar_path)
            detected = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,20))
            for (x, y, w, h) in detected:
                plates.append((x, y, w, h))
        except Exception:
            pass
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.018*peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / float(h)
            if 2.0 < aspect < 6.5 and w > 60 and h > 15 and w*h > 1500:
                dup = any(abs(x-px)<30 and abs(y-py)<30 for (px,py,pw,ph) in plates)
                if not dup:
                    plates.append((x, y, w, h))
    return plates

def extract_text(img_region):
    reader = get_ocr_reader()
    if reader is not None:
        try:
            results = reader.readtext(img_region, detail=1, paragraph=False)
            texts = []
            for (bbox, text, conf) in results:
                clean = re.sub(r'[^A-Z0-9\-]', '', text.upper().strip())
                if clean and len(clean) >= 3 and conf > 0.25:
                    texts.append({'text': clean, 'confidence': round(conf*100, 1)})
            if texts:
                return texts
        except Exception as e:
            print(f"[OCR Error] {e}")
    return [{'text': 'PLATE_DETECTED', 'confidence': 50.0}]

def process_image(img_bytes):
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, [], "Could not decode image"
        h, w = img.shape[:2]
        if w > 1280:
            scale = 1280/w
            img = cv2.resize(img, (1280, int(h*scale)))
        annotated = img.copy()
        plate_results = []
        plates = detect_plates_classical(img)
        if not plates:
            preprocessed = preprocess_plate(img)
            texts = extract_text(preprocessed)
            if texts:
                plate_results.append({'box': None, 'texts': texts,
                    'best_text': texts[0]['text'], 'confidence': texts[0]['confidence']})
            cv2.putText(annotated, "Scanning full frame...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
        else:
            for (x, y, pw, ph) in plates:
                pad = 6
                x1=max(0,x-pad); y1=max(0,y-pad)
                x2=min(img.shape[1],x+pw+pad); y2=min(img.shape[0],y+ph+pad)
                plate_roi = img[y1:y2, x1:x2]
                preprocessed = preprocess_plate(plate_roi)
                texts = extract_text(preprocessed)
                best_text = texts[0]['text'] if texts else 'UNREADABLE'
                conf = texts[0]['confidence'] if texts else 0.0
                plate_results.append({'box':[x1,y1,x2,y2],'texts':texts,
                    'best_text':best_text,'confidence':conf})
                cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),3)
                label = f"{best_text} ({conf:.0f}%)"
                (lw,lh),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated,(x1,y1-lh-14),(x1+lw+10,y1),(0,255,0),-1)
                cv2.putText(annotated,label,(x1+4,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)
        _,buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        return img_b64, plate_results, None
    except Exception as e:
        print(f"[Pipeline Error] {e}")
        return None, [], str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        img_bytes = file.read()
        img_b64, plates, err = process_image(img_bytes)
        if err:
            return jsonify({'error': err}), 500
        if img_b64 is None:
            return jsonify({'error': 'Could not process image'}), 500
        entry = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'filename': file.filename,
            'plates': [p['best_text'] for p in plates],
            'count': len(plates)
        }
        detection_log.insert(0, entry)
        if len(detection_log) > 50:
            detection_log.pop()
        return jsonify({'image': img_b64, 'plates': plates, 'count': len(plates)})
    except Exception as e:
        print(f"[/upload Error] {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/webcam_frame', methods=['POST'])
def webcam_frame():
    try:
        # FIX 2: force=True + silent=True handles malformed JSON gracefully
        data = request.get_json(force=True, silent=True)
        if not data or 'frame' not in data:
            return jsonify({'error': 'No frame data'}), 400
        frame_data = data['frame']
        if ',' in frame_data:
            frame_data = frame_data.split(',', 1)[1]
        img_bytes = base64.b64decode(frame_data)
        img_b64, plates, err = process_image(img_bytes)
        if err:
            return jsonify({'error': err}), 500
        if plates:
            entry = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'filename': 'webcam',
                'plates': [p['best_text'] for p in plates],
                'count': len(plates)
            }
            detection_log.insert(0, entry)
            if len(detection_log) > 50:
                detection_log.pop()
        return jsonify({'image': img_b64, 'plates': plates, 'count': len(plates)})
    except Exception as e:
        print(f"[/webcam_frame Error] {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/log')
def get_log():
    try:
        return jsonify(detection_log[:20])
    except Exception:
        return jsonify([])

@app.route('/clear_log', methods=['POST'])
def clear_log():
    detection_log.clear()
    return jsonify({'status': 'cleared'})

@app.route('/status')
def status():
    reader = get_ocr_reader()
    return jsonify({'status':'running','easyocr': reader is not None})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  ANPR System Starting...")
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    get_ocr_reader()
    app.run(debug=False, host='0.0.0.0', port=5000)
