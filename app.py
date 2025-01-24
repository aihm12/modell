from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image, ImageStat, ImageEnhance, ImageOps
import pytesseract
import base64
import io
import cv2
import numpy as np

app = Flask(__name__)

# تحميل النموذج
model = YOLO('yolov8_license_plate.pt')

def get_dominant_color(image):
    """تحليل اللون الغالب في الصورة"""
    image = image.convert("RGB")
    stat = ImageStat.Stat(image)
    r, g, b = stat.mean  # متوسط الألوان (RGB)
    if r > g and r > b:
        return "red"
    elif g > r and g > b:
        return "yellow"
    elif b > r and b > g:
        return "blue"
    return "unknown"

def preprocess_image_with_opencv(image):
    """تحسين الصورة باستخدام OpenCV"""
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blurred = cv2.GaussianBlur(binary, (5, 5), 0)
    return Image.fromarray(blurred)

def crop_bottom_third(image):
    """قص الثلث السفلي من الصورة"""
    width, height = image.size
    return image.crop((0, int(height * 2 / 3), width, height))

def split_bottom_section(image):
    """تقسيم الثلث السفلي إلى قسمين"""
    width, height = image.size
    city_code = image.crop((0, 0, int(width / 4), height))  # الربع الأيسر
    plate_number = image.crop((int(width / 4), 0, width, height))  # الباقي
    return city_code, plate_number

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the License Plate Recognition API. Use /predict endpoint for predictions.", 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # تشغيل الموديل والحصول على النتيجة
        results = model.predict(image)
        
        # استخراج الإحداثيات (أول صندوق فقط)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return jsonify({'error': 'No license plate detected'}), 400
        
        coordinates = boxes[0].tolist()  # أول صندوق (الإحداثيات)
        x_min, y_min, x_max, y_max = [int(coord) for coord in coordinates]

        # قص الصورة بناءً على الإحداثيات
        cropped_plate = image.crop((x_min, y_min, x_max, y_max))

        # فحص اللون الغالب
        dominant_color = get_dominant_color(cropped_plate)
        if dominant_color == "red":
            plate_type = "نقل"
        elif dominant_color == "yellow":
            plate_type = "أجرة"
        elif dominant_color == "blue":
            plate_type = "خصوصي"
        else:
            plate_type = "غير معروف"

        # قص الثلث السفلي
        bottom_third = crop_bottom_third(cropped_plate)

        # تقسيم الثلث السفلي إلى قسمين
        city_code_image, plate_number_image = split_bottom_section(bottom_third)

        # تحسين الصور قبل OCR
        processed_city_code = preprocess_image_with_opencv(city_code_image)
        processed_plate_number = preprocess_image_with_opencv(plate_number_image)

        # قراءة النصوص باستخدام OCR
        city_code = pytesseract.image_to_string(
            processed_city_code, lang="eng", config="--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ).strip()
        plate_number = pytesseract.image_to_string(
            processed_plate_number, lang="eng", config="--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ).strip()

        response = {
            'coordinates': [x_min, y_min, x_max, y_max],
            'city_code': city_code,
            'plate_number': plate_number,
            'plate_type': plate_type
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='192.168.0.44', port=10000)
