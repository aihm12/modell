from flask import Flask, request, jsonify
import base64
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# تحميل النموذج
model = YOLO("yolov8_license_plate.pt")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))

        # إجراء التنبؤ
        results = model.predict(image)

        # استخراج الإحداثيات وأرقام اللوحة
        coordinates = results[0].boxes.xyxy[0].tolist()  # إحداثيات أول لوحة
        plate_number = results[0].boxes.cls[0]  # رقم اللوحة (يمكنك تخصيص هذه الخطوة حسب النموذج)

        return jsonify({
            "coordinates": coordinates,
            "plate_number": plate_number
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
