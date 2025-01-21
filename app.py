from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import base64
import io

app = Flask(__name__)

# تحميل النموذج
model = torch.jit.load('yolov8_license_plate.pt')
model.eval()

# تعريف التحويلات اللازمة
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        transformed_image = transform(image).unsqueeze(0)

        # تشغيل النموذج
        output = model(transformed_image)
        coordinates = output['coordinates'].detach().numpy().flatten().tolist()
        plate_number = output['plate_number']

        response = {
            'coordinates': coordinates,
            'plate_number': plate_number,
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
