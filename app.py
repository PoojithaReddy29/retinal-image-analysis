import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model for Diabetic Retinopathy detection
model = load_model('retinal_model.h5')

# Define the class labels
class_labels = [
    'Dry AMD', 'Glaucoma', 'Normal Fundus', 'Wet AMD', 'Mild DR',
    'Moderate DR', 'Severe DR', 'Proliferate DR', 'Cataract',
    'Hypertensive Retinopathy', 'Pathological Myopia'
]

# Define severity-based advice
severity_advice = {
    0: {
        "advice": "Dry AMD (Age-Related Macular Degeneration) detected. Consult with an ophthalmologist for possible treatments such as anti-VEGF therapy and lifestyle adjustments like diet and quitting smoking to slow progression. Avoid smoking, maintain a healthy diet rich in antioxidants, and protect eyes from UV light by wearing sunglasses. Regular eye check-ups are essential to monitor progression."
    },
    1: {
        "advice": "Glaucoma detected. Seek immediate consultation with an ophthalmologist. Early treatment with medications or surgery can prevent irreversible vision loss. Monitor eye pressure regularly, use prescribed eye drops as directed, avoid lifting heavy objects, and maintain a healthy weight and blood pressure to reduce strain on the eyes."
    },
    2: {
        "advice": "Normal Fundus detected. Maintain eye health with regular check-ups and a balanced diet, including antioxidants, to prevent future issues. Avoid eye strain by taking regular breaks from screens, maintain a balanced diet rich in vitamins A, C, and E, and protect eyes from UV exposure."
    },
    3: {
        "advice": "Wet AMD (Age-Related Macular Degeneration) detected. Urgent consultation with an ophthalmologist is required. Anti-VEGF injections or photodynamic therapy might be recommended for treatment. Avoid smoking and excessive alcohol. Protect eyes from UV light and ensure regular eye exams to monitor disease progression."
    },
    4: {
        "advice": "Mild Diabetic Retinopathy detected. Schedule an appointment with an ophthalmologist for further evaluation and better blood sugar control to prevent worsening of the condition. Control blood sugar levels through a balanced diet, regular exercise, and medication as prescribed. Regular eye exams are crucial to catch any changes early."
    },
    5: {
        "advice": "Moderate Diabetic Retinopathy detected. Regular follow-ups and better glucose control are recommended to prevent further progression, and laser therapy may be considered. Maintain tight control over blood sugar levels, monitor blood pressure, and avoid smoking. Regular eye exams will help in early detection of further damage."
    },
    6: {
        "advice": "Severe Diabetic Retinopathy detected. Immediate consultation with an ophthalmologist is essential for potential treatments such as laser therapy or surgery to prevent significant vision loss. Tight blood sugar control is crucial. Avoid smoking and excessive alcohol, and maintain regular check-ups to monitor eye health. Laser therapy or injections may be needed to prevent further damage."
    },
    7: {
        "advice": "Proliferative Diabetic Retinopathy detected. Urgent medical intervention is required. Vitrectomy surgery or anti-VEGF injections may be necessary to save vision. Strict control of blood sugar and blood pressure is essential. Regular follow-ups with an ophthalmologist are necessary for timely intervention to prevent vision loss."
    },
    8: {
        "advice": "Cataract detected. Surgery might be necessary, please consult an ophthalmologist. Cataract surgery can restore vision effectively and improve quality of life. Wear sunglasses to protect eyes from UV rays, avoid smoking, and maintain a healthy lifestyle. Regular eye exams are essential to track the cataract's progression."
    },
    9: {
        "advice": "Hypertensive Retinopathy detected. Consult with an ophthalmologist and manage blood pressure. Strict hypertension control can prevent further retinal damage and preserve vision. Follow a healthy diet low in salt, exercise regularly, and take prescribed medications to control blood pressure. Routine eye exams are important for monitoring any retinal changes."
    },
    10: {
        "advice": "Pathological Myopia detected. Consult with an ophthalmologist for potential treatment, which may include corrective surgery, and to monitor for complications such as retinal detachment or macular degeneration. Regular eye exams are crucial to monitor for complications. Protect eyes from trauma and avoid excessive strain from close-up work or screen use."
    }
}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/prediction')
def prediction():
    return render_template('prediction.html')


@app.route('/analysis')
def analysis():
    return render_template('analysis.html')


def highlight_damage(image_path, output_path="static/uploads/highlighted_damage.png", threshold_path="static/uploads/threshold_image.png"):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    highlighted_img = img.copy()
    cv2.drawContours(highlighted_img, contours, -1, (0, 0, 255), 2)

    cv2.imwrite(output_path, highlighted_img)
    cv2.imwrite(threshold_path, thresholded)


    return img, highlighted_img, thresholded


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        original_img, highlighted_img, thresholded_img = highlight_damage(file_path)

        img = image.load_img(file_path, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_labels[predicted_class]
        advice = severity_advice.get(predicted_class, {"advice": "No advice available"})

        original_img_base64 = base64.b64encode(cv2.imencode('.png', original_img)[1]).decode('utf-8')
        highlighted_img_base64 = base64.b64encode(cv2.imencode('.png', highlighted_img)[1]).decode('utf-8')
        threshold_img_base64 = base64.b64encode(cv2.imencode('.png', thresholded_img)[1]).decode('utf-8')

        return jsonify({
            'prediction': predicted_label,
            'advice': advice['advice'],
            'original_image': original_img_base64,
            'highlighted_image': highlighted_img_base64,
            'threshold_image': threshold_img_base64
        })

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 400


if __name__ == '__main__':
    app.run(debug=True)
