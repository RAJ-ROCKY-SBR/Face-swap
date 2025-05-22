
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import face_recognition
import cv2
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def swap_faces(base_img_path, face_img_path, output_path):
    base_image = face_recognition.load_image_file(base_img_path)
    face_image = face_recognition.load_image_file(face_img_path)

    base_face_locations = face_recognition.face_locations(base_image)
    face_face_locations = face_recognition.face_locations(face_image)

    if not base_face_locations or not face_face_locations:
        return False

    base_encoding = face_recognition.face_encodings(base_image, base_face_locations)[0]
    face_encoding = face_recognition.face_encodings(face_image, face_face_locations)[0]

    # Replace face area with new face (simple placeholder logic)
    base_image_cv2 = cv2.imread(base_img_path)
    face_image_cv2 = cv2.imread(face_img_path)
    face_image_cv2 = cv2.resize(face_image_cv2, (base_image_cv2.shape[1], base_image_cv2.shape[0]))

    result = cv2.addWeighted(base_image_cv2, 0.5, face_image_cv2, 0.5, 0)
    cv2.imwrite(output_path, result)
    return True

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        base_file = request.files['base']
        face_file = request.files['face']

        if base_file and face_file:
            base_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(base_file.filename))
            face_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(face_file.filename))
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')

            base_file.save(base_path)
            face_file.save(face_path)

            success = swap_faces(base_path, face_path, output_path)
            if success:
                return render_template('index.html', result_image=output_path)
            else:
                return render_template('index.html', error="Face not detected. Try clearer images.")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
