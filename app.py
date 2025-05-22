from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def swap_faces(base_path, face_path, output_path):
    base_img = cv2.imread(base_path)
    face_img = cv2.imread(face_path)

    base_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    base_faces = face_cascade.detectMultiScale(base_gray, 1.1, 4)
    face_faces = face_cascade.detectMultiScale(face_gray, 1.1, 4)

    if len(base_faces) == 0 or len(face_faces) == 0:
        return False

    (x, y, w, h) = base_faces[0]
    (fx, fy, fw, fh) = face_faces[0]

    face_crop = face_img[fy:fy+fh, fx:fx+fw]
    face_resized = cv2.resize(face_crop, (w, h))

    base_img[y:y+h, x:x+w] = face_resized
    cv2.imwrite(output_path, base_img)
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
                return render_template('index.html', error="Face not detected in one of the images.")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
