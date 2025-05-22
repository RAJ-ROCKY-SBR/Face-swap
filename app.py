
from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def swap_faces(base_path, face_path, output_path):
    base_img = cv2.imread(base_path)
    face_img = cv2.imread(face_path)

    base_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    base_faces = face_cascade.detectMultiScale(base_gray, 1.3, 5)
    face_faces = face_cascade.detectMultiScale(face_gray, 1.3, 5)

    if len(base_faces) == 0 or len(face_faces) == 0:
        return False

    (x, y, w, h) = base_faces[0]
    (fx, fy, fw, fh) = face_faces[0]

    face_crop = face_img[fy:fy+fh, fx:fx+fw]
    face_resized = cv2.resize(face_crop, (w, h))

    mask = 255 * np.ones(face_resized.shape, face_resized.dtype)
    center = (x + w // 2, y + h // 2)
    blended = cv2.seamlessClone(face_resized, base_img, mask, center, cv2.NORMAL_CLONE)
    cv2.imwrite(output_path, blended)
    return True

@app.route('/', methods=['GET', 'POST'])
def index():
    result_image = None
    error = None
    if request.method == 'POST':
        base = request.files['base']
        face = request.files['face']

        if base and face:
            base_filename = secure_filename(base.filename)
            face_filename = secure_filename(face.filename)

            base_path = os.path.join(app.config['UPLOAD_FOLDER'], base_filename)
            face_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')

            base.save(base_path)
            face.save(face_path)

            success = swap_faces(base_path, face_path, output_path)
            if success:
                result_image = output_path
            else:
                error = "Face not detected in one or both images."

    return render_template('index.html', result_image=result_image, error=error)

if __name__ == '__main__':
    app.run(debug=True)
