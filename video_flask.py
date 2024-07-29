from flask import Flask, render_template, request
from keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

def predict_video(video_path):
    camera = cv2.VideoCapture(video_path)
    predictions = []

    while True:
        ret, image = camera.read()
        if not ret:
            break

        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1

        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        predictions.append((class_name[2:], confidence_score * 100))

    camera.release()
    
    return predictions

@app.route('/')
def home():
    return render_template("video.html")

@app.route("/submit", methods=['POST'])
def submit():
    if request.method == 'POST':
        img = request.files['my_video']
        img_path = "static/" + img.filename
        img.save(img_path)

        predictions = predict_video(img_path)

        max_acc= max(enumerate(predictions), key=lambda x: x[1][1])[0]
        predictions=predictions[max_acc][0]
        # Rendering the template with predictions and video path
        return render_template("video.html", predictions=predictions, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
