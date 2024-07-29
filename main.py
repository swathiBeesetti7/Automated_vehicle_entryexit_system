from flask import Flask, render_template, request,flash,redirect
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2
import os
import pandas as pd
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the pre-trained model and labels
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Function to process image and make predictions
def predict_image(image):
    # Prepare the image for prediction
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make predictions
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

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

# Function to save data to Excel file
def save_to_excel(excel_file, predictions):
    try:
        # Try to load an existing Excel file
        recorded_predictions = pd.read_excel(excel_file)
    except FileNotFoundError:
        # If it does not exist, initialize an empty DataFrame
        recorded_predictions = pd.DataFrame(columns=['Class Name', 'Timestamp'])

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for class_name in predictions:
        new_record = pd.DataFrame([[class_name, current_time]],
                                  columns=['Class Name', 'Timestamp'])
        recorded_predictions = pd.concat([recorded_predictions, new_record], ignore_index=True)

    recorded_predictions.to_excel(excel_file, index=False)


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("home.html")

@app.route("/selectImage",methods = ['GET', 'POST'])
def selectImage():
     return render_template("image.html")

@app.route("/selectVideo",methods = ['GET', 'POST'])
def selectVideo():
     return render_template("video.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'my_image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['my_image']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # Process the uploaded image and make predictions
            image = Image.open(file).convert("RGB")
            class_name, confidence_score = predict_image(image)
            img_path = "static/" + file.filename
            image.save(img_path)
            return render_template("image.html", prediction=class_name[2:], confidence_score=confidence_score, img_path=img_path)
    return render_template("image.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        img = request.files['my_video']
        vd_path = "static/" + img.filename
        img.save(vd_path)

        folder_path = "Records"

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Get the current date and time
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Create a filename with the current date, .xlsx extension, and folder path
        filename = os.path.join(folder_path, f"model_{current_date}.xlsx")

        predictions = predict_video(vd_path)

        val=[]
        for pred in predictions:
             if pred[0] not in val:
                  val.append(pred[0])

        save_to_excel(filename, val)

        max_acc= max(enumerate(predictions), key=lambda x: x[1][1])[0]
        prediction=predictions[max_acc][0]
        # Rendering the template with predictions and video path
        return render_template("video.html", predictions=prediction, img_path=vd_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)