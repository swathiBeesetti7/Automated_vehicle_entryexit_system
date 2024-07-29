from flask import Flask, render_template, request,flash,redirect
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

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


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("image.html")


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


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)