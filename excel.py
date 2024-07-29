from keras.models import load_model
import cv2
import os
import numpy as np
import pandas as pd  # Import pandas for Excel operations
from datetime import datetime


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)


# Load the model
model = load_model("keras_model.h5", compile=False)


# Load the labels
class_names = [name.strip() for name in open("labels.txt", "r").readlines()]


# Specify the existing folder path
folder_path = "Records"


if not os.path.exists(folder_path):
    os.makedirs(folder_path)


# Get the current date and time
current_date = datetime.now().strftime("%Y-%m-%d")


# Create a filename with the current date, .xlsx extension, and folder path
filename = os.path.join(folder_path, f"model_{current_date}.xlsx")


# Initialize or load an existing Excel file for unique bike numbers
excel_file = filename
try:
    # Try to load an existing Excel file
    recorded_bikes = pd.read_excel(excel_file)
except FileNotFoundError:
    # If it does not exist, initialize an empty DataFrame
    recorded_bikes = pd.DataFrame(columns=['Bike Number', 'Timestamp'])


# Function to check if a bike number has already been recorded
def is_bike_number_recorded(bike_number, df):
    return bike_number in df['Bike Number'].values


# CAMERA can be 0 or 1 based on the default camera of your computer
camera = cv2.VideoCapture('static\VID20240320151959.mp4')


while True:
    # Grab the webcam's image.
    ret, image = camera.read()
    if not ret:
        print("Failed to grab frame. Skipping...")
        break


    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)


    # Convert image to numpy array and preprocess for the model
    image_array = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image_array = (image_array / 127.5) - 1


    # Predict the model
    prediction = model.predict(image_array)
    index = np.argmax(prediction)
    bike_number = class_names[index]
   
    # If the bike number has not been recorded, append it to the DataFrame and save
    if not is_bike_number_recorded(bike_number, recorded_bikes):
        new_record = pd.DataFrame([[bike_number, datetime.now()]], columns=['Bike Number', 'Timestamp'])
        recorded_bikes = pd.concat([recorded_bikes, new_record], ignore_index=True)
        recorded_bikes.to_excel(excel_file, index=False)
        print(f"Recorded New Bike Number: {bike_number}")


    # Show the image in a window (optional, can be disabled for faster performance)
    # cv2.imshow("Webcam Image", image)
    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)


    # Listen to the keyboard for presses.
    if keyboard_input == 27:
        break


camera.release()
cv2.destroyAllWindows()