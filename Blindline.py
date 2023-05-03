import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
import cv2 
from time import sleep
from datetime import datetime
from tflite_runtime.interpreter import Interpreter
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
import time
import glob
import subprocess
from playsound import playsound


# The click of the button runs the proccess of capturing an image, classifying it, and then playing the corresponding mp3 sound 
def button_callback(channel):
    
    model_path = "BlindLine_Arabic.tflite"
    label_path = "arabic_labels.txt"
    sounds = 'ArabicVoice/*'
    
    labels = load_labels(label_path)

    # Opens camera and captures image
    cap = cv2.VideoCapture(-1)
    ret, frame = cap.read()
    
    now = datetime.now()
    now = str(now)

    filename = now +'.JPG'
    
    # saves image - not neccosarry just for reference 
    THE_IMAGE = cv2.imwrite(filename, frame)
    
    # Load the Model
    global interpreter
    interpreter = Interpreter(model_path)
    print("Model Loaded Successfully.")
    _, height, width, _ = interpreter.get_input_details()[0]['shape']

    # Image Preprocessing
    image = cv2.imread(filename)
    image = image / 255
    image = cv2.resize(image,(width,height))
    image = np.expand_dims(image,axis=0)
    image = image.astype('float32')

    classify_lite = interpreter.get_signature_runner('serving_default')

    # Make Predictoins
    predictions_lite = classify_lite(input_1=image)['dense']
    
    # Print(predictions_lite) Try to run through local softmax function
    score_lite = tf.nn.softmax(predictions_lite)

    classification_label = labels[np.argmax(score_lite)]
    print("Image Label is :", classification_label, ", with Accuracy :", 100 * np.max(score_lite), "%.")
    
    # Based on classification, plays the matching sound
    for file in glob.glob(sounds):
        if classification_label in file:
            print(file)
            play_mp3(file)
            print('sound complete')

    
def load_labels(path): # Read the labels from the text file as a Python list.
    class_names = []
    with open(path, 'r') as f:
        for line in f:
            x = line[:-1]
            class_names.append(x)
    f.close()
    return class_names
def play_mp3(path):
    subprocess.Popen(['mpg123','-q',path]).wait()


# The program runs on loop untill ENTER is pressed on a keyboard

GPIO.setwarnings(False) # Ignore warning for now
GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 10 to be an input pin and set initial value to be pulled low (off)

GPIO.add_event_detect(10,GPIO.FALLING,callback=button_callback) # Setup event on pin 10 rising edge

message = input("Press enter to quit\n\n") # Run until someone presses enter

GPIO.cleanup() # Clean up