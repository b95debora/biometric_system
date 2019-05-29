import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
import keras
from keras import regularizers
import pickle
from sklearn.metrics import classification_report


gender_model = tf.keras.models.load_model('gender.model')
race_model = tf.keras.models.load_model('race.model')
age_model = tf.keras.models.load_model('age3.model')

GENDER_CATEGORIES = ["male", "female"]
RACE_CATEGORIES = ["white", "black", "asian", "indian", "others"]
AGE_CATEGORIES = ["0-5", "8-13", "15-20", "25-32", "38-50", "55-70", "80-116"]

cap = cv2.VideoCapture(0)


font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (30,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

while True:
	
	
	ret, image_np = cap.read()
	image = cv2.resize(image_np,(800,600))
	cv2.imshow('image', image)
	face_img = image.copy()
	face_detected = 0
	gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.1, 5)
	if(len(faces)==1):
		face_detected = 1
		for (x, y, w, h )in faces:
			face_img = image[y:y+h, x:x+w].copy()

	x = []
	new_array = cv2.resize(face_img, (150, 150)) 
	x.append(new_array)
	x = np.array(x).reshape(-3, 150, 150, 3)
	x = x/255.0
	predicted_gender = gender_model.predict(x).argmax(axis=-1)
	predicted_gender = GENDER_CATEGORIES[predicted_gender.item(0)]
	predicted_race = race_model.predict(x).argmax(axis=-1)
	predicted_race = RACE_CATEGORIES[predicted_race.item(0)]
	predicted_age = age_model.predict(x).argmax(axis=-1)
	predicted_age = AGE_CATEGORIES[predicted_age.item(0)]
	output = predicted_gender+" - "+predicted_race+" - "+predicted_age
	cv2.putText(image, output, bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
	cv2.imshow('image', image)
	print(output)
	

	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		cap.release()
		break

