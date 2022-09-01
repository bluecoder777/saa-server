import tensorflow as tf  # noqa E402
import cv2
from tensorflow.keras.models import load_model


from fastapi import FastAPI, File, 


# load model
path = r"C:\Users\fed\Documents\MiniProject\MiniServer\trainedmodel50.h6"
photo =r"C:\Users\fed\Documents\MiniProject\MiniServer\Pimtest.jpg"
model = load_model(path)

# analysis
import numpy as np
from tensorflow import keras
app = FastAPI()


@app.get("/")
async def predict(file: bytes = File()):
	with open('image.jpg','wb') as image:
		image.write(file)
	img = cv2.imread(photo)
	img_2 = cv2.resize(img,(224,224))
	img2 = np.array(img_2)
	img_2 = img_2/255
	img_2 = tf.expand_dims(img_2, 0)

	# predictions
	predicted = model.predict([img_2])
	predicted = np.argmax(predicted, axis=1)
	result = predicted[0]
	if result == 0:
		return {"message": "pimples"}
	else: 
		return {"message": "no pimples"}


    