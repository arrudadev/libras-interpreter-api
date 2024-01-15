from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import numpy as np
import utils

app = FastAPI()
model = tf.keras.models.load_model('model')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LandmarksList(BaseModel):
  landmarks: List[float]


@app.get("/")
def hello_word():
  return {"response": "hello word"}


@app.post("/predict/")
def predict(landmarks_list: LandmarksList):
  input_data = utils.landmarks_to_input_data(landmarks_list.landmarks)
  prediction = model.predict(input_data)
  signal = utils.SIGNALS[np.argmax(prediction)]

  return {"signal": signal}
