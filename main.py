from fastapi import FastAPI, File, UploadFile
from fastapi import WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import numpy as np
import mediapipe as mp
import cv2
import utils
import json

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

app = FastAPI()
model = tf.keras.models.load_model('model')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LandmarksDict(BaseModel):
  landmarks: List[float]


@app.get("/")
def hello_word():
  return {"response": "hello word"}


@app.post("/predict/")
def predict(landmarks_dict: LandmarksDict):
  input_data = utils.landmarks_to_input_data(landmarks_dict.landmarks)
  prediction = model.predict(input_data)
  signal = utils.SIGNALS[np.argmax(prediction)]

  return {"signal": signal}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
  await websocket.accept()
  while True:
    data = await websocket.receive_text()
    landmarks_dict = json.loads(data)
    input_data = utils.landmarks_to_input_data(landmarks_dict['landmarks'])
    prediction = model.predict(input_data)
    signal = utils.SIGNALS[np.argmax(prediction)]

    await websocket.send_text(signal)


@app.post('/predict-image/')
async def predict_image(file: UploadFile = File(...)):
  buffer = await file.read()
  image = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)

  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  results = hands.process(image_rgb)
  landmarks = results.multi_hand_landmarks

  if landmarks:
    coordinates = utils.landmark_coordinates(landmarks)

    if len(coordinates) == 42:
      input_data = utils.landmarks_to_input_data(coordinates)
      prediction = model.predict(input_data)
      signal = utils.SIGNALS[np.argmax(prediction)]

      return {"signal": signal}
    else:
      return {"signal": "NO_SIGNAL_FOUND"}
  else:
    return {"signal": "NO_SIGNAL_FOUND"}
