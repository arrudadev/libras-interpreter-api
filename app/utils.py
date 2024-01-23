import cv2
import numpy as np
import mediapipe as mp
from .ai.model import signals_model, scaler, pca

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()


def image_from_buffer(buffer):
  return cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)


def image_landmarks(image):
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  results = hands.process(image_rgb)
  return results.multi_hand_landmarks


def serialize_landmarks(landmarks):
  serialized_landmarks = []

  for hand_landmarks in landmarks:
    for i in range(len(hand_landmarks.landmark)):
      x = hand_landmarks.landmark[i].x
      y = hand_landmarks.landmark[i].y

      serialized_landmarks.append(x)
      serialized_landmarks.append(y)

  return serialized_landmarks


def has_two_hands_in_image(landmarks):
  number_of_two_hands_landmarks = 84

  return len(landmarks) == number_of_two_hands_landmarks


def predict_signal(landmarks):
  signals = [
      'CASA',
      'GASOLINA_LEFT',
      'GASOLINA_RIGHT',
      'PALAVRA_LEFT',
      'PALAVRA_RIGHT',
      'PEDRA_LEFT',
      'PEDRA_RIGHT'
  ]

  landmarks_reshape = np.array(landmarks).reshape(1, -1)
  landmarks_scaled = scaler.transform(landmarks_reshape)
  landmarks_pca = pca.transform(landmarks_scaled)
  model_input = {f'PC{i+1}': landmarks_pca[:, i]
                 for i in range(landmarks_pca.shape[1])}

  prediction = signals_model.predict(model_input)
  signal = signals[np.argmax(prediction)]

  return signal.split('_')[0]
