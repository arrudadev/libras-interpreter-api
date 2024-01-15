import cv2
import mediapipe as mp
import utils
import requests
from requests.exceptions import HTTPError

PREDICT_URL = 'http://localhost:8000/predict/'

capture = cv2.VideoCapture(2)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

try:
  while True:
    _, frame = capture.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    landmarks = results.multi_hand_landmarks

    if landmarks:
      coordinates = utils.landmark_coordinates(landmarks)

      if len(coordinates) == 42:
        body = {'landmarks': coordinates}
        response = requests.post(PREDICT_URL, json=body)
        response.raise_for_status()
        print(response.text)

    cv2.imshow('frame', frame)

    letter_q_key = 25
    if cv2.waitKey(letter_q_key) == ord('q'):
      break
except HTTPError as http_err:
  print(f'HTTP error occurred: {http_err}')
except Exception as err:
  print(f'An error occurred: {err}')

capture.release()
cv2.destroyAllWindows()
