import numpy as np

SIGNALS = [
    'A',
    'E',
    'I',
    'O',
    'U'
]


LANDMARKS_NAMES = [
    'wrist_x',
    'wrist_y',
    'thumb_cmc_x',
    'thumb_cmc_y',
    'thumb_mcp_x',
    'thumb_mcp_y',
    'thumb_ip_x',
    'thumb_ip_y',
    'thumb_tip_x',
    'thumb_tip_y',
    'index_finger_mcp_x',
    'index_finger_mcp_y',
    'index_finger_pip_x',
    'index_finger_pip_y',
    'index_finger_dip_x',
    'index_finger_dip_y',
    'index_finger_tip_x',
    'index_finger_tip_y',
    'middle_finger_mcp_x',
    'middle_finger_mcp_y',
    'middle_finger_pip_x',
    'middle_finger_pip_y',
    'middle_finger_dip_x',
    'middle_finger_dip_y',
    'middle_finger_tip_x',
    'middle_finger_tip_y',
    'ring_finger_mcp_x',
    'ring_finger_mcp_y',
    'ring_finger_pip_x',
    'ring_finger_pip_y',
    'ring_finger_dip_x',
    'ring_finger_dip_y',
    'ring_finger_tip_x',
    'ring_finger_tip_y',
    'pinky_finger_mcp_x',
    'pinky_finger_mcp_y',
    'pinky_finger_pip_x',
    'pinky_finger_pip_y',
    'pinky_finger_dip_x',
    'pinky_finger_dip_y',
    'pinky_finger_tip_x',
    'pinky_finger_tip_y'
]


def landmarks_to_input_data(landmarks):
  input_data = {}

  for i in range(len(landmarks)):
    input_data[LANDMARKS_NAMES[i]] = np.array([landmarks[i]], dtype=np.float32)

  return input_data


def landmark_coordinates(landmarks):
  coordinates = []
  x_coordinates = []
  y_coordinates = []

  for hand_landmarks in landmarks:
    for i in range(len(hand_landmarks.landmark)):
      x = hand_landmarks.landmark[i].x
      y = hand_landmarks.landmark[i].y

      x_coordinates.append(x)
      y_coordinates.append(y)

    for i in range(len(hand_landmarks.landmark)):
      x = hand_landmarks.landmark[i].x
      y = hand_landmarks.landmark[i].y

      coordinates.append(x - min(x_coordinates))
      coordinates.append(y - min(y_coordinates))

  return coordinates
