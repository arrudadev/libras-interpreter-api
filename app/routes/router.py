from fastapi import APIRouter, File, UploadFile
from ..utils import image_from_buffer, image_landmarks, serialize_landmarks, has_two_hands_in_image, predict_signal

router = APIRouter()


@router.get("/")
def root():
  return {"response": "API is working"}


@router.post('/predict/')
async def predict(file: UploadFile = File(...)):
  buffer = await file.read()
  image = image_from_buffer(buffer)
  landmarks = image_landmarks(image)

  if landmarks:
    serialized_landmarks = serialize_landmarks(landmarks)

    if has_two_hands_in_image(serialized_landmarks):
      signal = predict_signal(serialized_landmarks)

      return {"signal": signal}
    else:
      return {"signal": "NO_SIGNAL_FOUND"}
  else:
    return {"signal": "NO_SIGNAL_FOUND"}
