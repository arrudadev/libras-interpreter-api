FROM python:3.9.7

WORKDIR /app

COPY . /app

RUN apt-get update && \
    apt-get install -y libgl1-mesa-dev && \
    pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]