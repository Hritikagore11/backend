FROM tensorflow/tensorflow:2.11.0-py3.9

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y libgl1-mesa-glx ffmpeg && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
