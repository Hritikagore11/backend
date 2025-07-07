FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y libgl1-mesa-glx ffmpeg && \
    pip install --upgrade pip && \
    pip install tensorflow==2.11.0 && \
    pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
