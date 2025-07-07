FROM python:3.10

WORKDIR /app
COPY . .

# Add this trick to block TensorFlow
RUN echo "tensorflow==999.0.0" > dont-install.txt

RUN apt-get update && apt-get install -y libgl1-mesa-glx ffmpeg && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --constraint dont-install.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
