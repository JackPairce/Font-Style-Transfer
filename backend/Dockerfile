FROM python:3.12-slim

RUN apt-get update && apt-get install -y make libgl1 libglib2.0-0

WORKDIR /app

COPY . .

RUN pip install --upgrade pip

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir pandas numpy flask flask_cors opencv-python pillow

CMD ["make", "run"]
