FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE $PORT

CMD ["sh", "-c", "streamlit run app.py --server.port $PORT --server.address 0.0.0.0"]
