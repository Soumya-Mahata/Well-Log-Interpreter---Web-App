FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Streamlit config for Hugging Face
RUN mkdir -p /root/.streamlit && \
    echo "[server]\n\
headless = true\n\
port = 7860\n\
address = 0.0.0.0\n\
enableCORS = false\n\
maxUploadSize = 200\n\
" > /root/.streamlit/config.toml

EXPOSE 7860

CMD ["streamlit", "run", "main.py", "--server.port=7860", "--server.address=0.0.0.0"]
