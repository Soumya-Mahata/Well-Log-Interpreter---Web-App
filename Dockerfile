FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (important for numpy, scipy, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Streamlit config (CRITICAL for HF)
RUN mkdir -p /root/.streamlit && \
    echo "\
[server]\n\
headless = true\n\
port = 7860\n\
enableCORS = false\n\
maxUploadSize = 200\n\
" > /root/.streamlit/config.toml

EXPOSE 7860

CMD ["streamlit", "run", "main.py"]
