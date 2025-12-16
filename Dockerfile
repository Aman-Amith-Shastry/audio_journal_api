FROM python:3.10-slim

# ---- System dependencies ----
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Set working directory ----
WORKDIR /app

# ---- Copy dependencies first (Docker cache optimization) ----
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy application code ----
COPY . .

# ---- Expose FastAPI port ----
EXPOSE 8000

# ---- Run server ----
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]