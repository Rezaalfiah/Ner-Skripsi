FROM python:3.10-slim

# Konfigurasi runtime Python dan port aplikasi di dalam container.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=7860

WORKDIR /app

# Library sistem yang dibutuhkan oleh dependensi machine learning.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Dependensi dipasang lebih dulu agar layer Docker dapat digunakan kembali.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY models ./models
COPY templates ./templates
COPY static ./static

EXPOSE 7860

# Gunicorn menjalankan aplikasi Flask melalui objek `app` di app.py.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "2", "--timeout", "180", "app:app"]
