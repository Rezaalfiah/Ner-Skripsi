FROM python:3.10-slim

# Konfigurasi runtime Python dan port aplikasi di dalam container.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    OMP_NUM_THREADS=2 \
    TF_NUM_INTRAOP_THREADS=2 \
    TF_NUM_INTEROP_THREADS=1 \
    WEB_WORKERS=1 \
    WEB_THREADS=2 \
    WEB_TIMEOUT=180 \
    STATIC_CACHE_SECONDS=86400 \
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

# Healthcheck hanya membuka homepage sehingga tidak memuat model machine learning.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '7860') + '/', timeout=3)"

# Satu worker mencegah salinan model memenuhi RAM; threads tetap memberi konkurensi.
# Seluruh nilai dapat dioverride melalui environment variable sesuai kapasitas VPS.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers ${WEB_WORKERS} --threads ${WEB_THREADS} --timeout ${WEB_TIMEOUT} --access-logfile - app:app"]
