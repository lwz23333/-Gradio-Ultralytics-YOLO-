FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10 \
    PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple \
    PIP_TRUSTED_HOST=mirrors.cloud.tencent.com \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 \
    KMP_DUPLICATE_LIB_OK=TRUE

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

COPY app_server.py /app/app_server.py

RUN mkdir -p /app/models /app/outputs

EXPOSE 7860

CMD ["python", "app_server.py"]
