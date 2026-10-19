# Image with the Python dependencies only. The repo itself is bind-mounted at /app by
# docker-compose, so code edits are live and outputs/caches stay on the host.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

WORKDIR /app

# lion_linker requires torch; the CPU build keeps the image ~1.5 GB instead of ~6 GB with CUDA.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["python", "-m", "streamlit", "run", "ui/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
