FROM python:3.11-slim

# System deps 
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs as a non-root user (UID 1000) 

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# SET WORKDIR
WORKDIR /home/user/app 

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY . .

# Port (HF Spaces expects 7860)
ENV PORT=7860 \
    PYTHONPATH=/home/user/app
EXPOSE 7860

# Switch to the non-root user for security
USER user

# Health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/ping || exit 1

# Entrypoint
CMD ["python", "-m", "server.app"]