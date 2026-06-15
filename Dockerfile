FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps kept minimal; PyMuPDF ships its own native bits.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

# Run as a non-root user.
RUN useradd --create-home --shell /bin/bash botuser \
    && chown -R botuser:botuser /app
USER botuser

CMD ["python", "-u", "main.py"]
