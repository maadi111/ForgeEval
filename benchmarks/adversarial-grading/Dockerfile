FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY workspace/ /workspace/
RUN pip install --no-cache-dir -r requirements.txt

CMD ["pytest", "-q"]
