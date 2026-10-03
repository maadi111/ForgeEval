FROM python:3.11-slim

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-c", "from retriever import DenseRetriever; r = DenseRetriever(); print('Retriever ready')"]
