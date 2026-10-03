FROM python:3.11-slim

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: train the model. Override CMD to run predict.py.
CMD ["python", "train.py", "--data", "data/train.csv", "--output", "model.pkl"]
