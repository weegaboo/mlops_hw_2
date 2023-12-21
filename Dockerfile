FROM python:3.11

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY src ./src

ENV HOST_IP=0.0.0.0
ENV HOST_PORT=64094

CMD uvicorn app:app --host $HOST_IP --port $HOST_PORT
