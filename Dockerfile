# Используйте официальный образ Python
FROM python:3.11

# Установите рабочую директорию в контейнере
WORKDIR /app

# Скопируйте файлы зависимостей и установите их
# Это предполагает, что у вас есть файл requirements.txt в корне проекта
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Скопируйте исходный код вашего приложения в контейнер
COPY app.py ./
COPY src ./src

ENV HOST_IP=0.0.0.0
ENV HOST_PORT=64094

# Задайте команду для запуска приложения
CMD uvicorn app:app --host $HOST_IP --port $HOST_PORT
