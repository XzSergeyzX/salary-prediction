# База: легкий питон
FROM python:3.10-slim

# Установим рабочую папку
WORKDIR /app

# Скопируем файлы
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Добавим остальной код
COPY . .

# Откроем порт, на котором будет крутиться FastAPI
EXPOSE 8000

# Запускаем FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
