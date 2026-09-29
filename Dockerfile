FROM python:3.12-slim
WORKDIR /app
COPY requirements-full.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-full.txt
COPY . .
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn datacourt.main:app --host 0.0.0.0 --port ${PORT}"]
