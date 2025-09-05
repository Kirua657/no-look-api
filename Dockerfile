FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8080
# genai/main.py の FastAPI を起動
CMD ["uvicorn", "genai.main:app", "--host", "0.0.0.0", "--port", "8080"]
