FROM python:3.13-slim
WORKDIR /app
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt
COPY . .
RUN mkdir -p /app/data/gmc /app/data/lc /app/models
EXPOSE 8000
CMD ["sh", "-c", "python -m src.data_fusion && python -m src.train && python -m src.null_impact && uvicorn api.main:app --host 0.0.0.0 --port 8000"]