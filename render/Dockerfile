# Slim API image for a small instance (Render free: 0.1 CPU, 512 MB).
# It serves the product that the scheduled workers compute; no scientific stack inside.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RIUA_ROLE=api
WORKDIR /app/backend
RUN pip install --no-cache-dir "fastapi==0.115.*" "uvicorn==0.32.*" "numpy==2.1.*"
COPY backend/riua/__init__.py riua/__init__.py
COPY backend/riua/api.py riua/api.py
EXPOSE 10000
CMD ["sh", "-c", "uvicorn riua.api:app --host 0.0.0.0 --port ${PORT:-10000}"]
