# Riuà — full server: production cycle (radar nowcast, NWP, ensemble, hydrology) + API.
#   docker build -t riua .
#   docker run -p 8000:8000 -e RIUA_ROLE=all -v riua-state:/data riua
# RIUA_ROLE=api serves a product computed elsewhere (see render/Dockerfile for the slim image).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    RIUA_ROLE=all RIUA_STATE=/data/state RIUA_OUT=/data/out RIUA_CYCLE_MIN=10 MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend backend
COPY geo/out geo/out
COPY geo/hydro geo/hydro

WORKDIR /app/backend
VOLUME /data
EXPOSE 8000
CMD ["sh", "-c", "uvicorn riua.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
