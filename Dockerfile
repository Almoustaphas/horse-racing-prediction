FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Europe/Paris

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scraper/ scraper/
COPY prediction/ prediction/
COPY api/ api/

# Base de données et modèles montés en volumes (voir docker-compose.yml)
RUN useradd --create-home appuser && mkdir -p database models && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
