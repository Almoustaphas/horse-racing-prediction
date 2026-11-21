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
COPY deploy/ deploy/

# Données : volumes (docker-compose) ou disque persistant (Render, DATA_DIR=/app/data)
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p database models data \
    && chown -R appuser /app \
    && install -m 0755 deploy/hrp /usr/local/bin/hrp \
    && mkdir -p -m 0700 /root/.ssh

# L'image démarre en root ; entrypoint.sh abandonne les droits root (utilisateur appuser)
ENTRYPOINT ["/app/deploy/entrypoint.sh"]
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
