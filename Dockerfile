FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home appuser

COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser models/house_price_model.joblib ./models/house_price_model.joblib
COPY --chown=appuser:appuser models/input_schema.json ./models/input_schema.json
COPY --chown=appuser:appuser data/evaluation/model_comparison.csv ./data/evaluation/model_comparison.csv
COPY --chown=appuser:appuser data/interpretation/random_forest_importances.csv ./data/interpretation/random_forest_importances.csv

USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]