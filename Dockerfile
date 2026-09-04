FROM python:3.11-slim@sha256:9c900dea9e8fb7e16277c179b555cc72d29a352dbc33cff48ad5a0412fd5bfc7

ENV PIP_DEFAULT_TIMEOUT=180 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
# Swap the full xgboost wheel for the official CPU-only build: identical
# Python API (import xgboost), avoids pulling nvidia-nccl-cu12 (~342MB)
# that CPU inference never uses.
RUN grep -vE '^(xgboost==|streamlit|shap|plotly)' requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt \
    && pip install --no-cache-dir xgboost-cpu==3.2.0

COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser models/ models/
COPY --chown=appuser:appuser dashboard/ dashboard/

RUN useradd --system --uid 10001 --create-home appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app/data
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4); sys.exit(0 if r.status==200 else 1)"

CMD ["uvicorn", "src.score_service:app", "--host", "0.0.0.0", "--port", "8000"]
