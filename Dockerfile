# CoreRec: interactions file in, recommendation API out.
#
#   docker build -t corerec .
#   docker run -p 8000:8000 -v "$PWD:/data" corerec /data/events.csv
#   curl -X POST localhost:8000/recommend -H 'Content-Type: application/json' \
#        -d '{"user_id": "u0042", "top_k": 5}'
#
# Any `corerec serve` flag works after the path, e.g. `--model EASE`.
# With no arguments it serves the bundled demo file.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only torch by default: the CUDA wheels add several GB the server does not
# need. For a GPU image: --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install torch --index-url "$TORCH_INDEX"

COPY pyproject.toml README.md ./
COPY corerec ./corerec
RUN pip install ".[serving]"

COPY sample_data/events.csv /app/sample_data/events.csv

RUN useradd --create-home corerec
USER corerec

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

ENTRYPOINT ["corerec", "serve", "--host", "0.0.0.0", "--port", "8000"]
CMD ["/app/sample_data/events.csv"]
