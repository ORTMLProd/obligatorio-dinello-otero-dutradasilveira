# syntax=docker/dockerfile:1
# Minimal MLflow tracking server (Electivo 1). Self-contained image with the same
# MLflow version we train against, so the UI reads our runs without tag-guessing a
# published image. sqlite backend store + local artifact root both live under /mlflow,
# a named volume, so runs and the model registry survive container restarts.
FROM python:3.12-slim-bookworm

RUN pip install --no-cache-dir mlflow==3.13.0

EXPOSE 5000

# --serve-artifacts + --artifacts-destination makes the server PROXY artifact upload/download
# over HTTP (artifact URIs become mlflow-artifacts:/...). Without this the client would try to
# write to the server's local artifact path on its own filesystem and fail.
# --allowed-hosts: MLflow 3 rejects requests whose Host header isn't localhost/private-IP by
# default (anti DNS-rebinding). Add localhost (local docker-compose) and the EB domain pattern
# (deploy) explicitly, or the UI 403s when accessed via the public *.elasticbeanstalk.com URL.
# The Host header includes the port when it's non-default (e.g. "...elasticbeanstalk.com:5500"),
# and the wildcard match doesn't strip it — so both the bare and ":5500" patterns are needed.
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--serve-artifacts", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--allowed-hosts", "localhost,127.0.0.1,localhost:5500,127.0.0.1:5500,*.elasticbeanstalk.com,*.elasticbeanstalk.com:5500"]
