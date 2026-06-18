FROM python:3.10-slim

# RDKit's pip wheel bundles its own native dependencies; libxrender/libxext
# are still needed by some RDKit drawing code paths even when unused here.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gauge_core ./gauge_core
COPY app ./app
COPY models ./models
COPY example_data ./example_data
COPY tests ./tests
COPY kg_types.py ./
COPY pyproject.toml LICENSE CITATION.cff README.md .env.example ./
# .env (with real secrets) is intentionally NOT copied into the image -- pass
# it at run time instead: `docker run --env-file .env ...`

EXPOSE 8501
HEALTHCHECK CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["python3", "-m", "streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
