# Installing GAUGE

You do not need any programming experience to install or run GAUGE — follow
whichever option below matches your setup.

## Option A — Docker (recommended, works the same on Mac/Linux/Windows)

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (one-time).
2. From a terminal in this folder:
   ```bash
   docker build -t gauge .
   docker run -p 8501:8501 gauge
   ```
3. Open http://localhost:8501 in your browser.

## Option B — Conda environment

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (one-time).
2. From a terminal in this folder:
   ```bash
   conda env create -f environment.yml
   conda activate gauge
   ./run_gauge.sh        # macOS/Linux
   run_gauge.bat         # Windows (double-click also works)
   ```
3. Open http://localhost:8501 in your browser. A browser tab usually opens automatically.

## Option C — plain pip (advanced users)

```bash
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
./run_gauge.sh
```

## Verifying the install

```bash
pytest tests/ -q
```
All tests should pass (they load the real model bundles and run real
predictions, so this also confirms the model files are intact).

## Troubleshooting

- **Port already in use**: set `GAUGE_PORT=8888` before running the launcher.
- **`ImportError ... CXXABI_1.3.15 ... libicui18n`**: this means the app was
  started without going through `run_gauge.sh`/`run_gauge.bat` (which set
  the environment correctly) — re-run via the launcher script, or via Docker.
- **Slow first prediction**: the first prediction after launch loads and
  caches the model bundle (a few seconds); subsequent predictions are fast.
