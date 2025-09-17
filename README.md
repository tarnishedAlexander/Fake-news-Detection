# Fake News Detection
## Prerequisites

### Docker and Docker Compose installed
- Python 3.8+
- Jupyter Notebook

### Setup and Run Instructions

- Build the Docker image:
```bash
docker compose build
```

- Start the Docker container:

```bash
docker compose up -d
```

- Access the container:
```bash
docker compose exec -it fakeNews /bin/bash
```

- Install dependencies:
```bash
pip install torch transformers pandas numpy scikit-learn tqdm matplotlib seaborn
```

Run the main script:

```bash
python main.py
```

- Start Jupyter Notebook (optional, for analysis):

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```


Notes:

Access Jupyter Notebook via the provided URL after starting.
Ensure port 8888 is open if accessing remotely.
