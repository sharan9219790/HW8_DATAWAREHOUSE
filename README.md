# HW8 – Airflow + Pinecone Pipeline

## Prereqs
- Docker Desktop
- Pinecone account + API key (serverless project)

## Setup
1. In the Airflow UI (http://localhost:8080), create Variables:
   - `PINECONE_API_KEY` = your_key
   - `PINECONE_INDEX_NAME` = `semantic-search-fast`
   - `PINECONE_CLOUD` = `aws`
   - `PINECONE_REGION` = `us-west-2`

2. Start services:
   ```bash
   cd /Users/spartan/Downloads/DATA226/HW8
   docker compose up airflow-init
   docker compose up -d

