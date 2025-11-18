from datetime import datetime, timedelta
import os
import logging
from typing import List, Tuple

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable

import pandas as pd
import requests

# ===== Paths =====
DATA_DIR = "/opt/airflow/data"
RAW_CSV = os.path.join(DATA_DIR, "medium_data.csv")
PROC_PARQUET = os.path.join(DATA_DIR, "medium_processed.parquet")

# Slide dataset URL
MEDIUM_URL = "https://s3-geospatial.s3.us-west-2.amazonaws.com/medium_data.csv"

# ===== Task functions =====
def download_csv(**context):
    os.makedirs(DATA_DIR, exist_ok=True)
    r = requests.get(MEDIUM_URL, timeout=60)
    r.raise_for_status()
    with open(RAW_CSV, "wb") as f:
        f.write(r.content)
    logging.info(f"Downloaded: {RAW_CSV}")

def preprocess(**context):
    df = pd.read_csv(RAW_CSV)

    # keep minimal columns; ensure an id
    base_cols = [c for c in ["id", "title", "subtitle"] if c in df.columns]
    if "title" not in base_cols:
        raise ValueError("Input CSV must contain a 'title' column.")
    df = df[base_cols].copy()

    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})

    df["subtitle"] = df.get("subtitle", pd.Series([""] * len(df))).fillna("")
    df = df.dropna(subset=["title"])
    df["metadata"] = df.apply(
        lambda row: {"title": (str(row["title"]) + " " + str(row["subtitle"])).strip()},
        axis=1,
    )
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    df.to_parquet(PROC_PARQUET, index=False)
    logging.info(f"Processed rows: {len(df)} -> {PROC_PARQUET}")

def create_index(**context):
    from pinecone import Pinecone, ServerlessSpec
    from pinecone.core.client.exceptions import PineconeApiException

    api_key = Variable.get("PINECONE_API_KEY")
    index_name = Variable.get("PINECONE_INDEX_NAME", default_var="semantic-search-fast")

    # Try the user-specified cloud/region first
    primary_cloud = Variable.get("PINECONE_CLOUD", default_var="aws")
    primary_region = Variable.get("PINECONE_REGION", default_var="us-west-2")

    # Fallbacks commonly available on free tier
    fallbacks: List[Tuple[str, str]] = [
        (primary_cloud, primary_region),   # try what user set first
        ("aws", "us-east-1"),
        ("gcp", "us-central1"),
    ]

    pc = Pinecone(api_key=api_key)

    # if already exists anywhere accessible, skip create
    existing = [idx["name"] for idx in pc.list_indexes()]
    if index_name in existing:
        logging.info(f"Index already exists: {index_name}")
        return

    last_err = None
    for cloud, region in fallbacks:
        try:
            logging.info(f"Attempting to create index '{index_name}' in {cloud}/{region} ...")
            pc.create_index(
                name=index_name,
                dimension=384,  # all-MiniLM-L6-v2 output size
                metric="dotproduct",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            logging.info(f"Created index '{index_name}' in {cloud}/{region}")
            return
        except PineconeApiException as e:
            msg = getattr(e, "body", str(e))
            logging.warning(f"Create index failed in {cloud}/{region}: {msg}")
            last_err = e
        except Exception as e:  # catch-all just in case
            logging.warning(f"Unexpected error creating index in {cloud}/{region}: {e}")
            last_err = e

    # If all attempts failed, raise the last error
    raise last_err if last_err else RuntimeError("Unable to create index in any region tried.")

def embed_and_upsert(**context):
    from sentence_transformers import SentenceTransformer
    from pinecone import Pinecone

    api_key = Variable.get("PINECONE_API_KEY")
    index_name = Variable.get("PINECONE_INDEX_NAME", default_var="semantic-search-fast")

    df = pd.read_parquet(PROC_PARQUET)
    df["id"] = df["id"].astype(str)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    df["values"] = df["metadata"].apply(lambda x: model.encode(x["title"]).tolist())

    df_upsert = df[["id", "values", "metadata"]].copy()

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    index.upsert_from_dataframe(df_upsert)
    logging.info(f"Upserted {len(df_upsert)} vectors into {index_name}")

def test_query(**context):
    from sentence_transformers import SentenceTransformer
    from pinecone import Pinecone

    api_key = Variable.get("PINECONE_API_KEY")
    index_name = Variable.get("PINECONE_INDEX_NAME", default_var="semantic-search-fast")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_vec = model.encode("what is ethics in AI").tolist()

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    res = index.query(
        vector=query_vec,
        top_k=10,
        include_metadata=True,
        include_values=False,
    )

    for i, m in enumerate(res["matches"], 1):
        title = (m.get("metadata") or {}).get("title", "")
        logging.info(f"{i}. score={m['score']:.4f} title={title}")

# ===== DAG =====
default_args = {
    "owner": "spartan",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="pinecone_medium_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["pinecone", "semantic-search"],
) as dag:
    t1 = PythonOperator(task_id="download_csv", python_callable=download_csv)
    t2 = PythonOperator(task_id="preprocess", python_callable=preprocess)
    t3 = PythonOperator(task_id="create_index", python_callable=create_index)
    t4 = PythonOperator(task_id="embed_and_upsert", python_callable=embed_and_upsert)
    t5 = PythonOperator(task_id="test_query", python_callable=test_query)

    t1 >> t2 >> t3 >> t4 >> t5

