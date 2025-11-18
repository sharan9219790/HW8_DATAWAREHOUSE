# Build a custom Airflow image with all deps preinstalled.
FROM apache/airflow:2.9.1

# Prevent pip from writing bytecode; speed up installs a bit
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=180

# Install required libs for your DAGs
# Pin lightweight, compatible versions where possible
RUN pip install --no-cache-dir \
    sentence-transformers==2.7.0 \
    pinecone==4.1.0 \
    pandas \
    requests \
    pyarrow
