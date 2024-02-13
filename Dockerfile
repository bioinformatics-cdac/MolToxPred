FROM python:3.11.8-slim-bullseye
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir  -r /app/requirements.txt 
RUN apt update && apt install -y default-jre libgomp1 && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY . /app/
RUN chmod +x /app/moltox
ENV PATH=/app:$PATH
# WORKDIR /app
