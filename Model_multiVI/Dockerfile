FROM nvidia/cuda:11.8.0-base-ubuntu22.04
FROM python:3.12 AS base

# Install Python 3.11 and pip
RUN apt-get update && apt-get install -y python3-pip

WORKDIR /multivi

# Copy the entire mowgli/ directory into the container
COPY scvi/ /multivi/
COPY main.py /multivi/
COPY requirements.txt /multivi/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Mount data
VOLUME /multivi/data_TEA

CMD ["/bin/bash"]