FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Install Python 3.11 and pip
RUN apt-get update && apt-get install -y python3.10 python3-pip

WORKDIR /mowgli

# Copy the entire mowgli/ directory into the container
COPY mowgli/*.py /mowgli/
COPY main.py /mowgli/
COPY requirements.txt /mowgli/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Mount data
VOLUME /mowgli/data_TEA

EXPOSE 8000

# CMD ["python3", "main.py"]