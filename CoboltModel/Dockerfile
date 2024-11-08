FROM continuumio/miniconda3

WORKDIR /CoboltModel

COPY environment.yml .
# COPY . . 
RUN conda env create -f environment.yml 
SHELL ["conda", "run", "-n", "cobolt_env", "/bin/bash", "-c"]

COPY  cobolt_main.py .

# VOLUME ["/data"]

# Run the main Python file within the Conda environment
CMD ["conda", "run", "-n", "cobolt_env", "python", "-u", "cobolt_main.py"]