FROM continuumio/miniconda3 AS nmr-respredict-ms

ENV PYTHON_VERSION=3.10
ENV RDKIT_VERSION=2023.09.1

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    apt-get update -y && \
    apt-get install -y curl && \
    conda update -n base -c defaults conda

RUN conda install -c conda-forge python>=PYTHON_VERSION
# RUN conda install -c conda-forge rdkit>=RDKIT_VERSION
RUN python3 -m pip install -U pip

RUN pip3 install rdkit

RUN conda install -y scikit-learn && \
    conda install -y numba && \
    conda install -y numpy && \
    conda install -y tqdm && \
    conda install -y networkx && \
    conda install -y click && \
    conda install -y pandas && \
    conda install -y pyarrow && \
    conda install -y pytorch

COPY . /nmr-respredict
WORKDIR /nmr-respredict