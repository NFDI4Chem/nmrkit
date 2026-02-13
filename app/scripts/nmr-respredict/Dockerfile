FROM continuumio/miniconda3:24.1.2-0 AS nmr-respredict-ms

ENV PYTHON_VERSION=3.10

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    conda update -n base -c defaults conda

RUN conda install -c conda-forge python>=PYTHON_VERSION
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