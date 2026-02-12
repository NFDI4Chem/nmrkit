FROM continuumio/miniconda3:24.1.2-0 AS nmrkit-ms

ENV PYTHON_VERSION=3.10
ENV OPENBABEL_VERSION=v3.1

ARG RELEASE_VERSION
ENV RELEASE_VERSION=${RELEASE_VERSION}

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    openjdk-17-jre \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    conda update -n base -c defaults conda

RUN apt-get update && apt-get -y install docker.io

RUN conda install -c conda-forge python>=PYTHON_VERSION
RUN conda install -c conda-forge openbabel>=OPENBABEL_VERSION

RUN pip3 install rdkit

RUN python3 -m pip install -U pip

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64/
RUN export JAVA_HOME

RUN git clone "https://github.com/rinikerlab/lightweight-registration.git" lwreg
RUN chmod +x lwreg
RUN pip3 install --editable ./lwreg/.

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
COPY ./alembic.ini /code/alembic.ini

RUN pip3 install --upgrade setuptools pip && \
    apt-get update && apt-get install -y git

RUN pip3 install --no-cache-dir -r /code/requirements.txt

RUN python3 -m pip uninstall -y uvicorn

RUN python3 -m pip install uvicorn[standard]

COPY ./app /code/app

RUN curl -sL https://deb.nodesource.com/setup_current.x | bash -
RUN apt-get install -y nodejs
RUN npm install -g npm@latest

RUN npm install -g /code/app/scripts/nmr-cli

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "4", "--reload"]