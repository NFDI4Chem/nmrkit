FROM continuumio/miniconda3 AS nmrkit-ms

ENV PYTHON_VERSION=3.10
ENV RDKIT_VERSION=2023.03.1
ENV OPENBABEL_VERSION=v3.1

ARG RELEASE_VERSION
ENV RELEASE_VERSION=${RELEASE_VERSION}

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    apt-get update -y && \
    apt-get install -y openjdk-11-jre && \
    apt-get install -y curl && \
    conda update -n base -c defaults conda

RUN conda install -c conda-forge python>=PYTHON_VERSION
RUN conda install -c conda-forge rdkit>=RDKIT_VERSION
RUN conda install -c conda-forge openbabel>=OPENBABEL_VERSION

RUN python3 -m pip install -U pip

ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/
RUN export JAVA_HOME

RUN git clone "https://github.com/rinikerlab/lightweight-registration.git" lwreg
RUN chmod +x lwreg
RUN pip3 install --editable ./lwreg/.

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
COPY ./alembic.ini /code/alembic.ini

RUN pip3 install --upgrade setuptools pip
RUN pip3 install --no-cache-dir -r /code/requirements.txt

RUN python3 -m pip uninstall -y uvicorn

RUN python3 -m pip install uvicorn[standard]

COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "4", "--reload"]