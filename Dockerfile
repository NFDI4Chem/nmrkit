FROM debian:buster-20230227-slim AS nmr-predict-ms

ARG RELEASE_VERSION
ENV RELEASE_VERSION=${RELEASE_VERSION}

# Install runtime dependencies
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    apt-get update -y && \
    apt-get install -y openjdk-11-jre

RUN python3 -m pip install -U pip

ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/
RUN export JAVA_HOME

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt

RUN pip3 install --upgrade setuptools pip && \
    apt-get update && apt-get install -y git

RUN pip3 install --no-cache-dir -r /code/requirements.txt

RUN python3 -m pip uninstall -y uvicorn

RUN python3 -m pip install uvicorn[standard]

RUN pip3 install --no-cache-dir chembl_structure_pipeline --no-deps

COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]