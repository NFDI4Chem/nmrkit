# build the image ` docker build --tag nmr-cli  . ` 
# run the container ` docker run -it nmr-cli bash `

FROM node:20.9.0-alpine
# a temporary command to add bash to the Docker image to be able to run the container with an interactive Shell for testing purpose
RUN apk update && apk add bash
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /app

COPY package.json ./
COPY package-lock.json ./
RUN npm install

COPY . ./

#install the nmr-cli as a global package 
# for example, nmr-cli -u https://cheminfo.github.io/bruker-data-test/data/zipped/aspirin-1h.zip
RUN npm install . -g



