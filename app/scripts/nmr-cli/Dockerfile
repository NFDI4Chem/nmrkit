# build the image ` docker build --tag nmr-cli  . ` 
# run the container ` docker run -it nmr-cli bash `

FROM mcr.microsoft.com/playwright:v1.40.0-jammy

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /app

#ENV BASE_NMRIUM_URL=https://nmrium.nmrxiv.org/
ENV BASE_NMRIUM_URL=https://nmriumdev.nmrxiv.org/


COPY package.json ./
COPY package-lock.json ./

RUN npm install

COPY . ./

#install the nmr-cli as a global package 
# for example, nmr-cli -u https://cheminfo.github.io/bruker-data-test/data/zipped/aspirin-1h.zip
RUN npm install . -g



