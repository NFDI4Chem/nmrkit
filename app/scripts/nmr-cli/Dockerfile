# build the image ` docker build --tag nmr-cli  . ` 
# run the container ` docker run -it nmr-cli bash `

FROM mcr.microsoft.com/playwright:v1.56.1-noble

SHELL ["/bin/bash", "-o", "pipefail", "-c"]


WORKDIR /app

#ENV BASE_NMRIUM_URL=https://nmrium.nmrxiv.org/
ENV BASE_NMRIUM_URL=https://nmriumdev.nmrxiv.org/
ENV NMR_PREDICTION_URL=https://nmrshiftdb.nmr.uni-koeln.de/NmrshiftdbServlet/nmrshiftdbaction/quickcheck

COPY package.json ./

RUN npm install

COPY . ./

RUN npm run build

#install the nmr-cli as a global package 
# for example, nmr-cli parse-spectra -u https://cheminfo.github.io/bruker-data-test/data/zipped/aspirin-1h.zip
# nmr-cli predict -n "1H" --id 1 --type "nmr;1H;1d" --shifts "1" --solvent "Dimethylsulphoxide-D6 (DMSO-D6, C2D6SO)" -m $"\n Ketcher  6122516162D 1   1.00000     0.00000     0\n\n 16 17  0  0  0  0  0  0  0  0999 V2000\n    1.1954   -4.6484    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    2.9258   -4.6479    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    2.0622   -4.1483    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    2.9258   -5.6488    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    1.1954   -5.6533    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    2.0644   -6.1483    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    3.7902   -4.1495    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0\n    4.6574   -4.6498    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    3.7964   -6.1512    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0\n    4.6596   -5.6458    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    5.5228   -4.1488    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    5.5277   -6.1421    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n    6.3895   -4.6477    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n    5.5216   -3.1488    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n    7.2548   -4.1466    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    8.1215   -4.6455    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n  3  1  2  0  0  0  0\n  1  5  1  0  0  0  0\n  5  6  2  0  0  0  0\n  6  4  1  0  0  0  0\n  4  2  1  0  0  0  0\n  2  3  1  0  0  0  0\n  4  9  1  0  0  0  0\n  9 10  2  0  0  0  0\n 10  8  1  0  0  0  0\n  8  7  2  0  0  0  0\n  7  2  1  0  0  0  0\n  8 11  1  0  0  0  0\n 10 12  1  0  0  0  0\n 11 13  1  0  0  0  0\n 11 14  2  0  0  0  0\n 13 15  1  0  0  0  0\n 15 16  1  0  0  0  0\nM  END"
RUN npm install . -g



