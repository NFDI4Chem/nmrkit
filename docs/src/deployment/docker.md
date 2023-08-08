# Docker
NMRKit is containerized using Docker and is distributed publicly via the [Docker Hub](https://hub.docker.com/r/nfdi4chem/nmrkit), a cloud-based registry provided by Docker that allows developers to store, share, and distribute Docker images.

To use this image:

* Make sure you have [Docker](https://docs.docker.com/get-docker/) installed and configured on your target deployment environment.
* Pull the NMRKit image by providing the appropiate tag.

```bash
docker pull nfdi4chem/nmrkit:[tag]

```
* NMRKit uses [rdkit-cartridge-debian](https://hub.docker.com/r/informaticsmatters/rdkit-cartridge-debian) Postgres. Run the below command to spin up the image.

```bash
docker run -d --name postgres -e POSTGRES_PASSWORD=password -e POSTGRES_USER=nmrkit -p 5432:5432 informaticsmatters/rdkit-cartridge-debian:latest

```

* After the Postgres container is prepared to receive connections, initiate the NMRKit container by executing the following command and providing the Postgres credentials.

```bash
docker run -d -p 8080:80 --name nmrkit -e POSTGRES_PASSWORD=password -e POSTGRES_USER=nmrkit -e POSTGRES_SERVER=postgres -e POSTGRES_DB=nmr_predict nfdi4chem/nmrkit:[tag]

```