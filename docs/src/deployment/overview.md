# Deployment Overview

NMRKit supports three deployment modes:

| Mode | Use case | Documentation |
|------|----------|---------------|
| **Docker Compose (local)** | Development and testing | [Local Installation](../development/installation#docker) |
| **Docker Compose (ops)** | Dev / prod servers with Traefik | [Docker](./docker) |
| **Kubernetes (Helm)** | Cluster deployments | [Cluster Deployment](./cluster-deployment) |

## Public instances

| Environment | API docs | Host |
|-------------|----------|------|
| Development | [dev.nmrkit.nmrxiv.org/latest/docs](https://dev.nmrkit.nmrxiv.org/latest/docs) | `dev.nmrkit.nmrxiv.org` |
| Production | [nmrkit.nmrxiv.org/latest/docs](https://nmrkit.nmrxiv.org/latest/docs) | `nmrkit.nmrxiv.org` |

## Compose files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Full local stack (API, nmr-cli, nmr-respredict, DB, Redis, MinIO, monitoring) |
| `ops/docker-compose-dev.yml` | Dev server with Traefik reverse proxy |
| `ops/docker-compose-prod.yml` | Production server with Traefik |

## Required services

At minimum, the following containers must be running for full API functionality:

| Service | Required for |
|---------|-------------|
| `nmrkit-api` | All API endpoints |
| `nmr-converter` | Spectra, converter, and predict modules |
| `pgsql` | Registration module |
| `nmr-respredict` | Future respredict integration (optional today) |

## Environment configuration

Copy `env.template` to `.env` and set database credentials:

```bash
cp env.template .env
```

| Variable | Description |
|----------|-------------|
| `POSTGRES_USER` | PostgreSQL username |
| `POSTGRES_PASSWORD` | PostgreSQL password |
| `POSTGRES_SERVER` | Database host (`pgsql` in Compose) |
| `POSTGRES_PORT` | Database port (default `5432`) |
| `POSTGRES_DB` | Database name (`nmr_predict`) |
| `MINIO_ROOT_USER` | MinIO access key |
| `MINIO_ROOT_PASSWORD` | MinIO secret key |

## CI/CD

Docker images are built and published via GitHub Actions:

- `dev-build.yml` — builds `nfdi4chem/nmrkit:dev-latest` on pushes to `development`
- `prod-build.yml` — builds `nfdi4chem/nmrkit:latest` on releases
- `doc-deploy.yml` — deploys this documentation site to GitHub Pages

See [#95 — Update CI/CD and deployment script](https://github.com/NFDI4Chem/nmrkit/issues/95)
for ongoing improvements to the deployment pipeline.

## Quick start (local)

```bash
git clone https://github.com/NFDI4Chem/nmrkit.git
cd nmrkit
cp env.template .env
docker compose up -d
```

API docs: http://localhost:8080/latest/docs
