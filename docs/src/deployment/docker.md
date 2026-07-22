# Docker

NMRKit is containerized using Docker and distributed via
[Docker Hub](https://hub.docker.com/r/nfdi4chem/nmrkit).

## Pull the image

```bash
docker pull nfdi4chem/nmrkit:latest        # production
docker pull nfdi4chem/nmrkit:dev-latest    # development
```

## Docker Compose (recommended)

The root `docker-compose.yml` starts the full stack:

```bash
cp env.template .env
docker compose up -d
```

| Service | Container | Port | Purpose |
|---------|-----------|------|---------|
| `web` | `nmrkit-api` | 8080→80 | FastAPI application |
| `nmr-load-save` | `nmr-converter` | — | nmr-cli spectra parsing & prediction |
| `nmr-respredict` | `nmr-respredict` | — | Residual NMR prediction (future) |
| `pgsql` | — | 5433→5432 | PostgreSQL + RDKit cartridge |
| `redis` | — | 6380→6379 | Cache |
| `minio` | — | 9002, 8901 | Object storage |
| `prometheus` | `nmrkit_prometheus` | 9090 | Metrics |
| `grafana` | `nmrkit_grafana` | 3000 | Dashboards |

Open http://localhost:8080/latest/docs for the Scalar API reference.

### Volumes

- `./app` is mounted into the API container for live code reload during development
- `/var/run/docker.sock` allows the API to `docker exec` into `nmr-converter`
- `shared-data` passes files between API, nmr-cli, and nmr-respredict containers

## Standalone container

For a minimal deployment with an external PostgreSQL instance:

```bash
# Start PostgreSQL with RDKit cartridge
docker run -d --name postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_USER=nmrkit \
  -p 5432:5432 \
  informaticsmatters/rdkit-cartridge-debian:latest

# Start NMRKit API
docker run -d -p 8080:80 --name nmrkit \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_USER=nmrkit \
  -e POSTGRES_SERVER=postgres \
  -e POSTGRES_DB=nmr_predict \
  --link postgres:postgres \
  nfdi4chem/nmrkit:latest
```

::: warning
Standalone mode does not include `nmr-converter`. Spectra parsing and prediction
endpoints require the nmr-cli container — use Docker Compose instead.
:::

## Server deployments (Traefik)

For hosted dev/prod environments, use the ops compose files:

```bash
# Development
docker compose -f ops/docker-compose-dev.yml up -d

# Production
docker compose -f ops/docker-compose-prod.yml up -d
```

These configurations use [Traefik](https://traefik.io/) as a reverse proxy:

| Environment | Host | Image tag |
|-------------|------|-----------|
| Development | `dev.nmrkit.nmrxiv.org` | `nfdi4chem/nmrkit:dev-latest` |
| Production | `nmrkit.nmrxiv.org` | `nfdi4chem/nmrkit:latest` |

## Health checks

Docker Compose health checks hit the registration module:

```
GET http://localhost:80/latest/registration/health
```

## See also

- [Deployment Overview](./overview) — all deployment options
- [nmr-cli service](../services/nmr-cli) — companion container details
- [Local Installation](../development/installation) — development setup
