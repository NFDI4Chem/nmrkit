# Versions

## API versioning

NMRKit uses URL-based versioning via `fastapi-versioning`:

| Prefix | Description |
|--------|-------------|
| `/latest/` | Alias for the current API version (recommended for development) |
| `/v1/` | Stable version 1 |

All module paths are appended to the version prefix. For example:

```
GET  /latest/chem/hosecode?smiles=CCO
POST /latest/spectra/parse/url
POST /latest/predict/
```

## Docker image tags

Images are published to [Docker Hub](https://hub.docker.com/r/nfdi4chem/nmrkit):

| Tag | Environment |
|-----|-------------|
| `dev-latest` | Development deployments (`dev.nmrkit.nmrxiv.org`) |
| `latest` | Production deployments (`nmrkit.nmrxiv.org`) |

Companion images:

| Image | Dev tag | Prod tag |
|-------|---------|----------|
| `nfdi4chem/nmr-cli` | `dev-latest` | `latest` |
| `nfdi4chem/nmr-respredict` | `dev-latest` | `latest` |

## Changelog

Release history is tracked in the repository [CHANGELOG.md](https://github.com/NFDI4Chem/nmrkit/blob/main/CHANGELOG.md).
