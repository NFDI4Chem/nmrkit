# nmr-respredict Service

The **nmr-respredict** container provides residual-based NMR chemical shift
prediction. It is distributed as a separate Docker image and is included in the
NMRKit Docker Compose stack for future integration.

## Docker image

| Registry | Image | Tags |
|----------|-------|------|
| Docker Hub | `nfdi4chem/nmr-respredict` | `dev-latest`, `latest` |

Source: [stefhk3/nmr-respredict-docker](https://github.com/stefhk3/nmr-respredict-docker)

## Role in NMRKit

In the current deployment, `nmr-respredict` runs alongside the API and shares
the `/shared` volume. The FastAPI application does not yet expose a dedicated
REST endpoint for this engine — prediction is currently routed through
**nmr-cli** using the `nmrdb.org` and `nmrshift` engines.

Planned integration is tracked in
[#62 — Integrate nmr-respredict Docker image](https://github.com/NFDI4Chem/nmrkit/issues/62).

## Compose configuration

```yaml
nmr-respredict:
  image: nfdi4chem/nmr-respredict:dev-latest
  container_name: nmr-respredict
  entrypoint: /bin/sh
  stdin_open: true
  tty: true
  volumes:
    - shared-data:/shared
```

## Requirements

- The container must be running for future respredict endpoints to work
- MOL/SDF structure files are passed via the shared volume at `/shared`
- PostgreSQL with RDKit cartridge is required for HOSE-code-based lookup

## Related documentation

- [Prediction module](../modules/prediction) — currently available engines
- [Architecture](../getting-started/architecture) — system overview
