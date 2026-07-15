# API Reference

NMRKit exposes an OpenAPI-documented REST API. Interactive documentation is
rendered with [Scalar](https://scalar.com/) (replacing the previous Swagger UI).

## Endpoints

| Environment | Scalar docs | OpenAPI JSON |
|-------------|-------------|--------------|
| Development | [dev.nmrkit.nmrxiv.org/latest/docs](https://dev.nmrkit.nmrxiv.org/latest/docs) | [/latest/openapi.json](https://dev.nmrkit.nmrxiv.org/latest/openapi.json) |
| Production | [nmrkit.nmrxiv.org/latest/docs](https://nmrkit.nmrxiv.org/latest/docs) | [/latest/openapi.json](https://nmrkit.nmrxiv.org/latest/openapi.json) |

Versioned docs are also available at `/v1/docs` and `/v1/openapi.json`.

## Using Scalar

Scalar provides:

- Browsable endpoint groups (chem, spectra, converter, predict, registration)
- Request/response schemas with examples
- A **Test Request** panel for trying endpoints directly in the browser

### Tips for long-running requests

Some endpoints can take 30–60 seconds (notably `nmrdb.org` predictions and
large spectra parsing). For these, prefer `curl` or Postman over the browser UI:

```bash
curl -X POST "https://dev.nmrkit.nmrxiv.org/latest/predict/" \
  -H "Content-Type: application/json" \
  -d '{
    "engine": "nmrshift",
    "structure": "\n  Mrv...\nM  END",
    "spectra": ["proton"],
    "options": { "frequency": 400 }
  }'
```

## Health checks

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Global API health |
| `GET /latest/registration/health` | Used by Docker Compose health checks |

## Further reading

- [Architecture](./architecture) — system components and data flow
- [Spectra module](../modules/spectra) — parsing and conversion routines
- [Prediction module](../modules/prediction) — prediction engines
- [nmr-cli service](../services/nmr-cli) — underlying CLI tool
