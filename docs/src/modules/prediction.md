# Prediction Module

The prediction module generates NMR spectra from molecular structures. Predictions
are executed by **nmr-cli** inside the `nmr-converter` container using one of two
supported engines.

**Base path:** `/latest/predict`

## Engines

| Engine | Spectra types | Typical time | Notes |
|--------|--------------|--------------|-------|
| `nmrshift` | proton, carbon | 5–10 s | Fast, local shift prediction |
| `nmrdb.org` | proton, carbon, cosy, hsqc, hmbc | 30–60 s | Remote API, supports 2D spectra |

::: tip
Use `curl` or Postman for `nmrdb.org` requests — browser-based Scalar tests may
time out on long predictions.
:::

## Endpoints

### `POST /`

Predict spectra from a MOL block string.

**Request body (nmrshift example):**

```json
{
  "engine": "nmrshift",
  "structure": "\n  Mrv2311...\nM  END",
  "spectra": ["proton"],
  "options": {
    "solvent": "Chloroform-D1 (CDCl3)",
    "frequency": 400,
    "nbPoints": 1024,
    "lineWidth": 1,
    "peakShape": "lorentzian"
  }
}
```

**Request body (nmrdb.org example):**

```json
{
  "engine": "nmrdb.org",
  "structure": "\n  Mrv2311...\nM  END",
  "spectra": ["proton", "carbon"],
  "options": {
    "name": "Benzene",
    "frequency": 400,
    "1d": {
      "proton": { "from": -1, "to": 12 },
      "carbon": { "from": -5, "to": 220 },
      "nbPoints": 131072,
      "lineWidth": 1
    },
    "autoExtendRange": true
  }
}
```

### `POST /file`

Predict from an uploaded MOL file (`multipart/form-data`).

| Form field | Description |
|------------|-------------|
| `file` | MOL file with molecular structure |
| `request` | JSON string with `engine`, `spectra`, and `options` (same schema as `POST /`) |

## nmrshift options

| Option | Default | Description |
|--------|---------|-------------|
| `solvent` | DMSO | NMR solvent (see API enum for full list) |
| `frequency` | 400 | Spectrometer frequency (MHz) |
| `from` / `to` | — | Spectrum range in ppm |
| `nbPoints` | 1024 | Number of data points |
| `lineWidth` | 1 | Peak linewidth |
| `peakShape` | `lorentzian` | `gaussian` or `lorentzian` |
| `tolerance` | 0.001 | Peak grouping tolerance |

## nmrdb.org options

| Option | Default | Description |
|--------|---------|-------------|
| `name` | `""` | Compound name |
| `frequency` | 400 | Spectrometer frequency (MHz) |
| `1d.proton` | -1 to 12 ppm | 1H range |
| `1d.carbon` | -5 to 220 ppm | 13C range |
| `1d.nbPoints` | 131072 | 1D data points |
| `2d.nbPoints` | 1024×1024 | 2D data points |
| `autoExtendRange` | `true` | Extend range if signals fall outside |

## Error codes

| Status | Meaning |
|--------|---------|
| `408` | Prediction timed out (nmrdb.org: 300 s, nmrshift: 120 s) |
| `422` | Invalid structure or CLI error |
| `500` | Docker or container unavailable |

## Example

```bash
curl -X POST "https://dev.nmrkit.nmrxiv.org/latest/predict/" \
  -H "Content-Type: application/json" \
  -d '{
    "engine": "nmrshift",
    "structure": "\n  CDK     09012310592D\n\n  6  6  0  0  0  0  0  0  0  0999 V2000\n    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\nM  END",
    "spectra": ["proton"],
    "options": { "frequency": 400 }
  }'
```

## See also

- [nmr-cli service](../services/nmr-cli)
- [nmr-respredict service](../services/nmr-respredict) — planned additional engine
- [Chemistry module](./chemistry) — HOSE codes used in shift prediction
