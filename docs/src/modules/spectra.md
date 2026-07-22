# Spectra Module

The spectra module parses experimental NMR data into
[NMRium](https://www.nmrium.org/)-compatible JSON. All operations delegate to
the **nmr-cli** `parse-spectra` toolchain running in the `nmr-converter`
container.

**Base path:** `/latest/spectra`

## Endpoints

### `POST /parse/url`

Parse spectra from a remote URL.

**Request body (JSON):**

```json
{
  "url": "https://example.com/spectra/sample.zip",
  "capture_snapshot": false,
  "auto_processing": true,
  "auto_detection": true,
  "raw_data": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `url` | URL | required | Link to NMR data (JCAMP-DX, Bruker zip, etc.) |
| `capture_snapshot` | bool | `false` | Generate a PNG snapshot of the spectrum |
| `auto_processing` | bool | `false` | Process FID → FT spectra automatically |
| `auto_detection` | bool | `false` | Detect ranges and zones automatically |
| `raw_data` | bool | `false` | Embed raw data in the output |

**Response:** NMRium-compatible JSON (`application/json`).

### `POST /parse/file`

Parse an uploaded spectra file (`multipart/form-data`).

| Form field | Type | Default | Description |
|------------|------|---------|-------------|
| `file` | file | required | JCAMP-DX, Bruker zip, or other supported format |
| `capture_snapshot` | bool | `false` | Generate snapshot image |
| `auto_processing` | bool | `false` | FID → FT processing |
| `auto_detection` | bool | `false` | Range/zone detection |
| `raw_data` | bool | `false` | Include raw data |

### `POST /parse/publication-string`

Reconstruct a spectrum from an ACS-style NMR publication string. Send the string
as **plain text** (not JSON).

**Example body:**

```
1H NMR (400 MHz, CDCl3) δ 7.26 (s, 1H), 2.10 (s, 3H)
```

### `POST /parse/peaks`

Convert a peak list into a simulated 1D spectrum.

**Request body (JSON):**

```json
{
  "peaks": [
    { "x": 7.26, "y": 1, "width": 1 },
    { "x": 2.10, "y": 1, "width": 1 }
  ],
  "options": {
    "nucleus": "1H",
    "frequency": 400,
    "nbPoints": 131072,
    "from": 0,
    "to": 12
  }
}
```

| Peak field | Required | Description |
|------------|----------|-------------|
| `x` | yes | Chemical shift (ppm) |
| `y` | no | Intensity (default `1.0`) |
| `width` | no | Peak width in Hz (default `1.0`) |

## Supported input formats

- JCAMP-DX (`.jdx`, `.dx`)
- Bruker directories (zipped)
- Other formats supported by [nmr-load-save](https://github.com/cheminfo/nmr-load-save)

## Error codes

| Status | Meaning |
|--------|---------|
| `408` | Processing exceeded 120 s timeout |
| `422` | Parse error or invalid input |
| `500` | Docker or `nmr-converter` container unavailable |

## Example

```bash
curl -X POST "https://dev.nmrkit.nmrxiv.org/latest/spectra/parse/url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://cheminfo.github.io/bruker-data-test/data/zipped/aspirin-1h.zip",
    "auto_processing": true
  }' \
  -o spectrum.json
```

## See also

- [nmr-cli service](../services/nmr-cli)
- [Converter module](./converter) — legacy URL conversion endpoint
