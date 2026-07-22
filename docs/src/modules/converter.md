# Converter Module

The converter module provides a lightweight endpoint to convert NMR raw data from
a remote URL into [NMRium](https://www.nmrium.org/)-compatible JSON.

**Base path:** `/latest/convert`

::: info
For full parsing options (auto-processing, detection, snapshots), use the
[Spectra module](./spectra) instead.
:::

## Endpoints

### `GET /spectra`

Fetch NMR data from a URL and convert it to NMRium JSON.

**Query parameter:** `url` — publicly accessible URL to NMR data

```bash
curl "https://dev.nmrkit.nmrxiv.org/latest/convert/spectra?url=https://example.com/data.zip"
```

**Response:** NMRium-compatible JSON.

## Supported formats

Any format recognized by [nmr-load-save](https://github.com/cheminfo/nmr-load-save)
and **nmr-cli**, including Bruker archives and JCAMP-DX files.

## See also

- [Spectra module](./spectra) — full parsing with processing options
- [nmr-cli service](../services/nmr-cli)
