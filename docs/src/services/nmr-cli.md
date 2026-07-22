# nmr-cli Service

The **nmr-cli** container (`nmr-converter`) packages the
[nmr-load-save](https://github.com/cheminfo/nmr-load-save) toolchain and related
NMR processing utilities. NMRKit invokes it via `docker exec` for spectra
parsing, format conversion, and structure-based prediction.

## Docker image

| Registry | Image | Tags |
|----------|-------|------|
| Docker Hub | `nfdi4chem/nmr-cli` | `dev-latest`, `latest` |

In `docker-compose.yml` the service is named `nmr-load-save` with container
name `nmr-converter`.

## Commands used by NMRKit

| nmr-cli command | API endpoint | Description |
|-----------------|--------------|-------------|
| `parse-spectra` | `POST /spectra/parse/url`, `POST /spectra/parse/file` | Parse Bruker, JCAMP-DX, zipped archives into NMRium JSON |
| `parse-publication-string` | `POST /spectra/parse/publication-string` | Reconstruct spectrum from ACS publication string |
| `peaks-to-nmrium` | `POST /spectra/parse/peaks` | Generate spectrum from a peak list |
| `predict` | `POST /predict/`, `POST /predict/file` | Predict spectra via nmrdb.org or nmrshift |
| `-u <url>` | `GET /convert/spectra` | Legacy URL-based conversion |

### parse-spectra flags

| Flag | API parameter | Description |
|------|---------------|-------------|
| `-u` | `url` | Remote URL to spectra file |
| `-p` | `file` / `auto_processing` | File path or enable FID → FT processing |
| `-s` | `capture_snapshot` | Capture spectra image snapshot |
| `-d` | `auto_detection` | Automatic range/zone detection |
| `-r` | `raw_data` | Include raw data in output |

## Manual testing

Build and run the container locally:

```bash
docker pull nfdi4chem/nmr-cli:dev-latest
docker run -it --name nmr-converter nfdi4chem/nmr-cli:dev-latest bash
```

Inside the container:

```bash
nmr-cli parse-spectra -u https://cheminfo.github.io/bruker-data-test/data/zipped/aspirin-1h.zip
nmr-cli predict -s "<mol block>" --engine nmrshift --spectra proton
```

## Shared volume

The `/shared` volume is mounted in both `nmrkit-api` and `nmr-converter` so
uploaded MOL files can be passed to `nmr-cli predict` without `docker cp`.

## Timeouts

| Operation | Timeout |
|-----------|---------|
| Spectra parsing | 120 s |
| nmrshift prediction | 120 s |
| nmrdb.org prediction | 300 s |

## Related issues

- [#56](https://github.com/NFDI4Chem/nmrkit/issues/56) — NodeJS Docker image with nmr-load-save
- [#57](https://github.com/NFDI4Chem/nmrkit/issues/57) — Docker connection between nmrkit and nmr-load-save
