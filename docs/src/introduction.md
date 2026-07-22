# Welcome to NMRKit

NMRKit is a collection of microservices for NMR data processing, conversion,
prediction, and molecule registration. It provides a versioned REST API backed
by FastAPI, with interactive documentation powered by Scalar.

## Key capabilities

| Capability | Module | Description |
|------------|--------|-------------|
| Spectra parsing | [Spectra](./modules/spectra) | Parse Bruker, JCAMP-DX, and zipped archives into NMRium JSON |
| Peak list conversion | [Spectra](./modules/spectra) | Generate spectra from peak lists or publication strings |
| Format conversion | [Converter](./modules/converter) | Convert raw NMR data to NMRium format |
| NMR prediction | [Prediction](./modules/prediction) | Predict spectra via nmrdb.org or nmrshift engines |
| HOSE codes | [Chemistry](./modules/chemistry) | Generate HOSE codes for shift prediction |
| Molecule registration | [Registration](./modules/registration) | Register and query molecules via lwreg |

## Services

NMRKit runs as a Docker Compose stack. Besides the API itself, two companion
containers provide specialised NMR processing:

- **[nmr-cli](./services/nmr-cli)** — spectra parsing, conversion, and prediction
- **[nmr-respredict](./services/nmr-respredict)** — residual-based prediction (integration in progress)

See [Architecture](./getting-started/architecture) for a full system diagram.

## Quick links

- **Documentation:** https://nfdi4chem.github.io/nmrkit
- **API reference (dev):** https://dev.nmrkit.nmrxiv.org/latest/docs
- **API reference (prod):** https://nmrkit.nmrxiv.org/latest/docs
- **Source code:** https://github.com/NFDI4Chem/nmrkit
- **Issues:** https://github.com/NFDI4Chem/nmrkit/issues
- **Help desk:** https://helpdesk.nfdi4chem.de/

## Getting started

1. [Local Installation](./development/installation) — run NMRKit with Docker Compose
2. [API Reference](./getting-started/api-reference) — explore endpoints with Scalar
3. [Deployment](./deployment/overview) — deploy to dev, prod, or Kubernetes

Found a bug or have a feature request? Please [open an issue](https://github.com/NFDI4Chem/nmrkit/issues).
