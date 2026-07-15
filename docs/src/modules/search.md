# Search Module

::: warning Planned feature
The search module is not yet exposed via the REST API. This page describes the
planned functionality tracked in
[#66 — Expose correlation package functionality](https://github.com/NFDI4Chem/nmrkit/issues/66).
:::

## Planned scope

Expose [nmr-correlation](https://github.com/cheminfo/nmr-correlation) as a
standalone routine:

**Input:**

- URL to a spectra ZIP archive
- Molecular formula
- Optional C/H tolerance overrides

**Output:**

- Correlation object linking experimental peaks to predicted assignments

## Current workaround

Use the [Spectra](./spectra) and [Prediction](./prediction) modules separately,
then correlate results in your client application.
