# Chemistry Module

The chemistry module provides cheminformatics utilities for NMR workflows,
including HOSE (Hierarchically Ordered Spherical Environment) code generation.

**Base path:** `/latest/chem`

## Endpoints

### `GET /hosecode`

Generate HOSE codes for every atom in a molecule.

**Query parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `smiles` | required | SMILES string (e.g. `CCCC1CC1`) |
| `framework` | `cdk` | `cdk` or `rdkit` |
| `spheres` | `3` | Number of bond spheres (1–10) |
| `usestereo` | `false` | Include stereochemistry (CDK only) |

**Response:** Array of HOSE code strings, one per atom.

```bash
curl "https://dev.nmrkit.nmrxiv.org/latest/chem/hosecode?smiles=CCO&framework=cdk&spheres=3"
```

**Example response:**

```json
[
  "C(CC,CC,&)",
  "C(CC,C&,&)",
  "C(CC,CC,&)"
]
```

## Frameworks

| Framework | Library | Stereo support |
|-----------|---------|----------------|
| `cdk` | Chemistry Development Kit | Yes (`usestereo`) |
| `rdkit` | RDKit | No |

HOSE codes encode the local chemical environment around each atom and are
widely used in NMR chemical shift prediction and database lookup.

## Error codes

| Status | Meaning |
|--------|---------|
| `409` | Molecule already exists (unique constraint) |
| `422` | Invalid SMILES or parse error |

## See also

- [Prediction module](./prediction) — structure-based spectrum prediction
- [nmr-respredict service](../services/nmr-respredict) — HOSE-based residual prediction
