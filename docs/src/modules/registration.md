# Registration Module

The registration module provides molecule registration, lookup, and retrieval
using [lwreg](https://github.com/rdkit/lwreg) backed by PostgreSQL with the
RDKit cartridge.

**Base path:** `/latest/registration`

## Endpoints

### `POST /init`

Initialize (or re-initialize) the registration database.

::: danger
This destroys all existing registration data. Set `confirm` to `true` to proceed.
:::

```json
{ "confirm": true }
```

### `POST /register`

Register one or more molecules. Send **plain text** (not JSON):

**SMILES input** (one per line):

```
CCCC
CCCCO
c1ccccc1
```

**SDF block** (with `$$$$` delimiters) is also accepted.

**Response:** Array of registry numbers or status strings:

```json
[1, "DUPLICATE", 3]
```

| Value | Meaning |
|-------|---------|
| integer | New molregno for a registered molecule |
| `"DUPLICATE"` | Molecule already exists |
| `"PARSE_FAILURE"` | Could not parse the structure |

### `GET /query`

Check if a molecule is registered.

**Query parameter:** `smi` — SMILES string

**Response:** Array of matching molregnos (empty if not found).

```bash
curl "https://dev.nmrkit.nmrxiv.org/latest/registration/query?smi=CCO"
```

### `POST /retrieve`

Retrieve molecules by registry ID.

**Request body:**

```json
[1, 2, 3]
```

**Response:** Array of `[molregno, data, format]` tuples.

### `GET /health`

Health check used by Docker Compose deployments.

## Configuration

Registration requires PostgreSQL environment variables (see
[Local Installation](../development/installation#docker)):

| Variable | Description |
|----------|-------------|
| `POSTGRES_USER` | Database user |
| `POSTGRES_PASSWORD` | Database password |
| `POSTGRES_SERVER` | Hostname (`pgsql` in Docker Compose) |
| `POSTGRES_DB` | Database name (default: `nmr_predict`) |
| `STANDARDIZATION` | lwreg standardization mode (default: `tautomer`) |

## See also

- [Architecture](../getting-started/architecture) — PostgreSQL component
