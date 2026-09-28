# Validation Module

The validation module checks ¹H and ¹³C assignments of a structure against
[nmrshiftdb2 quickcheck](https://nmrshiftdb.nmr.uni-koeln.de/) HOSE-code
predictions. It is executed by **nmr-cli** (`validate-assignments`) inside the
`nmr-converter` container.

**Base path:** `/latest/validate`

The prediction is a reference, not the truth: a poor fit flags assignments for a
closer look, and the author keeps the final word.

## Endpoints

### `POST /assignments`

**Request body:**

```json
{
  "structure": { "molfile": "\n  Mnova...\nM  END", "source": "mnova" },
  "conditions": { "solvent": "CDCl3" },
  "assignments": [
    { "nucleus": "13C", "atoms": [1, 3], "label": "C-2, C-6", "shift": 106.65 },
    { "nucleus": "13C", "atoms": [5], "label": "C-4", "shift": 143.52 },
    { "nucleus": "1H", "atoms": [1, 3], "label": "H-2, H-6", "shift": 7.12, "n_h": 2 },
    { "nucleus": "1H", "atoms": [9], "label": "H-5'a", "shift": 2.31, "diastereotopic": "a" }
  ],
  "unassigned_peaks": [{ "nucleus": "13C", "shift": 77.16, "kind": "solvent" }]
}
```

| Field | Notes |
|-------|-------|
| `structure.molfile` | V2000. Explicit H atoms are allowed and are stripped before the servlet call. |
| `assignments[].atoms` | 1-based molfile indices. For ¹H either the carrying heavy atom or an explicit H. Equivalent atoms share one entry. |
| `assignments[].label` | Author label. `"C-2, C-6"` with two atoms gives one report row per atom. |
| `assignments[].diastereotopic` | Marks the two protons of a CH₂; the pair is compared by its mean. |
| `unassigned_peaks` | `unknown` peaks join the structure fit; `solvent` and `impurity` peaks are ignored. |
| `options.fallback_tolerances` | Per nucleus `{ok, fail}` in ppm (defaults: ¹³C 3/6, ¹H 0.3/0.6). |

**Response:** a report with two layers and a combined verdict.

| Key | Question it answers |
|-----|---------------------|
| `reports["13C"]`, `reports["1H"]` | Does the shift list fit the structure? Mark 1–10, `accept`/`revise`/`reject`, penalties, per-atom deviation, HOSE spheres and codes, `in_database_likely`. Mirrors the nmrshiftdb2 quality report. |
| `assignment_check` | Are the shifts on the right atoms? Status per assignment (`ok`, `review`, `fail`, `not_assessable`), swap suggestions, equivalence violations, missing signals, solvent peaks, proton counts, referencing offset. |
| `verdict` | `accept`, `review`, `reject` or `not_assessable`; ¹³C drives it. |
| `adjustments` | Shifts nudged by ±0.001 ppm so the servlet keeps distinct signals with identical values. |
| `cached` | Identical requests are served from a 24 h in-memory cache. |

::: info Scoring
nmrshiftdb2 does not publish its mark formula. The mark is an approximation
(0.5 points per ppm mean deviation, 2 per red or missing atom, 1 per yellow atom,
halved for predictions with at most 2 spheres) and is flagged with
`mark_is_approximate`. Predictions with fewer than 4 HOSE spheres can lead to
`review`, never to `fail`.
:::

**Errors:**

| Status | Meaning |
|--------|---------|
| 422 | Invalid request, unsupported molfile or unknown atom indices |
| 408 | Validation timed out |
| 503 | nmrshiftdb2 quickcheck unavailable, retry later |
| 500 | Docker or `nmr-converter` not available |

## CLI

```bash
cat assignment-set.json | nmr-cli validate-assignments
```

Exit codes: `2` invalid input, `3` quickcheck unavailable.

## Related modules

- [Prediction](./prediction) — nmrshift engine used as the reference
- [Spectra](./spectra) — parse experimental data for validation input
