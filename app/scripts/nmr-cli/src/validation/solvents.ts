import type { Nucleus } from './types'

interface SolventInfo {
  nmrshiftdb: string
  residual: Record<Nucleus, number[]>
}

const SOLVENTS: { match: RegExp; info: SolventInfo }[] = [
  {
    match: /cdcl3|chloroform/,
    info: { nmrshiftdb: 'Chloroform-D1 (CDCl3)', residual: { '1H': [7.26], '13C': [77.16] } },
  },
  {
    match: /dmso|dimethylsulfoxide|dimethylsulphoxide/,
    info: {
      nmrshiftdb: 'Dimethylsulphoxide-D6 (DMSO-D6, C2D6SO)',
      residual: { '1H': [2.5], '13C': [39.52] },
    },
  },
  {
    match: /cd3od|methanol/,
    info: { nmrshiftdb: 'Methanol-D4 (CD3OD)', residual: { '1H': [3.31], '13C': [49.0] } },
  },
  {
    match: /d2o|deuteriumoxide|deuterium oxide|water/,
    info: { nmrshiftdb: 'Deuteriumoxide (D2O)', residual: { '1H': [4.79], '13C': [] } },
  },
  {
    match: /acetone/,
    info: {
      nmrshiftdb: 'Acetone-D6 ((CD3)2CO)',
      residual: { '1H': [2.05], '13C': [29.84, 206.26] },
    },
  },
  {
    match: /ccl4|tetrachloro/,
    info: { nmrshiftdb: 'TETRACHLORO-METHANE (CCl4)', residual: { '1H': [], '13C': [96.1] } },
  },
  {
    match: /pyridin/,
    info: {
      nmrshiftdb: 'Pyridin-D5 (C5D5N)',
      residual: { '1H': [8.74, 7.58, 7.22], '13C': [150.35, 135.91, 123.87] },
    },
  },
  {
    match: /c6d6|benzene/,
    info: { nmrshiftdb: 'Benzene-D6 (C6D6)', residual: { '1H': [7.16], '13C': [128.06] } },
  },
  {
    match: /thf|tetrahydrofuran/,
    info: {
      nmrshiftdb: 'Tetrahydrofuran-D8 (THF-D8, C4D4O)',
      residual: { '1H': [3.58, 1.72], '13C': [67.21, 25.31] },
    },
  },
]

export function resolveSolvent(solvent: string | undefined): SolventInfo {
  const normalized = (solvent ?? '').toLowerCase().replace(/[\s_-]/g, '')
  const found = SOLVENTS.find(({ match }) => match.test(normalized))
  return found?.info ?? { nmrshiftdb: 'Any', residual: { '1H': [], '13C': [] } }
}
