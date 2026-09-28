export type Nucleus = '13C' | '1H'

export const NUCLEI: readonly Nucleus[] = ['13C', '1H']

export interface Tolerance {
  ok: number
  fail: number
}

/**
 * One observed signal. `atoms` are 1-based indices of the submitted molfile.
 * For 1H, an index may point at the heavy atom carrying the protons
 * (NMReDATA convention) or at an explicit H atom. An empty `atoms` list is a
 * signal without assignment; it only contributes to the structure fit.
 */
export interface AssignmentInput {
  nucleus: Nucleus
  atoms: number[]
  label?: string
  shift: number
  multiplicity?: string
  n_h?: number
  diastereotopic?: 'a' | 'b'
}

export interface UnassignedPeakInput {
  nucleus: Nucleus
  shift: number
  kind?: 'solvent' | 'impurity' | 'unknown'
}

export interface AssignmentSetInput {
  structure: { molfile: string; source?: string }
  conditions?: {
    solvent?: string
    temperature_k?: number
    frequency_mhz?: Partial<Record<Nucleus, number>>
  }
  assignments: AssignmentInput[]
  unassigned_peaks?: UnassignedPeakInput[]
  options?: {
    fallback_tolerances?: Partial<Record<Nucleus, Tolerance>>
  }
}

export interface QuickcheckShift {
  atom: number
  prediction: number
  real: number
  diff: number
  status: string
  hoseCode: string
  spheres: number
}

export interface QuickcheckResult {
  id: number
  type: string
  statistics: {
    accept: number
    warning: number
    reject: number
    missing: number
    total: number
  }
  shifts: QuickcheckShift[]
}

export interface QuickcheckInput {
  id: number
  type: string
  shifts: string
  solvent: string
}

export type QuickcheckClient = (
  molfile: string,
  inputs: QuickcheckInput[],
) => Promise<QuickcheckResult[]>

export type ReportStatus = 'green' | 'yellow' | 'red' | 'missing' | 'impossible'

export interface ReportAtomRow {
  label: string
  atoms: number[]
  observed: number | null
  predicted: number | null
  deviation: number | null
  status: ReportStatus
  spheres: number
  shift_values: number | null
  hose_code: string | null
  pair?: string
}

export interface NucleusReport {
  mark: number
  mark_is_approximate: true
  result: 'accept' | 'revise' | 'reject'
  penalties: {
    mean_deviation: { ppm: number; points: number }
    red_or_missing: { count: number; points: number }
    yellow: { count: number; points: number }
  }
  statistics: QuickcheckResult['statistics']
  in_database_likely: boolean
  atoms: ReportAtomRow[]
}

export type AssignmentRowStatus = 'ok' | 'review' | 'fail' | 'not_assessable'

export interface AssignmentRowResult {
  label: string
  nucleus: Nucleus
  atoms: number[]
  observed: number
  predicted: number | null
  delta: number | null
  spheres: number | null
  status: AssignmentRowStatus
  status_source: 'nmrshiftdb' | 'tolerance' | null
  reasons: string[]
}

export interface AssignmentIssue {
  type:
    | 'missing_signal'
    | 'extra_signal'
    | 'equivalence_violation'
    | 'accidental_overlap'
    | 'solvent_assigned'
    | 'proton_count'
    | 'prediction_impossible'
    | 'unknown_atom'
  nucleus: Nucleus
  labels: string[]
  atoms: number[]
  message: string
}

export interface SwapSuggestion {
  type: 'swap'
  nucleus: Nucleus
  labels: [string, string]
  error_reduction: number
}

export interface AssignmentCheck {
  result: 'consistent' | 'review' | 'inconsistent' | 'not_assessable'
  rows: AssignmentRowResult[]
  offset: Partial<Record<Nucleus, number | null>> & {
    suspected_referencing_error: boolean
  }
  suggestions: SwapSuggestion[]
  issues: AssignmentIssue[]
}

export interface ValidationReport {
  engine: { name: 'nmrshift'; source: 'nmrshiftdb2 quickcheck'; url: string }
  solvent: string
  verdict: 'accept' | 'review' | 'reject' | 'not_assessable'
  reports: Partial<Record<Nucleus, NucleusReport>>
  assignment_check: AssignmentCheck
  adjustments: { nucleus: Nucleus; label: string; sent: number; observed: number }[]
}
