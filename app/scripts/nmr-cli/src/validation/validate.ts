import { InvalidStructureError, prepareStructure } from './molfile'
import type { PreparedStructure } from './molfile'
import { resolveSolvent } from './solvents'
import { analyseTopology } from './topology'
import type { Topology } from './topology'
import { NUCLEI } from './types'
import type {
  AssignmentCheck,
  AssignmentInput,
  AssignmentIssue,
  AssignmentRowResult,
  AssignmentRowStatus,
  AssignmentSetInput,
  Nucleus,
  NucleusReport,
  QuickcheckClient,
  QuickcheckInput,
  QuickcheckResult,
  ReportAtomRow,
  ReportStatus,
  SwapSuggestion,
  Tolerance,
  ValidationReport,
} from './types'

export class InvalidInputError extends Error {}

const DEFAULT_TOLERANCES: Record<Nucleus, Tolerance> = {
  '13C': { ok: 3, fail: 6 },
  '1H': { ok: 0.3, fail: 0.6 },
}

const QUICKCHECK_TYPES: Record<Nucleus, { id: number; type: string }> = {
  '13C': { id: 1, type: 'nmr;13C;1d' },
  '1H': { id: 2, type: 'nmr;1H;1d' },
}

/** Values closer than this are treated as the same signal by the servlet. */
const SAME_VALUE = 0.0005
const NUDGE = 0.001

/** Reliable HOSE predictions start at four spheres; below that an atom can at most be flagged for review. */
const RELIABLE_SPHERES = 4

const SWAP_MIN_REDUCTION: Record<Nucleus, number> = { '13C': 2, '1H': 0.2 }
const SWAP_MIN_PREDICTION_GAP: Record<Nucleus, number> = { '13C': 0.5, '1H': 0.05 }
const EQUIVALENCE_SPREAD: Record<Nucleus, number> = { '13C': 0.5, '1H': 0.05 }
const REFERENCING_OFFSET: Record<Nucleus, number> = { '13C': 1, '1H': 0.1 }
const SOLVENT_WINDOW: Record<Nucleus, number> = { '13C': 0.3, '1H': 0.03 }
const IN_DATABASE_MEAN_DEVIATION: Record<Nucleus, number> = { '13C': 1, '1H': 0.1 }
/**
 * nmrshiftdb2 does not publish its mark formula; these weights reproduce the
 * example reports and are exposed as `mark_is_approximate`.
 */
const MARK_POINTS = { perPpmMeanDeviation: 0.5, redOrMissing: 2, yellow: 1 }

const ATOM_STATUS: Record<AssignmentRowStatus, ReportStatus> = {
  ok: 'green',
  review: 'yellow',
  fail: 'red',
  not_assessable: 'yellow',
}

export interface ValidateOptions {
  quickcheck: QuickcheckClient
  url: string
}

interface NormalizedRow {
  index: number
  input: AssignmentInput
  label: string
  /** Original indices of the heavy atoms the signal belongs to. */
  carriers: number[]
  /** Carrier of each submitted atom, aligned with `input.atoms`. */
  atomCarriers: (number | null)[]
  unresolved: number[]
  sent: number
}

interface AtomPrediction {
  prediction: number | null
  spheres: number
  hoseCode: string | null
  statuses: string[]
  reals: number[]
}

type Predictions = Map<Nucleus, Map<number, AtomPrediction>>

export async function validateAssignments(
  input: AssignmentSetInput,
  options: ValidateOptions,
): Promise<ValidationReport> {
  assertValidInput(input)

  const structure = prepareStructure(input.structure.molfile)
  const topology = analyseTopology(structure)
  const solvent = resolveSolvent(input.conditions?.solvent)
  const tolerances = { ...DEFAULT_TOLERANCES, ...(input.options?.fallback_tolerances ?? {}) }

  const rows = normalizeRows(input.assignments, structure, topology)
  const adjustments = assignSentValues(rows, topology)
  const quickcheckInputs = buildQuickcheckInputs(rows, input, solvent.nmrshiftdb)

  const results = quickcheckInputs.length
    ? await options.quickcheck(structure.molfile, quickcheckInputs)
    : []
  const predictions = collectPredictions(results, structure, topology)
  const assignmentCheck = checkAssignments(rows, predictions, topology, structure, input, tolerances, solvent.residual)

  const reports: ValidationReport['reports'] = {}
  for (const nucleus of NUCLEI) {
    if (results.some((item) => item.id === QUICKCHECK_TYPES[nucleus].id)) {
      reports[nucleus] = buildNucleusReport(nucleus, predictions.get(nucleus)!, rows, assignmentCheck.rows, topology)
    }
  }

  return {
    engine: { name: 'nmrshift', source: 'nmrshiftdb2 quickcheck', url: options.url },
    solvent: solvent.nmrshiftdb,
    verdict: overallVerdict(reports, assignmentCheck),
    reports,
    assignment_check: assignmentCheck,
    adjustments: adjustments.map((row) => ({
      nucleus: row.input.nucleus,
      label: row.label,
      sent: round(row.sent, 4),
      observed: row.input.shift,
    })),
  }
}

function assertValidInput(input: AssignmentSetInput): void {
  if (!input || typeof input !== 'object') {
    throw new InvalidInputError('Input must be a JSON object')
  }
  if (typeof input.structure?.molfile !== 'string' || input.structure.molfile.trim() === '') {
    throw new InvalidInputError('structure.molfile is required')
  }
  if (!Array.isArray(input.assignments) || input.assignments.length === 0) {
    throw new InvalidInputError('assignments must contain at least one signal')
  }
  input.assignments.forEach((row, index) => {
    if (!NUCLEI.includes(row.nucleus)) {
      throw new InvalidInputError(`assignments[${index}].nucleus must be 13C or 1H`)
    }
    if (typeof row.shift !== 'number' || !Number.isFinite(row.shift)) {
      throw new InvalidInputError(`assignments[${index}].shift must be a number`)
    }
    if (!Array.isArray(row.atoms) || row.atoms.some((atom) => !Number.isInteger(atom) || atom < 1)) {
      throw new InvalidInputError(`assignments[${index}].atoms must be 1-based atom indices`)
    }
  })
}

function normalizeRows(
  assignments: AssignmentInput[],
  structure: PreparedStructure,
  topology: Topology,
): NormalizedRow[] {
  return assignments.map((input, index) => {
    const atomCarriers = input.atoms.map((atom) => resolveCarrier(atom, input.nucleus, structure, topology))
    const carriers = new Set(atomCarriers.filter((carrier): carrier is number => carrier !== null))

    return {
      index,
      input,
      label: input.label?.trim() || defaultLabel(input),
      carriers: [...carriers].sort((a, b) => a - b),
      atomCarriers,
      unresolved: input.atoms.filter((_, position) => atomCarriers[position] === null),
      sent: input.shift,
    }
  })
}

function resolveCarrier(
  atom: number,
  nucleus: Nucleus,
  structure: PreparedStructure,
  topology: Topology,
): number | null {
  if (nucleus === '1H') {
    const parent = structure.explicitHydrogenParents.get(atom)
    if (parent !== undefined) return parent

    const atomCount = structure.symbols.size
    if (atom > atomCount) {
      return topology.servletHydrogenOwners.get(atom - atomCount + structure.heavyAtomCount) ?? null
    }
    return (topology.hydrogenCounts.get(atom) ?? 0) > 0 ? atom : null
  }

  return structure.symbols.get(atom) === 'C' ? atom : null
}

function defaultLabel(input: AssignmentInput): string {
  if (input.atoms.length === 0) return `${input.shift}`
  const prefix = input.nucleus === '13C' ? 'C' : 'H'
  return `${prefix}-${input.atoms.join(', ')}${input.diastereotopic ?? ''}`
}

/**
 * The servlet collapses identical values, so two different environments with
 * the same observed shift would leave one of them "missing". Such duplicates
 * are nudged apart; identical values on equivalent atoms are sent once.
 */
function assignSentValues(rows: NormalizedRow[], topology: Topology): NormalizedRow[] {
  const adjusted: NormalizedRow[] = []

  for (const nucleus of NUCLEI) {
    const taken: { value: number; classes: string }[] = []
    const nucleusRows = rows
      .filter((row) => row.input.nucleus === nucleus)
      .sort((a, b) => a.input.shift - b.input.shift)

    for (const row of nucleusRows) {
      const classes = row.carriers.map((atom) => topology.classes.get(atom)).sort().join('|')
      const duplicate = taken.find((item) => Math.abs(item.value - row.input.shift) < SAME_VALUE)

      if (duplicate && classes !== '' && duplicate.classes === classes) {
        continue
      }

      let value = row.input.shift
      while (taken.some((item) => Math.abs(item.value - value) < SAME_VALUE)) {
        value += NUDGE
      }
      if (value !== row.input.shift) {
        row.sent = value
        adjusted.push(row)
      }
      taken.push({ value, classes })
    }
  }

  return adjusted
}

function buildQuickcheckInputs(
  rows: NormalizedRow[],
  input: AssignmentSetInput,
  solvent: string,
): QuickcheckInput[] {
  return NUCLEI.flatMap((nucleus) => {
    const values = new Set<number>()
    for (const row of rows) {
      if (row.input.nucleus === nucleus) values.add(round(row.sent, 4))
    }
    for (const peak of input.unassigned_peaks ?? []) {
      if (peak.nucleus === nucleus && (peak.kind ?? 'unknown') === 'unknown') {
        values.add(round(peak.shift, 4))
      }
    }
    if (values.size === 0) return []

    return [
      {
        ...QUICKCHECK_TYPES[nucleus],
        shifts: [...values].sort((a, b) => a - b).map(String).join(';'),
        solvent,
      },
    ]
  })
}

function collectPredictions(
  results: QuickcheckResult[],
  structure: PreparedStructure,
  topology: Topology,
): Predictions {
  const predictions: Predictions = new Map(NUCLEI.map((nucleus) => [nucleus, new Map()]))

  for (const nucleus of NUCLEI) {
    const result = results.find((item) => item.id === QUICKCHECK_TYPES[nucleus].id)
    if (!result) continue
    const byAtom = predictions.get(nucleus)!

    for (const shift of result.shifts) {
      const carrier =
        nucleus === '13C'
          ? structure.toOriginal[shift.atom]
          : topology.servletHydrogenOwners.get(shift.atom)
      if (carrier === undefined) continue

      const impossible = shift.status.startsWith('prediction impossible')
      const entry = byAtom.get(carrier) ?? {
        prediction: null,
        spheres: Number.POSITIVE_INFINITY,
        hoseCode: null,
        statuses: [],
        reals: [],
      }
      if (!impossible) {
        entry.prediction = shift.prediction
        entry.spheres = Math.min(entry.spheres, shift.spheres)
        entry.hoseCode ??= shift.hoseCode || null
      }
      entry.statuses.push(impossible ? 'impossible' : shift.status)
      if (!impossible && shift.status !== 'missing') entry.reals.push(shift.real)
      byAtom.set(carrier, entry)
    }

    for (const entry of byAtom.values()) {
      if (!Number.isFinite(entry.spheres)) entry.spheres = 0
    }
  }

  return predictions
}

/**
 * Per-atom quality report on the author's assignments. The servlet only gets
 * a shift list and matches it to atoms itself, so its own matching can pair a
 * value with a different atom than the author did; observed values and
 * statuses therefore come from the assignment check, keeping both tables in
 * agreement.
 */
function buildNucleusReport(
  nucleus: Nucleus,
  byAtom: Map<number, AtomPrediction>,
  rows: NormalizedRow[],
  checked: AssignmentRowResult[],
  topology: Topology,
): NucleusReport {
  const atomRows: ReportAtomRow[] = []

  for (const [atom, entry] of [...byAtom.entries()].sort(([a], [b]) => a - b)) {
    const label = labelForAtom(atom, nucleus, rows)
    const base = {
      atoms: [atom],
      predicted: entry.prediction === null ? null : round(entry.prediction, 2),
      spheres: entry.spheres,
      shift_values: null,
      hose_code: entry.hoseCode,
    }

    if (entry.prediction === null) {
      atomRows.push({ ...base, label, observed: null, deviation: null, status: 'impossible' })
      continue
    }
    const covering = rowsAssignedTo(atom, nucleus, rows, topology)
    if (covering.length === 0) {
      atomRows.push({ ...base, label, observed: null, deviation: null, status: 'missing' })
      continue
    }

    const status = worstStatus(covering.map((row) => ATOM_STATUS[checked[row.index].status]))
    const observed = [...new Set(covering.map((row) => round(row.input.shift, 4)))].sort((a, b) => a - b)
    const prediction = entry.prediction

    if (nucleus === '1H' && observed.length === 2 && (topology.hydrogenCounts.get(atom) ?? 0) >= 2) {
      const stem = label.replace(/[ab]$/, '')
      const deviation = round(Math.abs(mean(observed) - prediction), 3)
      atomRows.push({ ...base, label: `${stem}a`, observed: observed[0], deviation, status, pair: `${stem}b` })
      atomRows.push({ ...base, label: `${stem}b`, observed: observed[1], deviation, status, pair: `${stem}a` })
      continue
    }

    const value = mean(observed)
    atomRows.push({ ...base, label, observed: round(value, 4), deviation: round(Math.abs(value - prediction), 3), status })
  }

  const scored = atomRows.filter((row) => row.deviation !== null)
  const meanDeviation = scored.length ? mean(scored.map((row) => row.deviation!)) : 0
  const lowConfidenceWeight = (row: ReportAtomRow) => (row.spheres <= 2 ? 0.5 : 1)
  const redOrMissing = atomRows.filter((row) => row.status === 'red' || row.status === 'missing')
  const yellow = atomRows.filter((row) => row.status === 'yellow')

  const deviationPoints = truncate(meanDeviation * MARK_POINTS.perPpmMeanDeviation, 2)
  const redPoints = round(redOrMissing.reduce((sum, row) => sum + MARK_POINTS.redOrMissing * lowConfidenceWeight(row), 0), 2)
  const yellowPoints = round(yellow.reduce((sum, row) => sum + MARK_POINTS.yellow * lowConfidenceWeight(row), 0), 2)
  const mark = Math.min(10, Math.max(1, Math.round(10 - deviationPoints - redPoints - yellowPoints)))

  const sixSphereShare = scored.length ? scored.filter((row) => row.spheres >= 6).length / scored.length : 0
  const red = atomRows.filter((row) => row.status === 'red')

  return {
    mark,
    mark_is_approximate: true,
    result: mark >= 8 && red.length === 0 ? 'accept' : mark >= 5 ? 'revise' : 'reject',
    penalties: {
      mean_deviation: { ppm: round(meanDeviation, 2), points: deviationPoints },
      red_or_missing: { count: redOrMissing.length, points: redPoints },
      yellow: { count: yellow.length, points: yellowPoints },
    },
    statistics: {
      accept: atomRows.filter((row) => row.status === 'green').length,
      warning: yellow.length,
      reject: red.length,
      missing: atomRows.filter((row) => row.status === 'missing').length,
      total: atomRows.length,
    },
    in_database_likely:
      scored.length >= 3 &&
      sixSphereShare >= 0.8 &&
      meanDeviation <= IN_DATABASE_MEAN_DEVIATION[nucleus],
    atoms: atomRows,
  }
}

/**
 * Name for an atom that has no signal of this nucleus, borrowed from the
 * carbon label when the author assigned one ("C-4'" gives "H-4'").
 */
function atomName(atom: number, nucleus: Nucleus, rows: NormalizedRow[]): string {
  const carbonRow = rows.find((row) => row.input.nucleus === '13C' && row.carriers.includes(atom) && row.input.label)
  if (!carbonRow) return `atom ${atom}`
  const carbonLabel = labelForAtom(atom, '13C', rows)
  if (nucleus === '13C') return carbonLabel
  return /^C(?=[-\d])/.test(carbonLabel) ? carbonLabel.replace(/^C/, 'H') : `H on ${carbonLabel}`
}

/**
 * Report rows are per atom, like the nmrshiftdb quality report. A label that
 * lists one name per submitted atom ("C-2, C-6") is split accordingly.
 */
function labelForAtom(atom: number, nucleus: Nucleus, rows: NormalizedRow[]): string {
  const row = rows.find((item) => item.input.nucleus === nucleus && item.carriers.includes(atom))
  if (!row) return String(atom)
  if (row.carriers.length === 1) {
    return row.input.diastereotopic ? row.label.replace(/[ab]$/, '') : row.label
  }

  const parts = row.label.split(/\s*(?:,|;|\band\b)\s*/).filter(Boolean)
  const position = row.atomCarriers.indexOf(atom)
  if (parts.length === row.input.atoms.length && position >= 0) return parts[position]
  return `${row.label} [${atom}]`
}

/**
 * The author's rows for an atom; an atom left out of a row that covers a
 * symmetry-equivalent atom ("C-2" for both C-2 and C-6) shares that row.
 */
function rowsAssignedTo(atom: number, nucleus: Nucleus, rows: NormalizedRow[], topology: Topology): NormalizedRow[] {
  const nucleusRows = rows.filter((row) => row.input.nucleus === nucleus)
  const own = nucleusRows.filter((row) => row.carriers.includes(atom))
  if (own.length) return own

  const cls = topology.classes.get(atom)
  if (cls === undefined) return []
  return nucleusRows.filter((row) => row.carriers.some((carrier) => topology.classes.get(carrier) === cls))
}

function worstStatus(statuses: ReportStatus[]): ReportStatus {
  if (statuses.includes('red')) return 'red'
  if (statuses.includes('yellow')) return 'yellow'
  return 'green'
}

function checkAssignments(
  rows: NormalizedRow[],
  predictions: Predictions,
  topology: Topology,
  structure: PreparedStructure,
  input: AssignmentSetInput,
  tolerances: Record<Nucleus, Tolerance>,
  residuals: Record<Nucleus, number[]>,
): AssignmentCheck {
  const issues: AssignmentIssue[] = []
  const pairMeans = diastereotopicPairMeans(rows, topology)

  const results: AssignmentRowResult[] = rows.map((row) => {
    const { nucleus } = row.input
    const reasons: string[] = []

    if (row.unresolved.length) {
      issues.push({
        type: 'unknown_atom',
        nucleus,
        labels: [row.label],
        atoms: row.unresolved,
        message:
          nucleus === '13C'
            ? 'Atom is not a carbon in the submitted structure.'
            : 'Atom carries no hydrogen in the submitted structure.',
      })
    }

    const entries = row.carriers.map((atom) => predictions.get(nucleus)?.get(atom))
    const valid = entries.filter((entry): entry is AtomPrediction => entry?.prediction != null)
    const impossible = row.carriers.filter((atom, i) => entries[i] && entries[i]!.prediction == null)
    if (impossible.length) {
      issues.push({
        type: 'prediction_impossible',
        nucleus,
        labels: [row.label],
        atoms: impossible,
        message: 'nmrshiftdb could not predict a shift for this atom.',
      })
    }

    const base = {
      label: row.label,
      nucleus,
      atoms: row.carriers,
      observed: row.input.shift,
    }

    if (row.carriers.length === 0 || valid.length === 0) {
      return { ...base, predicted: null, delta: null, spheres: null, status: 'not_assessable', status_source: null, reasons }
    }

    const predicted = mean(valid.map((entry) => entry.prediction!))
    const spheres = Math.min(...valid.map((entry) => entry.spheres))
    const pairMean = pairMeans.get(row.index)
    if (pairMean !== undefined) reasons.push('diastereotopic_pair_mean')
    const delta = (pairMean ?? row.input.shift) - predicted

    const matchedByServlet = valid.every((entry) =>
      entry.reals.some((real) => Math.abs(real - row.sent) < SAME_VALUE),
    )
    let status: AssignmentRowStatus
    let source: AssignmentRowResult['status_source']
    if (matchedByServlet && pairMean === undefined) {
      status = servletStatus(valid.flatMap((entry) => entry.statuses))
      source = 'nmrshiftdb'
    } else {
      status = toleranceStatus(Math.abs(delta), spheres, tolerances[nucleus])
      source = 'tolerance'
    }
    if (status === 'fail' && spheres < RELIABLE_SPHERES) {
      status = 'review'
      reasons.push('low_confidence_prediction')
    }

    return {
      ...base,
      predicted: round(predicted, 2),
      delta: round(delta, 3),
      spheres,
      status,
      status_source: source,
      reasons,
    }
  })

  issues.push(...structuralIssues(rows, results, topology, structure, input, residuals))
  const suggestions = swapSuggestions(results, pairMeans, rows, tolerances)
  const offset = referencingOffset(results)

  return {
    result: assignmentResult(results, suggestions, issues),
    rows: results,
    offset,
    suggestions,
    issues,
  }
}

/**
 * nmrshiftdb predicts one value per CH2, so diastereotopic protons assigned to
 * the same carbon are compared through their mean, as in its quality report.
 */
function diastereotopicPairMeans(rows: NormalizedRow[], topology: Topology): Map<number, number> {
  const byCarrier = new Map<number, NormalizedRow[]>()
  for (const row of rows) {
    if (row.input.nucleus !== '1H' || row.carriers.length !== 1) continue
    const carrier = row.carriers[0]
    if ((topology.hydrogenCounts.get(carrier) ?? 0) < 2) continue
    byCarrier.set(carrier, [...(byCarrier.get(carrier) ?? []), row])
  }

  const means = new Map<number, number>()
  for (const group of byCarrier.values()) {
    if (group.length < 2) continue
    const value = mean(group.map((row) => row.input.shift))
    for (const row of group) means.set(row.index, value)
  }
  return means
}

function servletStatus(statuses: string[]): AssignmentRowStatus {
  if (statuses.includes('reject')) return 'fail'
  if (statuses.includes('warning')) return 'review'
  return 'ok'
}

function toleranceStatus(deviation: number, spheres: number, tolerance: Tolerance): AssignmentRowStatus {
  const factor = spheres >= RELIABLE_SPHERES ? 1 : spheres === 3 ? 1.5 : 2
  if (deviation <= tolerance.ok * factor) return 'ok'
  if (deviation <= tolerance.fail * factor) return 'review'
  return 'fail'
}

function structuralIssues(
  rows: NormalizedRow[],
  results: AssignmentRowResult[],
  topology: Topology,
  structure: PreparedStructure,
  input: AssignmentSetInput,
  residuals: Record<Nucleus, number[]>,
): AssignmentIssue[] {
  const issues: AssignmentIssue[] = []

  for (const nucleus of NUCLEI) {
    const nucleusRows = rows.filter((row) => row.input.nucleus === nucleus)
    const assigned = nucleusRows.filter((row) => row.carriers.length > 0)
    if (assigned.length === 0) continue

    const byClass = new Map<string, NormalizedRow[]>()
    for (const row of assigned) {
      const classes = new Set(row.carriers.map((atom) => topology.classes.get(atom)!))
      if (classes.size > 1) {
        issues.push({
          type: 'accidental_overlap',
          nucleus,
          labels: [row.label],
          atoms: row.carriers,
          message: 'One shift is assigned to atoms that are not equivalent.',
        })
      }
      for (const cls of classes) byClass.set(cls, [...(byClass.get(cls) ?? []), row])
    }

    for (const group of byClass.values()) {
      const distinctCarriers = new Set(group.flatMap((row) => row.carriers))
      if (group.length < 2 || distinctCarriers.size < 2) continue
      const shifts = group.map((row) => row.input.shift)
      if (Math.max(...shifts) - Math.min(...shifts) > EQUIVALENCE_SPREAD[nucleus]) {
        issues.push({
          type: 'equivalence_violation',
          nucleus,
          labels: group.map((row) => row.label),
          atoms: [...distinctCarriers].sort((a, b) => a - b),
          message: 'Symmetry-equivalent atoms are assigned different shifts.',
        })
      }
    }

    const covered = new Set(assigned.flatMap((row) => row.carriers.map((atom) => topology.classes.get(atom))))
    const missing: number[] = []
    for (const [atom, cls] of topology.classes) {
      const isCarbon = structure.symbols.get(atom) === 'C'
      const relevant = nucleus === '13C' ? isCarbon : isCarbon && (topology.hydrogenCounts.get(atom) ?? 0) > 0
      if (relevant && !covered.has(cls)) missing.push(atom)
    }
    if (missing.length) {
      issues.push({
        type: 'missing_signal',
        nucleus,
        labels: missing.map((atom) => atomName(atom, nucleus, rows)),
        atoms: missing,
        message:
          nucleus === '13C'
            ? 'Carbon environments without an assigned signal (often quaternary carbons).'
            : 'C-H environments without an assigned signal.',
      })
    }

    for (const row of nucleusRows) {
      const result = results[rows.indexOf(row)]
      const nearSolvent = residuals[nucleus].some((value) => Math.abs(value - row.input.shift) <= SOLVENT_WINDOW[nucleus])
      if (nearSolvent && result.status !== 'ok' && result.status !== 'not_assessable') {
        issues.push({
          type: 'solvent_assigned',
          nucleus,
          labels: [row.label],
          atoms: row.carriers,
          message: 'Shift is at a residual solvent position and does not fit the prediction.',
        })
      }
    }

    if (nucleus === '1H') {
      for (const row of assigned) {
        if (row.input.n_h === undefined) continue
        const carried = row.carriers.reduce((sum, atom) => sum + (topology.hydrogenCounts.get(atom) ?? 0), 0)
        const shared = assigned.filter((other) => other.carriers.join() === row.carriers.join()).length
        if (shared === 1 && row.input.n_h !== carried) {
          issues.push({
            type: 'proton_count',
            nucleus,
            labels: [row.label],
            atoms: row.carriers,
            message: `Integral of ${row.input.n_h} H, but the assigned atoms carry ${carried} H.`,
          })
        }
      }
    }
  }

  for (const peak of input.unassigned_peaks ?? []) {
    if ((peak.kind ?? 'unknown') === 'solvent') continue
    issues.push({
      type: 'extra_signal',
      nucleus: peak.nucleus,
      labels: [String(peak.shift)],
      atoms: [],
      message: peak.kind === 'impurity' ? 'Signal marked as impurity.' : 'Signal without assignment.',
    })
  }

  return issues
}

function swapSuggestions(
  results: AssignmentRowResult[],
  pairMeans: Map<number, number>,
  rows: NormalizedRow[],
  tolerances: Record<Nucleus, Tolerance>,
): SwapSuggestion[] {
  const suggestions: SwapSuggestion[] = []
  const eligible = results
    .map((result, index) => ({ result, index }))
    .filter(
      ({ result, index }) =>
        result.predicted !== null &&
        (result.spheres ?? 0) >= RELIABLE_SPHERES &&
        !pairMeans.has(rows[index].index),
    )

  for (let i = 0; i < eligible.length; i++) {
    for (let j = i + 1; j < eligible.length; j++) {
      const a = eligible[i].result
      const b = eligible[j].result
      if (a.nucleus !== b.nucleus) continue
      if (Math.abs(a.predicted! - b.predicted!) <= SWAP_MIN_PREDICTION_GAP[a.nucleus]) continue

      const current = Math.abs(a.observed - a.predicted!) + Math.abs(b.observed - b.predicted!)
      const swappedA = Math.abs(a.observed - b.predicted!)
      const swappedB = Math.abs(b.observed - a.predicted!)
      const reduction = current - swappedA - swappedB

      if (
        reduction >= SWAP_MIN_REDUCTION[a.nucleus] &&
        Math.max(swappedA, swappedB) <= tolerances[a.nucleus].fail
      ) {
        suggestions.push({ type: 'swap', nucleus: a.nucleus, labels: [a.label, b.label], error_reduction: round(reduction, 2) })
      }
    }
  }

  return suggestions.sort((a, b) => b.error_reduction - a.error_reduction).slice(0, 10)
}

function referencingOffset(results: AssignmentRowResult[]): AssignmentCheck['offset'] {
  const offset: AssignmentCheck['offset'] = { suspected_referencing_error: false }

  for (const nucleus of NUCLEI) {
    const deltas = results
      .filter((row) => row.nucleus === nucleus && row.delta !== null && (row.spheres ?? 0) >= RELIABLE_SPHERES)
      .map((row) => row.delta!)
    if (deltas.length < 3) {
      offset[nucleus] = null
      continue
    }
    const value = round(median(deltas), 3)
    offset[nucleus] = value
    if (Math.abs(value) > REFERENCING_OFFSET[nucleus]) offset.suspected_referencing_error = true
  }

  return offset
}

function assignmentResult(
  results: AssignmentRowResult[],
  suggestions: SwapSuggestion[],
  issues: AssignmentIssue[],
): AssignmentCheck['result'] {
  const assessed = results.filter((row) => row.status !== 'not_assessable')
  if (assessed.length === 0) return 'not_assessable'
  if (assessed.every((row) => (row.spheres ?? 0) <= 2)) return 'not_assessable'
  if (assessed.some((row) => row.status === 'fail')) return 'inconsistent'
  if (
    assessed.some((row) => row.status === 'review') ||
    suggestions.length > 0 ||
    issues.some((issue) => issue.type === 'equivalence_violation' || issue.type === 'unknown_atom')
  ) {
    return 'review'
  }
  return 'consistent'
}

function overallVerdict(
  reports: ValidationReport['reports'],
  check: AssignmentCheck,
): ValidationReport['verdict'] {
  const primary = reports['13C'] ?? reports['1H']
  if (!primary) return 'not_assessable'
  if (primary.result === 'reject' || check.result === 'inconsistent') return 'reject'
  if (primary.result === 'revise' || check.result === 'review') return 'review'
  return 'accept'
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2
}

function round(value: number, digits: number): number {
  const factor = 10 ** digits
  return Math.round(value * factor) / factor
}

function truncate(value: number, digits: number): number {
  const factor = 10 ** digits
  return Math.floor(value * factor + 1e-9) / factor
}

export { InvalidStructureError }
