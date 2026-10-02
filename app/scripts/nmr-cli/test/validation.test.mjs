import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { validateAssignments, InvalidInputError } from '../build/validation/validate.js'
import { prepareStructure, InvalidStructureError } from '../build/validation/molfile.js'
import { analyseTopology } from '../build/validation/topology.js'

const __dirname = dirname(fileURLToPath(import.meta.url))
const fixture = (name) => readFileSync(join(__dirname, 'fixtures', name), 'utf-8')
const responses = JSON.parse(fixture('quickcheck-responses.json'))

function fakeQuickcheck(key) {
  const calls = []
  const client = async (molfile, inputs) => {
    calls.push({ molfile, inputs })
    return responses[key].result
  }
  return { client, calls }
}

async function run(input, key) {
  const { client, calls } = fakeQuickcheck(key)
  const report = await validateAssignments(input, { quickcheck: client, url: 'https://example.test/quickcheck' })
  return { report, calls }
}

const rowByLabel = (report, label) => report.assignment_check.rows.find((row) => row.label === label)

/** Mnova export of 3,4,5-trimethoxybenzaldehyde; atom 8 is the explicit aldehyde H. */
function trimethoxybenzaldehyde(overrides = {}) {
  const shifts = { 'C-1': 131.677, 'C-4': 143.518, ...overrides }
  return {
    structure: { molfile: fixture('trimethoxybenzaldehyde.mol'), source: 'mnova' },
    conditions: { solvent: 'CDCl3' },
    assignments: [
      { nucleus: '13C', atoms: [1, 3], label: 'C-6, C-2', shift: 106.646 },
      { nucleus: '13C', atoms: [2], label: 'C-1', shift: shifts['C-1'] },
      { nucleus: '13C', atoms: [4, 6], label: 'C-3, C-5', shift: 153.612 },
      { nucleus: '13C', atoms: [5], label: 'C-4', shift: shifts['C-4'] },
      { nucleus: '13C', atoms: [7], label: 'C-7', shift: 191.088 },
      { nucleus: '13C', atoms: [13, 15], label: 'C-8, C-10', shift: 56.258 },
      { nucleus: '13C', atoms: [14], label: 'C-9', shift: 61.009 },
      { nucleus: '1H', atoms: [1, 3], label: 'H-6, H-2', shift: 7.123, n_h: 2 },
      { nucleus: '1H', atoms: [8], label: 'H-7', shift: 9.862, n_h: 1 },
      { nucleus: '1H', atoms: [13, 15], label: 'H-8, H-10', shift: 3.926, n_h: 6 },
      { nucleus: '1H', atoms: [14], label: 'H-9', shift: 3.933, n_h: 3 },
    ],
  }
}

/** Nicotine, numbered as in test/fixtures/nicotine.mol (1 N-CH3, 3-6 pyrrolidine C, 7-12 pyridine). */
function nicotine() {
  const c = (atoms, label, shift) => ({ nucleus: '13C', atoms, label, shift })
  const h = (atoms, label, shift, extra = {}) => ({ nucleus: '1H', atoms, label, shift, ...extra })
  return {
    structure: { molfile: fixture('nicotine.mol') },
    conditions: { solvent: 'CDCl3' },
    assignments: [
      c([12], 'C-2', 149.91),
      c([7], 'C-3', 138.64),
      c([8], 'C-4', 135.15),
      c([9], 'C-5', 123.35),
      c([10], 'C-6', 148.46),
      c([6], "C-2'", 68.94),
      c([3], "C-5'", 56.88),
      c([4], "C-4'", 22.53),
      c([5], "C-3'", 35.34),
      c([1], 'N-CH3', 40.35),
      h([12], 'H-2', 8.55),
      h([8], 'H-4', 7.69),
      h([9], 'H-5', 7.24),
      h([10], 'H-6', 8.5),
      h([6], "H-2'", 3.08),
      h([3], "H-5'a", 2.31, { n_h: 1, diastereotopic: 'a' }),
      h([3], "H-5'b", 3.24, { n_h: 1, diastereotopic: 'b' }),
      h([4], "H-4'a", 1.81, { n_h: 1, diastereotopic: 'a' }),
      h([4], "H-4'b", 1.95, { n_h: 1, diastereotopic: 'b' }),
      h([5], "H-3'a", 1.73, { n_h: 1, diastereotopic: 'a' }),
      h([5], "H-3'b", 2.19, { n_h: 1, diastereotopic: 'b' }),
      h([1], 'N-CH3', 2.17, { n_h: 3 }),
    ],
  }
}

test('explicit hydrogens are stripped and heavy atoms renumbered', () => {
  const structure = prepareStructure(fixture('trimethoxybenzaldehyde.mol'))

  assert.equal(structure.heavyAtomCount, 14)
  assert.deepEqual(structure.toOriginal.slice(1), [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15])
  assert.equal(structure.explicitHydrogenParents.get(8), 7)
  assert.doesNotMatch(structure.molfile, /^\s+\S+\s+\S+\s+\S+\s+H\s/m)
  assert.doesNotMatch(structure.molfile, /M {2}ZZC/)
})

test('servlet hydrogen numbers follow the heavy atoms they belong to', () => {
  const topology = analyseTopology(prepareStructure(fixture('nicotine.mol')))
  const owners = Object.fromEntries(topology.servletHydrogenOwners)

  assert.deepEqual(
    Object.entries(owners).map(([hydrogen, owner]) => [Number(hydrogen), owner]),
    [
      [13, 1], [14, 1], [15, 1],
      [16, 3], [17, 3], [18, 4], [19, 4], [20, 5], [21, 5],
      [22, 6], [23, 8], [24, 9], [25, 10], [26, 12],
    ],
  )
})

test('a correct assignment gets mark 10 and a consistent result', async () => {
  const { report, calls } = await run(trimethoxybenzaldehyde(), 'trimethoxybenzaldehyde-correct')

  assert.equal(calls.length, 1)
  assert.deepEqual(
    calls[0].inputs.map(({ id, type, solvent }) => ({ id, type, solvent })),
    [
      { id: 1, type: 'nmr;13C;1d', solvent: 'Chloroform-D1 (CDCl3)' },
      { id: 2, type: 'nmr;1H;1d', solvent: 'Chloroform-D1 (CDCl3)' },
    ],
  )
  assert.equal(calls[0].inputs[0].shifts.split(';').length, 7)
  assert.equal(calls[0].inputs[1].shifts.split(';').length, 4)

  assert.equal(report.reports['13C'].mark, 10)
  assert.equal(report.reports['13C'].result, 'accept')
  assert.equal(report.reports['1H'].mark, 10)
  assert.equal(report.assignment_check.result, 'consistent')
  assert.equal(report.verdict, 'accept')
  assert.deepEqual(report.assignment_check.suggestions, [])

  const labels = report.reports['13C'].atoms.map((row) => row.label)
  assert.ok(labels.includes('C-2') && labels.includes('C-6'), `per-atom labels expected, got ${labels}`)
  assert.equal(rowByLabel(report, 'H-7').atoms[0], 7)
})

test('a wrong carbon shift is flagged red and fails the assignment', async () => {
  const { report } = await run(trimethoxybenzaldehyde({ 'C-4': 130 }), 'trimethoxybenzaldehyde-c4-wrong')

  const c4 = report.reports['13C'].atoms.find((row) => row.label === 'C-4')
  assert.equal(c4.status, 'red')
  assert.equal(report.reports['13C'].penalties.red_or_missing.count, 1)
  assert.ok(report.reports['13C'].mark < 8)
  assert.equal(rowByLabel(report, 'C-4').status, 'fail')
  assert.equal(report.assignment_check.result, 'inconsistent')
  assert.equal(report.verdict, 'reject')
})

test('swapped carbon labels are red in the quality report, fail the assignment and suggest a swap', async () => {
  const { report } = await run(
    trimethoxybenzaldehyde({ 'C-1': 143.518, 'C-4': 131.677 }),
    'trimethoxybenzaldehyde-correct',
  )

  const carbons = Object.fromEntries(report.reports['13C'].atoms.map((row) => [row.label, row]))
  assert.equal(carbons['C-1'].observed, 143.518)
  assert.equal(carbons['C-1'].status, 'red')
  assert.equal(carbons['C-4'].status, 'red')
  assert.ok(report.reports['13C'].mark < 8)
  assert.notEqual(report.reports['13C'].result, 'accept')
  assert.equal(rowByLabel(report, 'C-1').status, 'fail')
  assert.equal(rowByLabel(report, 'C-4').status, 'fail')
  assert.equal(report.assignment_check.result, 'inconsistent')
  assert.deepEqual(report.assignment_check.suggestions[0].labels.slice().sort(), ['C-1', 'C-4'])
})

test('nicotine: diastereotopic pairs are scored by their mean and low-sphere atoms never fail', async () => {
  const { report, calls } = await run(nicotine(), 'nicotine')

  assert.equal(calls[0].inputs[1].shifts.split(';').length, 12)
  assert.equal(report.reports['13C'].mark, 10)

  const pairRows = report.reports['1H'].atoms.filter((row) => row.atoms[0] === 3)
  assert.deepEqual(pairRows.map((row) => row.label), ["H-5'a", "H-5'b"])
  assert.deepEqual(pairRows.map((row) => row.observed), [2.31, 3.24])

  assert.equal(rowByLabel(report, "H-5'a").status, 'ok')
  assert.equal(rowByLabel(report, "H-5'b").status, 'ok')

  const h2prime = rowByLabel(report, "H-2'")
  assert.equal(h2prime.spheres, 3)
  assert.equal(h2prime.status, 'review')
  assert.ok(h2prime.reasons.includes('low_confidence_prediction'))

  assert.ok(report.assignment_check.rows.every((row) => row.status !== 'fail'))
  assert.equal(report.assignment_check.result, 'review')
})

test('the quality report follows the author\'s assignments, not the servlet\'s own matching', async () => {
  const response = structuredClone(responses['trimethoxybenzaldehyde-correct'])
  const proton = response.result.find((result) => result.id === 2)
  for (const shift of proton.shifts) {
    if (shift.atom === 15 || shift.atom === 16) shift.real = 9.862
    if (shift.atom === 17) shift.real = 7.123
  }
  responses['trimethoxybenzaldehyde-rematched'] = response

  const { report } = await run(trimethoxybenzaldehyde(), 'trimethoxybenzaldehyde-rematched')

  const protons = report.reports['1H'].atoms
  assert.equal(protons.find((row) => row.atoms[0] === 7).observed, 9.862)
  assert.equal(protons.find((row) => row.atoms[0] === 1).observed, 7.123)
  assert.ok(protons.every((row) => row.status === 'green'))
  assert.equal(report.reports['1H'].result, 'accept')
  assert.equal(report.assignment_check.result, 'consistent')
})

test('one proton that does not match keeps the nucleus from a good fit', async () => {
  const input = trimethoxybenzaldehyde()
  input.assignments.find((row) => row.label === 'H-7').shift = 8.2

  const { report } = await run(input, 'trimethoxybenzaldehyde-correct')

  const aldehyde = report.reports['1H'].atoms.find((row) => row.label === 'H-7')
  assert.equal(aldehyde.observed, 8.2)
  assert.equal(aldehyde.status, 'red')
  assert.equal(rowByLabel(report, 'H-7').status, 'fail')
  assert.equal(report.reports['1H'].mark, 8)
  assert.equal(report.reports['1H'].result, 'revise')
  assert.equal(report.reports['1H'].statistics.reject, 1)
})

test('unassigned C-H environments are reported by the author\'s carbon labels', async () => {
  const input = nicotine()
  input.assignments = input.assignments.filter((row) => !/^H-[34]'/.test(row.label))

  const { report } = await run(input, 'nicotine')

  const missing = report.assignment_check.issues.find((issue) => issue.type === 'missing_signal')
  assert.equal(missing.nucleus, '1H')
  assert.deepEqual(missing.labels, ["H-4'", "H-3'"])
  assert.deepEqual(missing.atoms, [4, 5])
})

test('1H atoms past the molfile resolve as implicit hydrogens (NMRium numbering)', async () => {
  const input = nicotine()
  const implicitHydrogens = { "H-2'": [22], "H-5'a": [16], "H-5'b": [17], 'N-CH3': [13, 14, 15] }
  input.assignments = input.assignments.map((row) =>
    row.nucleus === '1H' && implicitHydrogens[row.label] ? { ...row, atoms: implicitHydrogens[row.label] } : row,
  )

  const { report } = await run(input, 'nicotine')

  assert.deepEqual(rowByLabel(report, "H-2'").atoms, [6])
  assert.deepEqual(rowByLabel(report, "H-5'a").atoms, [3])
  assert.deepEqual(report.assignment_check.rows.find((row) => row.nucleus === '1H' && row.label === 'N-CH3').atoms, [1])
  assert.ok(!report.assignment_check.issues.some((issue) => issue.type === 'unknown_atom'))
})

test('distinct signals with the same shift are nudged so the servlet keeps both', async () => {
  const input = trimethoxybenzaldehyde()
  input.assignments.find((row) => row.label === 'H-9').shift = 3.926

  const { report, calls } = await run(input, 'trimethoxybenzaldehyde-correct')

  assert.equal(calls[0].inputs[1].shifts.split(';').length, 4)
  assert.equal(report.adjustments.length, 1)
  assert.equal(report.adjustments[0].observed, 3.926)
  assert.ok(Math.abs(report.adjustments[0].sent - 3.926) <= 0.0011)
})

test('invalid input and structures are rejected before calling the servlet', async () => {
  const { client, calls } = fakeQuickcheck('nicotine')
  const options = { quickcheck: client, url: 'https://example.test' }

  await assert.rejects(validateAssignments({ structure: { molfile: '' }, assignments: [] }, options), InvalidInputError)
  await assert.rejects(
    validateAssignments(
      { structure: { molfile: 'not a molfile' }, assignments: [{ nucleus: '13C', atoms: [1], shift: 10 }] },
      options,
    ),
    InvalidStructureError,
  )
  assert.equal(calls.length, 0)
})
