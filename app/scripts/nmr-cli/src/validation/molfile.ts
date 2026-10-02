export class InvalidStructureError extends Error {}

export interface PreparedStructure {
  /** V2000 molfile without explicit hydrogens, heavy atoms in original order. */
  molfile: string
  heavyAtomCount: number
  /** 1-based prepared index -> 1-based original index (position 0 unused). */
  toOriginal: number[]
  /** 1-based original heavy-atom index -> 1-based prepared index. */
  fromOriginal: Map<number, number>
  /** 1-based original explicit-H index -> 1-based original parent index. */
  explicitHydrogenParents: Map<number, number>
  /** Element symbols by 1-based original index. */
  symbols: Map<number, string>
}

const RENUMBERED_PROPERTIES = ['M  CHG', 'M  ISO', 'M  RAD']

/**
 * Removes explicit hydrogens from a V2000 molfile without touching the order
 * of the remaining atoms.
 *
 * The nmrshiftdb quickcheck servlet reports heavy atoms by their molfile
 * position, but an explicit H shifts that numbering and makes predictions for
 * later atoms fail, so every structure is sent H-free and mapped back.
 */
export function prepareStructure(molfile: string): PreparedStructure {
  const lines = molfile.replace(/\r\n?/g, '\n').split('\n')

  if (lines.length < 4) {
    throw new InvalidStructureError('Molfile is too short')
  }

  const countsLine = lines[3]
  if (countsLine.includes('V3000')) {
    throw new InvalidStructureError('Only V2000 molfiles are supported')
  }

  const atomCount = Number.parseInt(countsLine.slice(0, 3), 10)
  const bondCount = Number.parseInt(countsLine.slice(3, 6), 10)
  if (!Number.isFinite(atomCount) || !Number.isFinite(bondCount) || atomCount < 1) {
    throw new InvalidStructureError('Molfile counts line is invalid')
  }

  const atomLines = lines.slice(4, 4 + atomCount)
  const bondLines = lines.slice(4 + atomCount, 4 + atomCount + bondCount)
  if (atomLines.length !== atomCount || bondLines.length !== bondCount) {
    throw new InvalidStructureError('Molfile atom or bond block is truncated')
  }

  const symbols = new Map<number, string>()
  atomLines.forEach((line, index) => {
    symbols.set(index + 1, line.slice(31, 34).trim())
  })

  const isExplicitHydrogen = (index: number) => symbols.get(index) === 'H'

  const toOriginal: number[] = [0]
  const fromOriginal = new Map<number, number>()
  for (let index = 1; index <= atomCount; index++) {
    if (isExplicitHydrogen(index)) continue
    fromOriginal.set(index, toOriginal.length)
    toOriginal.push(index)
  }

  const explicitHydrogenParents = new Map<number, number>()
  const keptBonds: string[] = []
  for (const line of bondLines) {
    const first = Number.parseInt(line.slice(0, 3), 10)
    const second = Number.parseInt(line.slice(3, 6), 10)
    const firstIsH = isExplicitHydrogen(first)
    const secondIsH = isExplicitHydrogen(second)

    if (firstIsH || secondIsH) {
      if (firstIsH && !secondIsH) explicitHydrogenParents.set(first, second)
      if (secondIsH && !firstIsH) explicitHydrogenParents.set(second, first)
      continue
    }

    const newFirst = fromOriginal.get(first)
    const newSecond = fromOriginal.get(second)
    if (newFirst === undefined || newSecond === undefined) {
      throw new InvalidStructureError(`Bond references unknown atom: ${line.trim()}`)
    }
    keptBonds.push(pad3(newFirst) + pad3(newSecond) + line.slice(6))
  }

  const heavyAtomCount = toOriginal.length - 1
  const properties = lines
    .slice(4 + atomCount + bondCount)
    .filter((line) => RENUMBERED_PROPERTIES.some((prefix) => line.startsWith(prefix)))
    .map((line) => renumberPropertyLine(line, fromOriginal))
    .filter((line): line is string => line !== null)

  const output = [
    ...lines.slice(0, 3),
    pad3(heavyAtomCount) + pad3(keptBonds.length) + countsLine.slice(6),
    ...toOriginal.slice(1).map((original) => atomLines[original - 1]),
    ...keptBonds,
    ...properties,
    'M  END',
    '',
  ].join('\n')

  return {
    molfile: output,
    heavyAtomCount,
    toOriginal,
    fromOriginal,
    explicitHydrogenParents,
    symbols,
  }
}

function pad3(value: number): string {
  return String(value).padStart(3, ' ')
}

function renumberPropertyLine(line: string, fromOriginal: Map<number, number>): string | null {
  const prefix = line.slice(0, 6)
  const entries = line.slice(9).trim().split(/\s+/).map(Number)
  const pairs: [number, number][] = []
  for (let i = 0; i + 1 < entries.length; i += 2) {
    const atom = fromOriginal.get(entries[i])
    if (atom !== undefined) pairs.push([atom, entries[i + 1]])
  }
  if (pairs.length === 0) return null

  return (
    prefix +
    pad3(pairs.length) +
    pairs.map(([atom, value]) => ' ' + pad3(atom) + ' ' + pad3(value)).join('')
  )
}
