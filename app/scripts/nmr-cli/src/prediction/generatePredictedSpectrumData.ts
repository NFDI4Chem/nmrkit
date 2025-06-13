export interface ShiftsItem {
  atom: number
  prediction: number
  real: number
  diff: number
  status: 'accept' | 'missing' | string
  hoseCode: string
  spheres: number
}

interface LorentzianOptions {
  x: number
  fwhm: number
}

function lorentzian2(options: LorentzianOptions) {
  const { x, fwhm } = options
  return fwhm ** 2 / (4 * x ** 2 + fwhm ** 2)
}

interface GenerateSpectrumOptions {
  from?: number
  to?: number
  nbPoints?: number
  lineWidth?: number
  frequency?: number
  tolerance?: number
}

interface GroupItem {
  prediction: number
  count: number
  atoms: number[]
}

interface Data1D {
  x: number[]
  re: number[]
}

const getLorentzianFactor = (area = 0.9999) => {
  if (area >= 1) {
    throw new Error('area should be (0 - 1)')
  }
  const halfResidual = (1 - area) * 0.5
  const quantileFunction = (p: number) => Math.tan(Math.PI * (p - 0.5))
  return (
    (quantileFunction(1 - halfResidual) - quantileFunction(halfResidual)) / 2
  )
}

function groupEquivalentShifts(shifts: ShiftsItem[], tolerance = 0.001) {
  const groups: GroupItem[] = []

  for (const shift of shifts) {
    const match = groups.find(
      g => Math.abs(g.prediction - shift.prediction) < tolerance
    )
    if (match) {
      match.count += 1
      match.atoms.push(shift.atom)
    } else {
      groups.push({
        prediction: shift.prediction,
        count: 1,
        atoms: [shift.atom],
      })
    }
  }

  return groups
}

export function generatePredictedSpectrumData(
  shifts: ShiftsItem[],
  options: GenerateSpectrumOptions = {}
) {
  let { from, to } = options
  const {
    nbPoints = 10240,
    frequency = 400,
    lineWidth = 1,
    tolerance = 0.001,
  } = options

  if (!shifts || shifts.length === 0) return []

  const sortedShifts = shifts
    .slice(0)
    .sort((a, b) => a.prediction - b.prediction)

  // const acceptedShifts = sortedShifts.filter(shift =>
  //     shift.status === 'accept'
  // );

  const acceptedShifts = sortedShifts

  if (acceptedShifts.length === 0) return []
  from = from ?? acceptedShifts[0].prediction - 1
  to = to ?? (acceptedShifts.at(-1) as ShiftsItem).prediction + 1

  if (from >= to) {
    throw new Error("Invalid range: 'from' is greater or equal then 'to'")
  }

  const data: Data1D = { x: [], re: [] }
  const stepSize = (to - from) / (nbPoints - 1)
  const groupedShifts = groupEquivalentShifts(acceptedShifts, tolerance)

  console.log(groupedShifts)

  const limit = (lineWidth * getLorentzianFactor(0.99)) / frequency
  console.log(limit)
  for (let i = 0; i < nbPoints; i++) {
    const x = from + i * stepSize
    let intensity = 0
    for (const { prediction, count } of groupedShifts) {
      if (Math.abs(x - prediction) <= limit) {
        intensity +=
          lorentzian2({ x: x - prediction, fwhm: lineWidth / frequency }) *
          count
      }
    }
    if (intensity > 0) {
      data.x.push(x)
      data.re.push(intensity)
    }
  }

  return data
}
