import { peaksToXY } from 'nmr-processing'
import { CURRENT_EXPORT_VERSION } from '@zakodium/nmrium-core'
import type { NMRPeak1D } from '@zakodium/nmr-types'
import { castToArray } from './utilities/castToArray'

interface PeakInput {
  x: number
  y?: number
  width?: number
}

interface PeaksToNMRiumOptions {
  nucleus?: string
  solvent?: string
  frequency?: number
  from?: number
  to?: number
  nbPoints?: number
}

interface PeaksToNMRiumInput {
  peaks: PeakInput[]
  options?: PeaksToNMRiumOptions
}

function generateNMRiumFromPeaks(input: PeaksToNMRiumInput) {
  const { peaks, options = {} } = input
  const {
    nucleus = '1H',
    solvent = '',
    frequency = 400,
    from,
    to,
    nbPoints = 131072,
  } = options

  if (!peaks || peaks.length === 0) {
    throw new Error('Peaks array is empty or not provided')
  }

  const defaultWidth = 1
  const nmrPeaks: NMRPeak1D[] = peaks.map((peak) => ({
    x: peak.x,
    y: peak.y ?? 1,
    width: peak.width ?? defaultWidth,
  }))

  const xyOptions: Parameters<typeof peaksToXY>[1] = {
    frequency,
    nbPoints,
    ...(from !== undefined && { from }),
    ...(to !== undefined && { to }),
  }

  const { x, y } = peaksToXY(nmrPeaks, xyOptions)

  const info = {
    isFid: false,
    isComplex: false,
    dimension: 1,
    nucleus,
    originFrequency: frequency,
    baseFrequency: frequency,
    pulseSequence: '',
    solvent,
    isFt: true,
    name: '',
  }

  const spectrum = {
    id: crypto.randomUUID(),
    data: { x: castToArray(x), im: undefined, re: castToArray(y) },
    info,
  }

  return { data: { spectra: [spectrum] }, version: CURRENT_EXPORT_VERSION }
}

export { generateNMRiumFromPeaks }
export type { PeaksToNMRiumInput, PeakInput, PeaksToNMRiumOptions }
