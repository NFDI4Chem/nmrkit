import { buildCorrelationData } from 'nmr-correlation'
import type { Options as CorrelationOptions, Spectra } from 'nmr-correlation'
import { FifoLogger } from 'fifo-logger'
import type { NmriumState, Spectrum } from '@zakodium/nmrium-core'
import {
  buildWebSource,
  core,
  loadFileCollection,
  parsingOptions,
  processSpectra,
} from './parse/prase-spectra'
import { isSpectrum2D } from './parse/data/data2d/isSpectrum2D'

// Default tolerances confirmed by vcnainala on issue #66
const DEFAULT_TOLERANCE_H = 0.02
const DEFAULT_TOLERANCE_C = 0.25

export interface CorrelationInput {
  url?: string
  dir?: string
  mf: string
  toleranceH?: number
  toleranceC?: number
}

interface ReadSpectraOptions {
  url?: string
  dir?: string
}

async function readSpectra(
  options: ReadSpectraOptions,
  logger: FifoLogger
): Promise<Partial<NmriumState>> {
  const { url, dir } = options

  if (url) {
    const { state } = await core.readFromWebSource(buildWebSource(url), {
      ...parsingOptions,
      logger,
    })
    return state
  }

  if (dir) {
    const { state } = await core.read(await loadFileCollection(dir), {
      ...parsingOptions,
      logger,
    })
    return state
  }

  throw new Error('Either a spectra URL or a local directory path is required')
}

// buildCorrelationData needs detected ranges (1D) or zones (2D) to find
// correlations, so require isFt plus at least one detected range/zone.
// This also excludes spectra that failed to initiate or failed detection,
// since those never get ranges/zones populated either.
// Note: a pre-existing bug (see https://github.com/NFDI4Chem/nmrkit/issues/139)
// currently makes every spectrum fail initiation, so real cross-spectrum correlation links are untested here.
function filterSpectra(spectra: Spectrum[]): Spectrum[] {
  return spectra.filter(spectrum => {
    const { info } = spectrum
    if (info.isFt !== true) return false

    if (isSpectrum2D(spectrum)) {
      const { zones } = spectrum
      return zones.values.length > 0
    }

    const { ranges } = spectrum
    return ranges.values.length > 0
  })
}

function resolveTolerance(value: number | undefined, fallback: number): number {
  return value === undefined || Number.isNaN(value) ? fallback : value
}

export async function generateCorrelationData(input: CorrelationInput) {
  const { url, dir, mf, toleranceH, toleranceC } = input
  const logger = new FifoLogger()

  const state = await readSpectra({ url, dir }, logger)

  if (state.data) {
    processSpectra(
      state.data,
      { autoProcessing: true, autoDetection: true },
      logger
    )
  }

  const spectra = filterSpectra(state.data?.spectra ?? [])

  const options: CorrelationOptions = {
    mf,
    tolerance: {
      H: resolveTolerance(toleranceH, DEFAULT_TOLERANCE_H),
      C: resolveTolerance(toleranceC, DEFAULT_TOLERANCE_C),
    },
  }

  let correlationData
  try {
    correlationData = buildCorrelationData(spectra as Spectra, options)
  } catch (error) {
    throw new Error(
      `Failed to build correlation data: ${error instanceof Error ? error.message : String(error)}`
    )
  }

  return { ...correlationData, logs: logger.getLogs() }
}
