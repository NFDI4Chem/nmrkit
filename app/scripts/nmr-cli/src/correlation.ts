import { buildCorrelationData } from 'nmr-correlation'
import type { Options as CorrelationOptions, Spectra } from 'nmr-correlation'
import { FifoLogger } from 'fifo-logger'
import {
  buildWebSource,
  core,
  parsingOptions,
  processSpectra,
} from './parse/prase-spectra'

// Default tolerances
const DEFAULT_TOLERANCE_H = 0.02
const DEFAULT_TOLERANCE_C = 0.25

export interface CorrelationInput {
  url: string
  mf: string
  toleranceH?: number
  toleranceC?: number
}

function resolveTolerance(value: number | undefined, fallback: number): number {
  return value === undefined || Number.isNaN(value) ? fallback : value
}

export async function generateCorrelationData(input: CorrelationInput) {
  const { url, mf, toleranceH, toleranceC } = input
  const logger = new FifoLogger()

  const source = buildWebSource(url)

  const { state } = await core.readFromWebSource(source, {
    ...parsingOptions,
    logger,
  })

  const spectraBeforeProcessing = state.data ? [...state.data.spectra] : []

  if (state.data) {
    processSpectra(
      state.data,
      { autoProcessing: true, autoDetection: true },
      logger
    )
  }

  // processSpectra replaces a spectrum's array slot with a new object only
  // when it successfully parses it; on failure it leaves the original raw
  // object in place (see its catch block) instead of removing it. Compare
  // by reference against the pre-processing snapshot to filter those out,
  // so buildCorrelationData never sees a spectrum it can't actually read.
  // Note: a pre-existing bug (see https://github.com/NFDI4Chem/nmrkit/issues/139)
  // currently makes every spectrum fail this step, so real cross-spectrum correlation links are untested here.
  const spectra = (state.data?.spectra ?? []).filter(
    (spectrum, index) => spectrum !== spectraBeforeProcessing[index]
  )

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
