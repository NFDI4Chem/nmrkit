import https from 'https'
import axios from 'axios'

import type { QuickcheckClient, QuickcheckInput, QuickcheckResult } from './types'

export class QuickcheckUnavailableError extends Error {}

const DEFAULT_TIMEOUT_MS = 60_000

export function quickcheckUrl(): string {
  const url = process.env['NMR_PREDICTION_URL']
  if (!url) {
    throw new Error('Environment variable NMR_PREDICTION_URL is not defined.')
  }
  return url
}

/**
 * Calls the nmrshiftdb2 quickcheck servlet once for all nuclei.
 *
 * Certificate verification is disabled to match the existing nmrshift engine;
 * the servlet's certificate chain is not trusted by the container image.
 */
export function createQuickcheckClient(
  url: string,
  timeoutMs = DEFAULT_TIMEOUT_MS,
): QuickcheckClient {
  const httpsAgent = new https.Agent({ rejectUnauthorized: false })

  return async (molfile: string, inputs: QuickcheckInput[]): Promise<QuickcheckResult[]> => {
    try {
      const response = await axios.post<{ result: QuickcheckResult[] }>(
        url,
        { inputs, moltxt: molfile },
        { headers: { 'Content-Type': 'application/json' }, httpsAgent, timeout: timeoutMs },
      )
      if (!Array.isArray(response.data?.result)) {
        throw new QuickcheckUnavailableError('nmrshiftdb returned an unexpected response')
      }
      return response.data.result
    } catch (error) {
      if (error instanceof QuickcheckUnavailableError) throw error
      const message = axios.isAxiosError(error)
        ? `nmrshiftdb quickcheck failed: ${error.response?.status ?? error.code ?? error.message}`
        : `nmrshiftdb quickcheck failed: ${String(error)}`
      throw new QuickcheckUnavailableError(message)
    }
  }
}
