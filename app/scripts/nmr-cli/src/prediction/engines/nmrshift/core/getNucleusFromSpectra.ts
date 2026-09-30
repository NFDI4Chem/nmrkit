import { Experiment } from '../../base.js'
import { spectraTypeMap } from './spectraTypeMap.js'

export function getNucleusFromSpectra(spectra: Experiment[]): string {
  const nuclei = new Set<string>()
  for (const spectrum of spectra) {
    const entry = spectraTypeMap[spectrum]
    if (entry) {
      nuclei.add(entry.nucleus)
    }
  }
  return nuclei.size > 0 ? [...nuclei].join(',') : '1H'
}
