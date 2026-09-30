import {
  calculateRelativeFrequency,
  PredictionBase2D,
  signals2DToZ,
} from 'nmr-processing'
import { generateName } from './generateName.js'
import { getSpectralWidth } from './getSpectralWidth.js'
import { initiateDatum2D } from '../../../../parse/data/data2d/initiateDatum2D.js'
import { adjustAlpha } from '../../../../utilities/adjustAlpha.js'
import { mapZones } from './mapZones.js'
import { PredictionOptions } from '../nmrdb.engine.js'

export function generated2DSpectrum(params: {
  options: PredictionOptions
  spectrum: PredictionBase2D
  experiment: string
  color: string
}) {
  const { spectrum, options, experiment, color } = params
  const { signals, zones, nuclei } = spectrum
  const xOption = (options['1d'] as any)[nuclei[0]]
  const yOption = (options['1d'] as any)[nuclei[1]]

  const width = nuclei[0] === nuclei[1] ? 0.02 : { x: 0.02, y: 0.2133 }
  const frequency = calculateRelativeFrequency(nuclei, options.frequency)

  const minMaxContent = signals2DToZ(signals, {
    from: { x: xOption.from, y: yOption.from },
    to: { x: xOption.to, y: yOption.to },
    nbPoints: {
      x: options['2d'].nbPoints.x,
      y: options['2d'].nbPoints.y,
    },
    width,
    factor: 3,
  })

  const SpectrumName = generateName(options.name, {
    frequency,
    experiment,
  })

  const spectralWidth = getSpectralWidth(experiment, options)
  const datum = initiateDatum2D({
    data: { rr: { ...minMaxContent, noise: 0.01 } },
    display: {
      positiveColor: color,
      negativeColor: adjustAlpha(color, 40),
    },
    info: {
      name: SpectrumName,
      title: SpectrumName,
      nucleus: nuclei,
      originFrequency: frequency,
      baseFrequency: frequency,
      pulseSequence: 'prediction',
      spectralWidth,
      experiment,
    },
  })

  datum.zones.values = mapZones(zones)
  return datum
}
