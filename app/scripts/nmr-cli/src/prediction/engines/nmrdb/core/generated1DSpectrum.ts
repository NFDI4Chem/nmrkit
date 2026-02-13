import { getRelativeFrequency, mapRanges, signalsToRanges, signalsToXY, updateIntegralsRelativeValues } from "nmr-processing"
import { generateName } from "./generateName"
import { initiateDatum1D } from "../../../../parse/data/data1D/initiateDatum1D"
import { PredictionOptions } from "../nmrdb.engine"

export function generated1DSpectrum(params: {
    options: PredictionOptions
    spectrum: any
    experiment: string
    color: string
}) {
    const { spectrum, options, experiment, color } = params
    const { signals, joinedSignals, nucleus } = spectrum

    const {
        name,
        '1d': { nbPoints, lineWidth },
        frequency: freq,
    } = options

    const SpectrumName = generateName(name, { frequency: freq, experiment })
    const frequency = getRelativeFrequency(nucleus, {
        frequency: freq,
        nucleus,
    })

    const { x, y } = signalsToXY(signals, {
        ...(options['1d'] as any)[nucleus],
        frequency,
        nbPoints,
        lineWidth,
    })

    const first = x[0] ?? 0
    const last = x.at(-1) ?? 0
    const getFreqOffset = (freq: any) => {
        return (first + last) * freq * 0.5
    }

    const datum = initiateDatum1D(
        {
            data: { x, im: null, re: y },
            display: { color },
            info: {
                nucleus,
                originFrequency: frequency,
                baseFrequency: frequency,
                frequencyOffset: Array.isArray(frequency)
                    ? frequency.map(getFreqOffset)
                    : getFreqOffset(frequency),
                pulseSequence: 'prediction',
                spectralWidth: Math.abs(first - last),
                solvent: '',
                experiment,
                isFt: true,
                name: SpectrumName,
                title: SpectrumName,
            },
        },
        {},
    )

    datum.ranges.values = mapRanges(
        signalsToRanges(joinedSignals, { frequency }),
        datum,
    )
    updateIntegralsRelativeValues(datum)

    return datum
}

