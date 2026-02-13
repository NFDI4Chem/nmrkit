import { xMinMaxValues } from "ml-spectra-processing"
import { Experiment } from "../../base"
import { isProton } from "../../../../utilities/isProton"
import { Prediction1D, Prediction2D } from "nmr-processing"
import { PredictedSpectraResult, PredictionOptions } from "../nmrdb.engine"

export function checkFromTo(
    predictedSpectra: PredictedSpectraResult,
    inputOptions: PredictionOptions,
) {
    const setFromTo = (inputOptions: any, nucleus: any, fromTo: any) => {
        inputOptions['1d'][nucleus].to = fromTo.to
        inputOptions['1d'][nucleus].from = fromTo.from
        if (fromTo.signalsOutOfRange) {
            signalsOutOfRange[nucleus] = true
        }
    }

    const { autoExtendRange, spectra } = inputOptions
    const signalsOutOfRange: Record<string, boolean> = {}

    for (const exp in predictedSpectra) {
        const experiment = exp as Experiment
        if (!spectra[experiment]) continue
        if (predictedSpectra[experiment]?.signals.length === 0) continue

        if (['carbon', 'proton'].includes(experiment)) {
            const spectrum = predictedSpectra[experiment] as Prediction1D
            const { signals, nucleus } = spectrum
            const { from, to } = (inputOptions['1d'] as any)[nucleus]
            const fromTo = getNewFromTo({
                deltas: signals.map((s) => s.delta),
                from,
                to,
                nucleus,
                autoExtendRange,
            })
            setFromTo(inputOptions, nucleus, fromTo)
        } else {
            const { signals, nuclei } = predictedSpectra[experiment] as Prediction2D
            for (const nucleus of nuclei) {
                const axis = isProton(nucleus) ? 'x' : 'y'
                const { from, to } = (inputOptions['1d'] as any)[nucleus]
                const fromTo = getNewFromTo({
                    deltas: signals.map((s) => s[axis].delta),
                    from,
                    to,
                    nucleus,
                    autoExtendRange,
                })
                setFromTo(inputOptions, nucleus, fromTo)
            }
        }
    }

    for (const nucleus of ['1H', '13C']) {
        if (signalsOutOfRange[nucleus]) {
            const { from, to } = (inputOptions['1d'] as any)[nucleus]
            if (autoExtendRange) {
                console.log(
                    `There are ${nucleus} signals out of the range, it was extended to ${from}-${to}.`,
                )
            } else {
                console.log(`There are ${nucleus} signals out of the range.`)
            }
        }
    }
}



function getNewFromTo(params: {
    deltas: number[]
    from: number
    to: number
    nucleus: string
    autoExtendRange: boolean
}) {
    const { deltas, nucleus, autoExtendRange } = params
    let { from, to } = params
    const { min, max } = xMinMaxValues(deltas)
    const signalsOutOfRange = from > min || to < max

    if (autoExtendRange && signalsOutOfRange) {
        const spread = isProton(nucleus) ? 0.2 : 2
        if (from > min) from = min - spread
        if (to < max) to = max + spread
    }

    return { from, to, signalsOutOfRange }
}