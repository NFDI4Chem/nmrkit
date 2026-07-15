import { PredictionOptions } from "../nmrdb.engine"

export function getSpectralWidth(experiment: string, options: PredictionOptions) {
    const formTo = options['1d']

    switch (experiment) {
        case 'cosy': {
            const { from, to } = formTo['1H']
            const diff = to - from
            return [diff, diff]
        }
        case 'hsqc':
        case 'hmbc': {
            const proton = formTo['1H']
            const carbon = formTo['13C']
            const protonDiff = proton.to - proton.from
            const carbonDiff = carbon.to - carbon.from
            return [protonDiff, carbonDiff]
        }
        default:
            return []
    }
}