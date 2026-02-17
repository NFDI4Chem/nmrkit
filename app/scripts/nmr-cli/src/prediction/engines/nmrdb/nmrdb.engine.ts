import type { Options } from 'yargs'
import type { Spectrum } from '@zakodium/nmrium-core'
import type {
    Predicted,
    PredictionBase1D,
    PredictionBase2D,
    PredictionOptionsByExperiment,
} from 'nmr-processing'
import { predict } from 'nmr-processing'
import { Molecule } from 'openchemlib'

import { defineEngine } from '../registry'
import type { Experiment } from '../base'
import { checkFromTo } from './core/checkFromTo'
import { generated1DSpectrum } from './core/generated1DSpectrum'
import { generated2DSpectrum } from './core/generated2DSpectrum'
export type PredictedSpectraResult = Partial<
    Record<Experiment, PredictionBase1D | PredictionBase2D>
>

export interface PredictionOptions {
    name: string
    frequency: number
    '1d': {
        '1H': { from: number; to: number }
        '13C': { from: number; to: number }
        nbPoints: number
        lineWidth: number
    }
    '2d': {
        nbPoints: { x: number; y: number }
    }
    autoExtendRange: boolean
    spectra: Record<Experiment, boolean>
}

// ============================================================================
// Map experiment names to prediction keys
// ============================================================================

const experimentToPredictKey: Record<string, string> = {
    proton: 'H',
    carbon: 'C',
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

export async function predictSpectra(
    molfile: string,
    spectra: Record<Experiment, boolean>,
): Promise<Predicted> {
    const molecule = Molecule.fromMolfile(molfile)

    const predictOptions: Record<string, PredictionOptionsByExperiment> = {}
    for (const [experiment, enabled] of Object.entries(spectra)) {
        if (!enabled) continue
        const key = experimentToPredictKey[experiment] ?? experiment
        predictOptions[key] = {}
    }

    return predict(molecule, { predictOptions })
}

export function generateSpectra(
    predictedSpectra: PredictedSpectraResult,
    options: PredictionOptions,
    color: string,
): Spectrum[] {
    const clonedOptions = structuredClone(options)
    checkFromTo(predictedSpectra, clonedOptions)

    const spectra: Spectrum[] = []

    for (const [experiment, spectrum] of Object.entries(predictedSpectra)) {
        if (!clonedOptions.spectra[experiment as Experiment]) continue

        switch (experiment) {
            case 'proton':
            case 'carbon': {
                spectra.push(
                    generated1DSpectrum({ spectrum, options: clonedOptions, experiment, color }),
                )
                break
            }
            case 'cosy':
            case 'hsqc':
            case 'hmbc': {
                spectra.push(
                    generated2DSpectrum({
                        spectrum: spectrum as PredictionBase2D,
                        options: clonedOptions,
                        experiment,
                        color,
                    }),
                )
                break
            }
            default:
                break
        }
    }

    return spectra
}

// ============================================================================
// ENGINE DEFINITION
// ============================================================================

export const nmrdbEngine = defineEngine({
    id: 'nmrdb.org',
    name: 'NMRDB.org',
    description: 'NMRDB.org prediction engine with 1D and 2D NMR support',
    supportedSpectra: ['proton', 'carbon', 'cosy', 'hmbc', 'hsqc'],

    options: {
        name: {
            type: 'string',
            description: 'Compound name',
            default: '',
        },
        frequency: {
            type: 'number',
            description: 'NMR frequency (MHz)',
            default: 400,
        },
        protonFrom: {
            type: 'number',
            description: 'Proton (1H) from in ppm',
            default: -1,
        },
        protonTo: {
            type: 'number',
            description: 'Proton (1H) to in ppm',
            default: 12,
        },
        carbonFrom: {
            type: 'number',
            description: 'Carbon (13C) from in ppm',
            default: -5,
        },
        carbonTo: {
            type: 'number',
            description: 'Carbon (13C) to in ppm',
            default: 220,
        },
        nbPoints1d: {
            type: 'number',
            description: '1D number of points',
            default: 2 ** 17,
        },
        lineWidth: {
            type: 'number',
            description: '1D line width',
            default: 1,
        },
        nbPoints2dX: {
            type: 'number',
            description: '2D spectrum X-axis points',
            default: 1024,
        },
        nbPoints2dY: {
            type: 'number',
            description: '2D spectrum Y-axis points',
            default: 1024,
        },
        autoExtendRange: {
            type: 'boolean',
            description: 'Auto extend range',
            default: true,
        },
    } as Record<string, Options>,

    requiredOptions: [],

    buildPayloadOptions(argv: Record<string, unknown>): PredictionOptions {
        const spectraObj: Record<Experiment, boolean> = {
            carbon: false,
            proton: false,
            cosy: false,
            hmbc: false,
            hsqc: false,
        }

        for (const experiment of argv.spectra as string[]) {
            spectraObj[experiment as Experiment] = true
        }

        return {
            name: (argv.name as string) || '',
            frequency: (argv.frequency as number) || 400,
            '1d': {
                '1H': {
                    from: (argv.protonFrom as number) ?? -1,
                    to: (argv.protonTo as number) ?? 12,
                },
                '13C': {
                    from: (argv.carbonFrom as number) ?? -5,
                    to: (argv.carbonTo as number) ?? 220,
                },
                nbPoints: (argv.nbPoints1d as number) || 2 ** 17,
                lineWidth: (argv.lineWidth as number) || 1,
            },
            '2d': {
                nbPoints: {
                    x: (argv.nbPoints2dX as number) || 1024,
                    y: (argv.nbPoints2dY as number) || 1024,
                },
            },
            spectra: spectraObj,
            autoExtendRange: argv.autoExtendRange !== false,
        }
    },

    async predict(structure, options) {
        const predictionOptions = this.buildPayloadOptions(options)

        const { spectra } = await predictSpectra(
            structure,
            predictionOptions.spectra,
        )

        return generateSpectra(spectra, predictionOptions, 'red')
    },
})