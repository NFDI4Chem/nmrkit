import type { Options } from 'yargs'
import type { Spectrum } from '@zakodium/nmrium-core'
import type {
    Predicted,
    PredictionBase1D,
    PredictionBase2D,
    PredictionOptionsByExperiment,
} from 'nmr-processing'
import {
    predict,

} from 'nmr-processing'
import { defineEngine } from '../registry'
import { Molecule } from 'openchemlib'
import { Experiment } from '../base'
import { checkFromTo } from './core/checkFromTo'
import { generated1DSpectrum } from './core/generated1DSpectrum'
import { generated2DSpectrum } from './core/generated2DSpectrum'



// ============================================================================
// TYPES
// ============================================================================


export type PredictedSpectraResult = Partial<Record<Experiment, PredictionBase1D | PredictionBase2D>>

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
// HELPER FUNCTIONS
// ============================================================================


export async function predictSpectra(
    molfile: string,
    predictedSpectra: Experiment[],
): Promise<Predicted> {
    const molecule = Molecule.fromMolfile(molfile);
    const predictOptions: Record<string, PredictionOptionsByExperiment> = {

    };
    for (const key in predictedSpectra) {
        if (!predictedSpectra[key]) continue;
        const experiment = key === 'proton' ? 'H' : key === 'carbon' ? 'C' : key;
        predictOptions[experiment] = {};
    }
    return predict(molecule, { predictOptions });
}



export function generateSpectra(
    predictedSpectra: PredictedSpectraResult,
    inputOptions: PredictionOptions,
    color: string,
): Spectrum[] {
    const options: PredictionOptions = structuredClone(inputOptions);

    checkFromTo(predictedSpectra, options);
    const spectra: Spectrum[] = [];
    for (const experiment in predictedSpectra) {
        if (options.spectra[experiment as Experiment]) {
            const spectrum = predictedSpectra[experiment as Experiment];
            switch (experiment) {
                case 'proton':
                case 'carbon': {
                    const datum = generated1DSpectrum({
                        spectrum,
                        options,
                        experiment,
                        color,
                    });
                    spectra.push(datum);
                    break;
                }
                case 'cosy':
                case 'hsqc':
                case 'hmbc': {
                    const datum = generated2DSpectrum({
                        spectrum: spectrum as PredictionBase2D,
                        options,
                        experiment,
                        color,
                    });
                    spectra.push(datum);
                    break;
                }
                default:
                    break;
            }
        }
    }
    return spectra;
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

    buildPayloadOptions(argv: any): PredictionOptions {
        const spectraObj: Record<Experiment, boolean> = {
            carbon: false,
            proton: false,
            cosy: false,
            hmbc: false,
            hsqc: false
        }
        for (const experiment of argv.spectra) {
            spectraObj[experiment as Experiment] = true
        }

        return {
            name: argv.name || '',
            frequency: argv.frequency || 400,
            '1d': {
                '1H': {
                    from: argv.protonFrom ?? -1,
                    to: argv.protonTo ?? 12,
                },
                '13C': {
                    from: argv.carbonFrom ?? -5,
                    to: argv.carbonTo ?? 220,
                },
                nbPoints: argv.nbPoints1d || 2 ** 17,
                lineWidth: argv.lineWidth || 1,
            },
            '2d': {
                nbPoints: {
                    x: argv.nbPoints2dX || 1024,
                    y: argv.nbPoints2dY || 1024,
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
            predictionOptions.spectra
        )

        return generateSpectra(
            spectra,
            predictionOptions,
            'red'
        )


    }
})