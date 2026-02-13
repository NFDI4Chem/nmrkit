import type { Options } from 'yargs'
import { type Spectrum } from '@zakodium/nmrium-core'
import type { GenerateSpectrumOptions, ShiftsItem } from './core/generatePredictedSpectrumData'
import { generatePredictedSpectrumData } from './core/generatePredictedSpectrumData'
import { defineEngine } from '../registry'

// ============================================================================
// TYPES
// ============================================================================
import https from 'https'
import axios from 'axios'
import { Experiment } from '../base'
import { extractInfoFromSpectra } from './core/extractInfoFromSpectra'
import { SpectraTypeMapItem } from './core/spectraTypeMap'



interface NMRShiftOptions {
    id: number
    shifts: string
    solvent: string
    spectra: Experiment[]
}

interface PredictionResponseItem {
    id: number
    type: string
    statistics: {
        accept: number
        warning: number
        reject: number
        missing: number
        total: number
    }
    shifts: ShiftsItem[]
}

interface PredictionResponse {
    result: PredictionResponseItem[]
}


type PredictionArgs = NMRShiftOptions & GenerateSpectrumOptions


async function callPredict(structure: string, options: Omit<NMRShiftOptions, 'spectra'> & { experiments: SpectraTypeMapItem[] }) {

    const url = process.env['NMR_PREDICTION_URL']

    if (!url) {
        throw new Error('Environment variable NMR_PREDICTION_URL is not defined.')
    }

    try {
        new URL(url).toString()
    } catch {
        throw new Error(`Invalid URL in NMR_PREDICTION_URL: "${url}"`)
    }


    const { id, shifts, solvent, experiments } = options;


    const httpsAgent = new https.Agent({
        rejectUnauthorized: false,
    })

    const requests: Promise<axios.AxiosResponse<any, any, {}>>[] = [];

    for (const experimentInfo of experiments) {
        const { type } = experimentInfo;
        const payload: any = {
            inputs: [
                {
                    id,
                    type,
                    shifts,
                    solvent,
                },
            ],
            moltxt: structure.trimEnd().replaceAll(/\\n/g, '\n'),
        }
        // Axios POST request with httpsAgent
        const request = axios.post(url, payload, {
            headers: {
                'Content-Type': 'application/json',
            },
            httpsAgent,
        })
        requests.push(request);

    }

    const results = await Promise.all(requests);
    return results;
}


async function predictNMR(structure: string,
    options: PredictionArgs): Promise<Spectrum[]> {

    try {
        const {
            id = 1,
            shifts = '1',
            solvent = 'Dimethylsulphoxide-D6 (DMSO-D6, C2D6SO)',
            from,
            to,
            nbPoints = 2 ** 18, // 256K
            frequency = 400,
            lineWidth = 1,
            tolerance = 0.001,
            peakShape = 'lorentzian',
            spectra
        } = options


        const experiments = extractInfoFromSpectra(spectra);

        const results = await callPredict(structure, { id, shifts, solvent, experiments })

        const outputSpectra: Spectrum[] = []
        let index = 0;
        for (const result of results) {
            const responseResult: PredictionResponse = result.data
            for (const result of responseResult.result) {
                const nucleus = experiments[index];
                const name = crypto.randomUUID()
                const data = generatePredictedSpectrumData(result.shifts, {
                    from,
                    to,
                    nbPoints,
                    lineWidth,
                    frequency,
                    tolerance,
                    peakShape,
                })

                const info = {
                    isFid: false,
                    isComplex: false,
                    dimension: 1,
                    originFrequency: frequency,
                    baseFrequency: frequency,
                    pulseSequence: '',
                    solvent,
                    isFt: true,
                    name,
                    nucleus,
                }

                outputSpectra.push({
                    id: crypto.randomUUID(),
                    data,
                    info,
                } as unknown as Spectrum)
            }
            index++;

        }



        return outputSpectra;
    } catch (error) {
        console.error(
            'Error:',
            error instanceof Error ? error.message : String(error)
        )

        if (axios.isAxiosError(error) && error.response) {
            console.error('Response data:', error.response.data)
        } else if (error instanceof Error && error.cause) {
            console.error('Network Error:', error.cause)
        }
    }

    return [];
}

// ============================================================================
// ENGINE DEFINITION
// ============================================================================

export const nmrshiftEngine = defineEngine({
    id: 'nmrshift',
    name: 'NMRShift',
    description: 'NMRShift prediction engine',

    supportedSpectra: ['proton', 'carbon'],
    options: {
        // NMRShift specific options
        id: {
            alias: 'i',
            type: 'number',
            description: 'Input ID',
            default: 1,
        },
        shifts: {
            type: 'string',
            description: 'Chemical shifts',
            default: '1',
        },
        solvent: {
            type: 'string',
            description: 'NMR solvent',
            default: 'Dimethylsulphoxide-D6 (DMSO-D6, C2D6SO)',
            choices: [
                'Any',
                'Chloroform-D1 (CDCl3)',
                'Dimethylsulphoxide-D6 (DMSO-D6, C2D6SO)',
                'Methanol-D4 (CD3OD)',
                'Deuteriumoxide (D2O)',
                'Acetone-D6 ((CD3)2CO)',
                'TETRACHLORO-METHANE (CCl4)',
                'Pyridin-D5 (C5D5N)',
                'Benzene-D6 (C6D6)',
                'neat',
                'Tetrahydrofuran-D8 (THF-D8, C4D4O)',
            ],
        },

        // Spectrum generation options
        from: {
            type: 'number',
            description: 'From in (ppm) for spectrum generation',
        },
        to: {
            type: 'number',
            description: 'To in (ppm) for spectrum generation',
        },
        nbPoints: {
            type: 'number',
            description: 'Number of points for spectrum generation',
            default: 2 ** 18,
        },
        lineWidth: {
            type: 'number',
            description: 'Line width for spectrum generation',
            default: 1,
        },
        frequency: {
            type: 'number',
            description: 'NMR frequency (MHz) for spectrum generation',
            default: 400,
        },
        tolerance: {
            type: 'number',
            description: 'Tolerance for spectrum generation',
            default: 0.001,
        },
        peakShape: {
            alias: 'ps',
            type: 'string',
            description: 'Peak shape algorithm for spectrum generation',
            default: 'lorentzian',
            choices: ['gaussian', 'lorentzian'],
        },
    } as Record<string, Options>,

    requiredOptions: ['solvent'],

    buildPayloadOptions(argv: any): NMRShiftOptions {
        return argv;
    },

    predict(structure, options) {
        return predictNMR(structure, options)
    }
})