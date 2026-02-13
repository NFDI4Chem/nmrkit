import type { Options } from 'yargs'
import type { Spectrum } from '@zakodium/nmrium-core'
import https from 'https'
import axios from 'axios'

import { defineEngine } from '../registry'
import type { Experiment } from '../base'
import type { GenerateSpectrumOptions, ShiftsItem } from './core/generatePredictedSpectrumData'
import { generatePredictedSpectrumData } from './core/generatePredictedSpectrumData'
import { extractInfoFromSpectra } from './core/extractInfoFromSpectra'
import type { SpectraTypeMapItem } from './core/spectraTypeMap'

interface NMRShiftPayload {
    id: number
    type: string
    shifts: string
    solvent: string
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

interface NMRShiftOptions {
    id: number
    shifts: string
    solvent: string
    spectra: Experiment[]
}

type PredictionArgs = NMRShiftOptions & GenerateSpectrumOptions

// ============================================================================
// HELPERS
// ============================================================================

function getBaseUrl(): string {
    const url = process.env['NMR_PREDICTION_URL']
    if (!url) {
        throw new Error('Environment variable NMR_PREDICTION_URL is not defined.')
    }
    try {
        new URL(url)
    } catch {
        throw new Error(`Invalid URL in NMR_PREDICTION_URL: "${url}"`)
    }
    return url
}

async function callPredict(
    structure: string,
    options: {
        id: number
        shifts: string
        solvent: string
        experiments: SpectraTypeMapItem[]
    },
): Promise<axios.AxiosResponse<PredictionResponse>[]> {
    const url = getBaseUrl()
    const { id, shifts, solvent, experiments } = options

    const httpsAgent = new https.Agent({ rejectUnauthorized: false })

    const requests = experiments.map((experimentInfo) => {
        const payload: NMRShiftPayload = {
            id,
            type: experimentInfo.type,
            shifts,
            solvent,
        }

        return axios.post<PredictionResponse>(url, {
            inputs: [payload],
            moltxt: structure,
        }, {
            headers: { 'Content-Type': 'application/json' },
            httpsAgent,
        })
    })

    return Promise.all(requests)
}

async function predictNMR(
    structure: string,
    options: PredictionArgs,
): Promise<Spectrum[]> {
    const {
        id = 1,
        shifts = '1',
        solvent = 'Dimethylsulphoxide-D6 (DMSO-D6, C2D6SO)',
        from,
        to,
        nbPoints = 1024,
        frequency = 400,
        lineWidth = 1,
        tolerance = 0.001,
        peakShape = 'lorentzian',
        spectra,
    } = options

    // Derive experiments (type + nucleus) from the --spectra list
    const experiments = extractInfoFromSpectra(spectra)

    if (experiments.length === 0) {
        throw new Error(
            `No supported experiments found for spectra [${spectra.join(', ')}]. ` +
            `Supported: proton, carbon.`,
        )
    }

    const results = await callPredict(structure, { id, shifts, solvent, experiments })

    const outputSpectra: Spectrum[] = []

    for (let i = 0; i < results.length; i++) {
        const response: PredictionResponse = results[i].data
        const experimentInfo = experiments[i]

        for (const item of response.result) {
            const data = generatePredictedSpectrumData(item.shifts, {
                from,
                to,
                nbPoints,
                lineWidth,
                frequency,
                tolerance,
                peakShape,
            })

            if (!data) continue

            const name = crypto.randomUUID()

            outputSpectra.push({
                id: crypto.randomUUID(),
                data,
                info: {
                    isFid: false,
                    isComplex: false,
                    dimension: 1,
                    originFrequency: frequency,
                    baseFrequency: frequency,
                    pulseSequence: '',
                    solvent,
                    isFt: true,
                    name,
                    nucleus: experimentInfo.nucleus,
                },
            } as unknown as Spectrum)
        }
    }

    return outputSpectra
}

// ============================================================================
// ENGINE DEFINITION
// ============================================================================

const SOLVENT_CHOICES = [
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
] as const

export const nmrshiftEngine = defineEngine({
    id: 'nmrshift',
    name: 'NMRShift',
    description: 'NMRShift prediction engine',

    supportedSpectra: ['proton', 'carbon'],

    options: {
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
            choices: SOLVENT_CHOICES,
        },
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
            default: 1024,
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
            description: 'Tolerance to group peaks with close shift',
            default: 0.001,
        },
        peakShape: {
            alias: 'ps',
            type: 'string',
            description: 'Peak shape algorithm',
            default: 'lorentzian',
            choices: ['gaussian', 'lorentzian'],
        },
    } as Record<string, Options>,

    requiredOptions: ['solvent'],

    buildPayloadOptions(argv: Record<string, unknown>): NMRShiftOptions {
        return {
            id: (argv.id as number) ?? 1,
            shifts: (argv.shifts as string) ?? '1',
            solvent: (argv.solvent as string) ?? 'Dimethylsulphoxide-D6 (DMSO-D6, C2D6SO)',
            spectra: argv.spectra as Experiment[],
        }
    },

    async predict(structure, options) {
        return predictNMR(structure, options as unknown as PredictionArgs)
    },
})