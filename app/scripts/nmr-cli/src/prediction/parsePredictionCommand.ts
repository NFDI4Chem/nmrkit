import { Argv, CommandModule, Options } from 'yargs'
import {
    generatePredictedSpectrumData,
    GenerateSpectrumOptions,
    ShiftsItem,
} from './generatePredictedSpectrumData'
import { v4 } from '@lukeed/uuid'
import { CURRENT_EXPORT_VERSION } from 'nmr-load-save'
import https from 'https'
import axios from 'axios'

interface PredictionParameters {
    molText: string
    id: number
    type: string
    shifts: string
    solvent: string
    nucleus: string
}

const predictionOptions: { [key in keyof GenerateSpectrumOptions]: Options } = {
    from: {
        type: 'number',
        description: 'From in (ppm)',
    },
    to: {
        type: 'number',
        description: 'To in (ppm)',
    },
    nbPoints: {
        type: 'number',
        description: 'Number of points',
        default: 1024,
    },
    lineWidth: {
        type: 'number',
        description: 'Line width',
        default: 1,
    },
    frequency: {
        type: 'number',
        description: 'NMR frequency (MHz)',
        default: 400,
    },
    tolerance: {
        type: 'number',
        description: 'Tolerance',
        default: 0.001,
    },
    peakShape: {
        alias: 'ps',
        type: 'string',
        description: 'Peak shape algorithm',
        default: 'lorentzian',
        choices: ['gaussian', 'lorentzian'],
    },
} as const

const nmrOptions: { [key in keyof PredictionParameters]: Options } = {
    id: {
        alias: 'i',
        type: 'number',
        description: 'Input ID',
        default: 1,
    },
    type: {
        alias: 't',
        type: 'string',
        description: 'NMR type',
        default: 'nmr;1H;1d',
        choices: ['nmr;1H;1d', 'nmr;13C;1d'],
    },
    shifts: {
        alias: 's',
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
    molText: {
        alias: 'm',
        type: 'string',
        description: 'MOL file content',
        requiresArg: true,
    },
    nucleus: {
        alias: 'n',
        type: 'string',
        description: 'Predicted nucleus',
        requiresArg: true,
        choices: ['1H', '13C'],
    },
} as const

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

async function predictNMR(options: PredictionArgs): Promise<void> {
    const url = process.env['NMR_PREDICTION_URL']

    if (!url) {
        throw new Error('Environment variable NMR_PREDICTION_URL is not defined.')
    }

    try {
        new URL(url).toString()
    } catch {
        throw new Error(`Invalid URL in NMR_PREDICTION_URL: "${url}"`)
    }

    try {
        const {
            id,
            type,
            shifts,
            solvent,
            from,
            to,
            nbPoints = 1024,
            frequency = 400,
            lineWidth = 1,
            tolerance = 0.001,
            molText,
            nucleus,
            peakShape = "lorentzian",
        } = options

        const payload: any = {
            inputs: [
                {
                    id,
                    type,
                    shifts,
                    solvent,
                },
            ],
            moltxt: molText.replaceAll(/\\n/g, '\n'),
        }

        const httpsAgent = new https.Agent({
            rejectUnauthorized: false,
        })

        // Axios POST request with httpsAgent
        const response = await axios.post(url, payload, {
            headers: {
                'Content-Type': 'application/json',
            },
            httpsAgent,
        })

        const responseResult: PredictionResponse = response.data
        const spectra = []

        for (const result of responseResult.result) {
            const name = v4()
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

            spectra.push({
                id: v4(),
                data,
                info,
            })
        }

        const nmrium = { data: { spectra }, version: CURRENT_EXPORT_VERSION }
        console.log(JSON.stringify(nmrium, null, 2))
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
}

type PredictionArgs = PredictionParameters & GenerateSpectrumOptions

// Define the prediction string command
export const parsePredictionCommand: CommandModule<{}, PredictionArgs> = {
    command: ['predict', 'p'],
    describe: 'Predict NMR spectrum from mol text',
    builder: (yargs: Argv<{}>): Argv<PredictionArgs> => {
        return yargs.options({
            ...nmrOptions,
            ...predictionOptions,
        }) as Argv<PredictionArgs>
    },
    handler: async argv => {
        await predictNMR(argv)
    },
}
