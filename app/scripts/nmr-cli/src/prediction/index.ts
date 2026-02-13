import { Argv, CommandModule } from 'yargs'
import { CURRENT_EXPORT_VERSION } from '@zakodium/nmrium-core'
import { engineRegistry } from './engines'
import { Experiment } from './engines/base'
// ============================================================================
// COMMON OPTIONS
// ============================================================================

const commonOptions = {
    engine: {
        alias: 'e',
        type: 'string',
        description: 'Prediction engine',
        demandOption: true,
        choices: engineRegistry.getIds(),
    },
    spectra: {
        type: 'array',
        description: 'Spectra types to predict',
        demandOption: true,
        choices: ['proton', 'carbon', 'cosy', 'hsqc', 'hmbc'],
    },
    structure: {
        alias: 's',
        type: 'string',
        description: 'MOL file content (structure)',
        requiresArg: true,
        demandOption: true,
    },
} as const

// ============================================================================
// VALIDATION
// ============================================================================

function validateEngineOptions(argv: any): void {
    const { engine: engineId, spectra } = argv
    const engine = engineRegistry.get(engineId)

    if (!engine) {
        throw new Error(
            `Unknown engine "${engineId}". Available engines: ${engineRegistry.getIds().join(', ')}`
        )
    }

    // Check if requested spectra are supported by the engine
    const unsupportedSpectra = spectra.filter(
        (spectrum: string) => !engine.supportedSpectra.includes(spectrum as Experiment)
    )

    if (unsupportedSpectra.length > 0) {
        throw new Error(
            `Engine "${engineId}" does not support the following spectra: ${unsupportedSpectra.join(', ')}\n\nSupported spectra for ${engineId}: ${engine.supportedSpectra.join(', ')}`
        )
    }

    // Check required options
    const missing = engine.requiredOptions.filter((option: string) => !argv[option])

    if (missing.length > 0) {
        throw new Error(
            `Engine "${engineId}" requires the following options: ${missing.join(', ')}\n\nUsage for ${engineId}:\n  --${missing.join(' --')}`
        )
    }

    // Custom validation if provided
    if (engine.validate) {
        const result = engine.validate(argv)
        if (result !== true) {
            throw new Error(result)
        }
    }
}

// ============================================================================
// MAIN PREDICTION FUNCTION
// ============================================================================

async function predictNMR(options: any): Promise<void> {
    const {
        engine: engineId,
        structure,
    } = options

    const engine = engineRegistry.get(engineId)!

    // Each engine handles its own prediction flow
    const spectraResults = await engine.predict(
        structure,
        options,
    )

    const nmrium = { data: { spectra: spectraResults }, version: CURRENT_EXPORT_VERSION }
    console.log(JSON.stringify(nmrium, null, 2))
}

// ============================================================================
// COMMAND MODULE
// ============================================================================

export const parsePredictionCommand: CommandModule<{}, any> = {
    command: ['predict', 'p'],
    describe: 'Predict NMR spectrum from mol text',
    builder: (yargs: Argv<any>): Argv<any> => {
        // Start with common options
        let yargsWithOptions = yargs.options(commonOptions)

        // Add all engine-specific options from registry
        for (const engine of engineRegistry.getAll()) {
            yargsWithOptions = yargsWithOptions.options(engine.options) as Argv<any>

        }

        return yargsWithOptions
    },
    handler: async (argv: any) => {
        try {
            // Validate engine-specific requirements
            validateEngineOptions(argv)
            await predictNMR(argv)
        } catch (error) {
            console.error(
                'Error:',
                error instanceof Error ? error.message : String(error)
            )

            process.exit(1)
        }
    },
}