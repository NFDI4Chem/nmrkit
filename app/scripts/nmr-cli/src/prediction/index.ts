import { Argv, CommandModule } from 'yargs'
import { readFileSync, existsSync, writeFileSync } from 'fs'
import { CURRENT_EXPORT_VERSION } from '@zakodium/nmrium-core'
import { engineRegistry } from './engines'
import type { Experiment } from './engines/base'

// ============================================================================
// STRUCTURE INPUT HANDLING
// ============================================================================

/**
 * Resolves structure input from multiple sources:
 * 1. File path (if file exists)
 * 2. Stdin (if --stdin flag is used)
 * 3. Inline MOL content (as fallback)
 */
function resolveStructureInput(options: {
    structure?: string
    stdin?: boolean
    file?: string
}): string {
    // Priority 1: Explicit file flag
    if (options.file) {
        if (!existsSync(options.file)) {
            throw new Error(`File not found: ${options.file}`)
        }
        return readFileSync(options.file, 'utf-8')
    }

    // Priority 2: Explicit stdin flag
    if (options.stdin) {
        return readStdinSync()
    }

    // Priority 3: Structure argument (-s flag) - ALWAYS treat as inline content
    if (options.structure) {
        // Do NOT check existsSync here - just return it as inline MOL content
        return options.structure.trimEnd().replaceAll(/\\n/g, '\n')
    }

    throw new Error('No structure input provided. Use --file, --stdin, or -s')
}

/**
 * Synchronously read from stdin
 * This works because yargs has already parsed args, so stdin is available
 */
function readStdinSync(): string {
    try {
        // File descriptor 0 is stdin
        return readFileSync(0, 'utf-8')
    } catch (error) {
        throw new Error('Failed to read from stdin. Is data being piped?')
    }
}

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
    // Option 1: File path (most explicit)
    file: {
        alias: 'f',
        type: 'string',
        description: 'Path to MOL file',
        conflicts: ['stdin', 'structure'],
    },
    // Option 2: Stdin flag
    stdin: {
        type: 'boolean',
        description: 'Read structure from stdin',
        conflicts: ['file', 'structure'],
    },
    // Option 3: Structure argument (inline MOL content only)
    structure: {
        alias: 's',
        type: 'string',
        description: 'Inline MOL content (use --file for file paths)',
        conflicts: ['file', 'stdin'],
    },
    output: {
        alias: 'o',
        type: 'string',
        description: 'Output file path (default: stdout)',
    }
} as const

// ============================================================================
// VALIDATION
// ============================================================================

function validateEngineOptions(argv: Record<string, unknown>): void {
    const engineId = argv.engine as string
    const spectra = argv.spectra as string[]
    const engine = engineRegistry.get(engineId)

    if (!engine) {
        const available = engineRegistry.getIds().join(', ')
        throw new Error(`Unknown engine "${engineId}". Available engines: ${available}`)
    }

    const unsupportedSpectra = spectra.filter(
        (s) => !engine.supportedSpectra.includes(s as Experiment),
    )

    if (unsupportedSpectra.length > 0) {
        throw new Error(
            `Engine "${engineId}" does not support: ${unsupportedSpectra.join(', ')}.\n` +
            `Supported spectra: ${engine.supportedSpectra.join(', ')}`,
        )
    }

    const missing = engine.requiredOptions.filter((opt) => !argv[opt])
    if (missing.length > 0) {
        throw new Error(
            `Engine "${engineId}" requires: ${missing.join(', ')}\n` +
            `Usage: --${missing.join(' --')}`,
        )
    }

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

async function predictNMR(options: Record<string, unknown>): Promise<string> {
    const engineId = options.engine as string
    const engine = engineRegistry.get(engineId)!

    // Resolve structure from input
    const structure = resolveStructureInput({
        structure: options.structure as string | undefined,
        stdin: options.stdin as boolean | undefined,
        file: options.file as string | undefined,
    })

    // DEBUG LOGGING
    console.error('[DEBUG] Received structure:', structure ? `${structure.length} chars` : 'undefined')
    console.error('[DEBUG] Structure type:', typeof structure)
    console.error('[DEBUG] Structure preview:', structure?.substring(0, 100))

    // Validate structure is not empty
    if (!structure || !structure.trim()) {
        throw new Error('Structure input is empty or undefined')
    }

    // Run prediction
    const spectraResults = await engine.predict(structure, options)

    // Build NMRium output
    const nmrium = {
        data: { spectra: spectraResults },
        version: CURRENT_EXPORT_VERSION,
    }
    const output = JSON.stringify(nmrium)

    // Handle output destination
    if (options.output) {
        const outputPath = options.output as string
        writeFileSync(outputPath, output, 'utf-8')
        console.error(`Results written to ${outputPath}`)
    }

    return output
}

// ============================================================================
// COMMAND MODULE
// ============================================================================

export const parsePredictionCommand: CommandModule<{}, Record<string, unknown>> = {
    command: ['predict', 'p'],
    describe: 'Predict NMR spectrum from mol text',
    builder: (yargs: Argv): Argv => {
        let y = yargs.options(commonOptions)

        for (const engine of engineRegistry.getAll()) {
            y = y.options(engine.options) as Argv<any>
        }

        return y
            .check((argv) => {
                // Ensure at least one input method is provided
                if (!argv.file && !argv.stdin && !argv.structure) {
                    throw new Error(
                        'Must provide structure input via --file, --stdin, or -s'
                    )
                }
                return true
            })
            .example(
                '$0 predict -e myengine --spectra proton -f molecule.mol',
                'Predict from file path'
            )
            .example(
                '$0 predict -e myengine --spectra proton --stdin < molecule.mol',
                'Predict from stdin (redirect)'
            )
            .example(
                'cat molecule.mol | $0 predict -e myengine --spectra proton --stdin',
                'Predict from stdin (pipe)'
            )
            .example(
                '$0 predict -e myengine --spectra proton -s "\\n MOL content..."',
                'Predict using -s with inline MOL content'
            )
            .example(
                '$0 predict -e myengine --spectra proton -f mol.mol -o results.json',
                'Save output to file'
            )
    },
    handler: async (argv) => {
        // DEBUG: See ALL arguments
        console.error('[DEBUG] Full argv:', JSON.stringify(argv, null, 2))
        console.error('[DEBUG] argv.structure exists?', argv.structure !== undefined)
        console.error('[DEBUG] argv.engine:', argv.engine)
        console.error('[DEBUG] argv.spectra:', argv.spectra)

        try {
            validateEngineOptions(argv)
            const output = await predictNMR(argv)
            console.log(output)
        } catch (error) {
            console.error('Error:', error instanceof Error ? error.message : String(error))
            console.error('Error stack:', error instanceof Error ? error.stack : '')
            process.exit(1)
        }

    },
}