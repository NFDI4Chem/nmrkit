import type { Options } from 'yargs'
import type { Spectrum } from '@zakodium/nmrium-core'

/**
 * Supported experiment types
 */
export type Experiment = 'proton' | 'carbon' | 'cosy' | 'hsqc' | 'hmbc'

/**
 * Nucleus types used in NMR
 */
export type Nucleus = '1H' | '13C'

/**
 * Map from experiment name to nucleus
 */
export const experimentToNucleus: Record<string, Nucleus> = {
    proton: '1H',
    carbon: '13C',
}

/**
 * Base interface that all engines must implement
 */
export interface Engine {
    /** Unique engine identifier (e.g., 'nmrdb.org') */
    readonly id: string

    readonly name: string
    readonly description: string
    readonly supportedSpectra: readonly Experiment[]

    /** Command-line options specific to this engine */
    readonly options: Record<string, Options>

    /** List of required option keys */
    readonly requiredOptions: readonly string[]

    /**
     * Build the payload options for the API request
     * @param argv - Command line arguments
     * @returns Options object to send in the API payload
     */
    buildPayloadOptions(argv: Record<string, unknown>): any

    /**
     * Predict and generate spectra
     * This is the main entry point for prediction
     * @param structure - MOL file content
     * @param options - Command line options
     * @returns Array of generated spectra
     */
    predict(
        structure: string,
        options: Record<string, unknown>,
    ): Promise<Spectrum[]>

    /**
     * Optional: Custom validation beyond required options
     * @param argv - Command line arguments
     * @returns true if valid, error message if invalid
     */
    validate?(argv: Record<string, unknown>): true | string
}