import type { Options } from 'yargs'
import type { Spectrum } from '@zakodium/nmrium-core'

/**
 * Supported experiment types
 */
export type Experiment = 'proton' | 'carbon' | 'cosy' | 'hsqc' | 'hmbc'

/**
 * Base interface that all engines must implement
 */
export interface Engine {
    /** Unique engine identifier (e.g., 'nmrdb.org') */
    id: string

    /** Human-readable name */
    name: string

    /** Short description */
    description: string

    /** List of supported experiment types */
    supportedSpectra: Experiment[]

    /** Command-line options specific to this engine */
    options: Record<string, Options>

    /** List of required option keys */
    requiredOptions: string[]

    /** 
     * Build the payload options for the API request
     * @param argv - Command line arguments
     * @returns Options object to send in the API payload
     */
    buildPayloadOptions(argv: any): any

    /**
     * Predict and generate spectra
     * This is the main entry point for prediction
     * @param structure - MOL file content
     * @param options - Command line options
     * @param color - Color for the spectrum
     * @returns Array of generated spectra
     */
    predict(
        structure: string,
        options: any,
    ): Promise<Spectrum[]>


    /**
     * Optional: Custom validation beyond required options
     * @param argv - Command line arguments
     * @returns true if valid, error message if invalid
     */
    validate?(argv: any): true | string
}