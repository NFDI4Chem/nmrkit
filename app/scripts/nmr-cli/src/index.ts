#!/usr/bin/env node
import yargs, { type Argv, type CommandModule, type Options } from 'yargs'
import { parseSpectra } from './parse/prase-spectra'
import { generateSpectrumFromPublicationString } from './publication-string'
import { generateNMRiumFromPeaks } from './peaks-to-nmrium'
import type { PeaksToNMRiumInput } from './peaks-to-nmrium'
import { hideBin } from 'yargs/helpers'
import { parsePredictionCommand } from './prediction'
import { readFileSync } from 'fs'
import { IncludeData } from '@zakodium/nmrium-core'

const usageMessage = `
Usage: nmr-cli  <command> [options]

Commands:
  parse-spectra                Parse a spectra file to NMRium file
  parse-publication-string     resurrect spectrum from the publication string 
  predict                      Predict spectrum from Mol 
  peaks-to-nmrium              Convert a peak list to NMRium object

Options for 'parse-spectra' command:
  -u, --url                File URL  
  -dir, --dir-path         Directory path  
  -s, --capture-snapshot   Capture snapshot  
  -p, --auto-processing    Automatic processing of spectrum (FID → FT spectra).
  -d, --auto-detection     Enable ranges and zones automatic detection.
  -o, --output             Output file path (optional)
  -r, --raw-data           Include raw data in the output instead of data source
  
Arguments for 'parse-publication-string' command:
  publicationString   Publication string
 
Options for 'predict' command:

Common options:
  -e, --engine        Prediction engine (required)                choices: ["nmrdb.org", "nmrshift"]
      --spectra       Spectra types to predict (required)         choices: ["proton", "carbon", "cosy", "hsqc", "hmbc"]
  -s, --structure     MOL file content (structure) (required)

nmrdb.org engine options:
      --name          Compound name (default: "")
      --frequency     NMR frequency (MHz) (default: 400)
      --protonFrom    Proton (1H) from in ppm (default: -1)
      --protonTo      Proton (1H) to in ppm (default: 12)
      --carbonFrom    Carbon (13C) from in ppm (default: -5)
      --carbonTo      Carbon (13C) to in ppm (default: 220)
      --nbPoints1d    1D number of points (default: 131072)
      --lineWidth     1D line width (default: 1)
      --nbPoints2dX   2D spectrum X-axis points (default: 1024)
      --nbPoints2dY   2D spectrum Y-axis points (default: 1024)
      --autoExtendRange  Auto extend range (default: true)

nmrshift engine options:
  -i, --id            Input ID (default: 1)
      --shifts        Chemical shifts (default: "1")
      --solvent       NMR solvent (default: "Dimethylsulphoxide-D6 (DMSO-D6, C2D6SO)")
                      choices: ["Any", "Chloroform-D1 (CDCl3)", "Dimethylsulphoxide-D6 (DMSO-D6, C2D6SO)",
                                "Methanol-D4 (CD3OD)", "Deuteriumoxide (D2O)", "Acetone-D6 ((CD3)2CO)",
                                "TETRACHLORO-METHANE (CCl4)", "Pyridin-D5 (C5D5N)", "Benzene-D6 (C6D6)",
                                "neat", "Tetrahydrofuran-D8 (THF-D8, C4D4O)"]
      --from          From in (ppm) for spectrum generation
      --to            To in (ppm) for spectrum generation
      --nbPoints      Number of points (default: 1024)
      --lineWidth     Line width (default: 1)
      --frequency     NMR frequency (MHz) (default: 400)
      --tolerance     Tolerance to group peaks with close shift (default: 0.001)
  -ps,--peakShape     Peak shape algorithm (default: "lorentzian") choices: ["gaussian", "lorentzian"]



Arguments for 'peaks-to-nmrium' command:
  Reads JSON from stdin with the following structure:
  {
    "peaks": [{ "x": 7.26, "y": 1, "width": 1 }, ...],
    "options": {
      "nucleus": "1H",          (default: "1H")
      "solvent": "",            (default: "")
      "frequency": 400,         (default: 400)
      "from": -1,               (optional, auto-computed from peaks)
      "to": 12,                 (optional, auto-computed from peaks)
      "nbPoints": 131072        (default: 131072)
    }
  }

Examples:
  nmr-cli  parse-spectra -u file-url -s                                   // Process spectra files from a URL and capture an image for the spectra
  nmr-cli  parse-spectra -dir directory-path -s                             // process a spectra files from a directory and capture an image for the spectra
  nmr-cli  parse-spectra -u file-url                                      // Process spectra files from a URL 
  nmr-cli  parse-spectra -dir directory-path                                // Process spectra files from a directory 
  nmr-cli  parse-publication-string "your publication string"
  echo '{"peaks":[{"x":7.26},{"x":2.10}]}' | nmr-cli peaks-to-nmrium     // Convert peaks to NMRium object
`

export interface FileOptionsArgs {
  /**  
   * -u, --url  
   * File URL to load remote spectra or data.
   */
  u?: string;

  /**  
   * -dir, --dir-path  
   * Local directory path for file input or output.
   */
  dir?: string;

  /**  
   * -s, --capture-snapshot  
   * Capture a visual snapshot of the current state or spectrum.
   */
  s?: boolean;

  /**  
   * -p, --auto-processing  
   * Automatically process spectrum from FID to FT spectra.  
   * Mandatory when automatic detection (`--auto-detection`) is enabled.
   */
  p?: boolean;

  /**  
   * -d, --auto-detection  
   * Perform automatic ranges and zones detection.
   */
  d?: boolean;
  /**
   *   -o, --output      
   *   Output file path
   */
  o?: string;
  /**
   *  -r, --raw-data   
   *   Include raw data in the output, defaults to dataSource
   */
  r?: boolean;

}

// Define options for parsing a spectra file
const fileOptions: { [key in keyof FileOptionsArgs]: Options } = {
  u: {
    alias: 'url',
    describe: 'File URL',
    type: 'string',
    nargs: 1,
  },
  dir: {
    alias: 'dir-path',
    describe: 'Directory path',
    type: 'string',
    nargs: 1,
  },
  s: {
    alias: 'capture-snapshot',
    describe: 'Capture snapshot',
    type: 'boolean',
  },
  p: {
    alias: 'auto-processing',
    describe: 'Auto processing',
    type: 'boolean',
  },
  d: {
    alias: 'auto-detection',
    describe: 'Ranges and zones auto detection',
    type: 'boolean',
  },
  o: {
    alias: 'output',
    type: 'string',
    description: 'Output file path',
  },
  r: {
    alias: 'raw-data',
    type: 'boolean',
    default: false,
    description: 'Include raw data in the output (default: dataSource)',
  },
} as const

const parseFileCommand: CommandModule<{}, FileOptionsArgs> = {
  command: ['parse-spectra', 'ps'],
  describe: 'Parse a spectra file to NMRium file',
  builder: yargs => {
    return yargs
      .options(fileOptions)
      .conflicts('u', 'dir') as Argv<FileOptionsArgs>
  },
  handler: argv => {
    parseSpectra(argv)
  },
}

// Define the parse publication string command
const parsePublicationCommand: CommandModule = {
  command: ['parse-publication-string', 'pps'],
  describe: 'Parse a publication string',
  handler: argv => {
    const publicationString = argv._[1]
    // Handle parsing publication string
    if (typeof publicationString == 'string') {
      const nmriumObject =
        generateSpectrumFromPublicationString(publicationString)

      console.log(JSON.stringify(nmriumObject))
    }
  },
}

// Define the peaks-to-nmrium command
const peaksToNMRiumCommand: CommandModule = {
  command: ['peaks-to-nmrium', 'ptn'],
  describe: 'Convert a peak list to NMRium object (reads JSON from stdin)',
  handler: () => {
    try {
      const stdinData = readFileSync(0, 'utf-8')
      const input: PeaksToNMRiumInput = JSON.parse(stdinData)
      const nmriumObject = generateNMRiumFromPeaks(input)
      console.log(JSON.stringify(nmriumObject))
    } catch (error) {
      console.error(
        'Error:',
        error instanceof Error ? error.message : String(error),
      )
      process.exit(1)
    }
  },
}

yargs(hideBin(process.argv))
  .usage(usageMessage)
  .command(parseFileCommand)
  .command(parsePublicationCommand)
  .command(parsePredictionCommand)
  .command(peaksToNMRiumCommand)
  .showHelpOnFail(true)
  .help()
  .parse()
