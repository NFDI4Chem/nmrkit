#!/usr/bin/env node
import yargs, { type Argv, type CommandModule, type Options } from 'yargs'
import { loadSpectrumFromURL, loadSpectrumFromFilePath } from './parse/prase-spectra'
import { generateSpectrumFromPublicationString } from './publication-string'
import { hideBin } from 'yargs/helpers'
import { parsePredictionCommand } from './prediction'

const usageMessage = `
Usage: nmr-cli  <command> [options]

Commands:
  parse-spectra                Parse a spectra file to NMRium file
  parse-publication-string     resurrect spectrum from the publication string 
  predict                      Predict spectrum from Mol 

Options for 'parse-spectra' command:
  -u, --url                File URL  
  -dir, --dir-path         Directory path  
  -s, --capture-snapshot   Capture snapshot  
  -p, --auto-processing    Automatic processing of spectrum (FID → FT spectra).
  -d, --auto-detection     Enable ranges and zones automatic detection.

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



Examples:
  nmr-cli  parse-spectra -u file-url -s                                   // Process spectra files from a URL and capture an image for the spectra
  nmr-cli  parse-spectra -dir directory-path -s                             // process a spectra files from a directory and capture an image for the spectra
  nmr-cli  parse-spectra -u file-url                                      // Process spectra files from a URL 
  nmr-cli  parse-spectra -dir directory-path                                // Process spectra files from a directory 
  nmr-cli  parse-publication-string "your publication string"
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

    const { u, dir } = argv;
    // Handle parsing the spectra file logic based on argv options
    if (u) {
      loadSpectrumFromURL({ u, ...argv }).then(result => {
        console.log(JSON.stringify(result))
      })
    }


    if (dir) {
      loadSpectrumFromFilePath({ dir, ...argv }).then(result => {
        console.log(JSON.stringify(result))
      })
    }

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

yargs(hideBin(process.argv))
  .usage(usageMessage)
  .command(parseFileCommand)
  .command(parsePublicationCommand)
  .command(parsePredictionCommand)
  .showHelpOnFail(true)
  .help()
  .parse()
