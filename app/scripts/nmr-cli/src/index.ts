#!/usr/bin/env node
import yargs, { type Argv, type CommandModule, type Options, } from "yargs";
import { loadSpectrumFromURL, loadSpectrumFromFilePath } from "./prase-spectra";
import { generateSpectrumFromPublicationString } from "./publication-string";


const usageMessage = `
Usage: nmr-cli  <command> [options]

Commands:
  parse-spectra                Parse a spectra file to NMRium file
  parse-publication-string     resurrect spectrum from the publication string 

Options for 'parse-spectra' command:
  -u, --url           File URL
  -p, --path          Directory path
  -s, --capture-snapshot   Capture snapshot

Arguments for 'parse-publication-string' command:
  publicationString   Publication string

Examples:
  nmr-cli  parse-spectra -u file-url -s                                   // Process spectra files from a URL and capture an image for the spectra
  nmr-cli  parse-spectra -p directory-path -s                             // process a spectra files from a directory and capture an image for the spectra
  nmr-cli  parse-spectra -u file-url                                      // Process spectra files from a URL 
  nmr-cli  parse-spectra -p directory-path                                // Process spectra files from a directory 
  nmr-cli  parse-publication-string "your publication string"
`;

interface FileOptionsArgs {
  u?: string;
  p?: string;
  s?: boolean;
}


// Define options for parsing a spectra file
const fileOptions: { [key in keyof FileOptionsArgs]: Options } = {
  u: {
    alias: 'url',
    describe: 'File URL',
    type: 'string',
    nargs: 1,
  },
  p: {
    alias: 'path',
    describe: 'Directory path',
    type: 'string',
    nargs: 1,
  },
  s: {
    alias: 'capture-snapshot',
    describe: 'Capture snapshot',
    type: 'boolean',
  },
} as const;


const parseFileCommand: CommandModule<{}, FileOptionsArgs> = {
  command: ['parse-spectra', 'ps'],
  describe: 'Parse a spectra file to NMRium file',
  builder: (yargs) => {
    return yargs.options(fileOptions).conflicts('u', 'p') as Argv<FileOptionsArgs>;
  },
  handler: (argv) => {
    // Handle parsing the spectra file logic based on argv options
    if (argv?.u) {
      loadSpectrumFromURL(argv.u, argv.s).then((result) => {
        console.log(JSON.stringify(result))
      })

    }

    if (argv?.p) {
      loadSpectrumFromFilePath(argv.p, argv.s).then((result) => {
        console.log(JSON.stringify(result))
      })
    }
  },
};

// Define the parse publication string command
const parsePublicationCommand: CommandModule = {
  command: ['parse-publication-string', 'pps'],
  describe: 'Parse a publication string',
  handler: (argv) => {
    const publicationString = argv._[1];
    // Handle parsing publication string
    if (typeof publicationString == "string") {
      const nmriumObject = generateSpectrumFromPublicationString(publicationString);


      console.log(JSON.stringify(nmriumObject));
    }
  },
};

yargs
  .usage(usageMessage)
  .command(parseFileCommand)
  .command(parsePublicationCommand)
  .showHelpOnFail(true)
  .help()
  .parse();




