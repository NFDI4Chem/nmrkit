#!/usr/bin/env node
const yargs = require("yargs");
const { loadSpectrumFromURL, loadSpectrumFromFilePath } = require("./prase-spectra");


const usageMessage = `
Usage: nmr-cli  <command> [options]

Commands:
  parse-spectra                Parse a spectra file to NMRium file

Options for 'parse-spectra' command:
  -u, --url           File URL
  -p, --path          Directory path
  -s, --capture-snapshot   Capture snapshot

Examples:
  nmr-cli  parse-spectra -u file-url -s                                   // Process spectra files from a URL and capture an image for the spectra
  nmr-cli  parse-spectra -p directory-path -s                             // process a spectra files from a directory and capture an image for the spectra
  nmr-cli  parse-spectra -u file-url                                      // Process spectra files from a URL 
  nmr-cli  parse-spectra -p directory-path                                // Process spectra files from a directory 
`;


// Define options for parsing a file
const fileOptions = {
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
};


// Define the 'parse-file' command
const parseFileCommand = {
  command: ['parse-spectra', 'ps'],
  describe: 'Parse a spectra file to NMRium file',
  builder: (yargs) => {
    return yargs.options(fileOptions).conflicts('u', 'p');
  },
  handler: (argv) => {
    // Handle parsing the spectra file logic based on argv options
    console.log(argv)
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

// Set up Yargs with the two commands
yargs
  .usage(usageMessage)
  .command(parseFileCommand)
  .showHelpOnFail(true)
  .help()
  .argv;




