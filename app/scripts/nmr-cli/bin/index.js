#!/usr/bin/env node
const { join, isAbsolute } = require("path");
const yargs = require("yargs");
const loader = require("nmr-load-save");
const fileUtils = require("filelist-utils");
const playwright = require('playwright');

const usageMessage = "Usage: nmr-cli -u <url> or -p <path> -s<Optional parameter to capture spectra snapshots>"


/**
 * How to Use the Command Line Tool:
 * Example 1: Process spectra files from a URL
 * Usage: nmr-cli -u https://example.com/file.zip
 * -------------------------------------------------------------------------
 * Example 2: process a spectra files from a directory
 * Usage: nmr-cli  -p /path/to/directory
 * -------------------------------------------------------------------------
 * you could also combine the above examples with an optional parameter to capturing a snapshot using the -s option
 * 
 */

const options = yargs
  .usage(usageMessage)
  .option("u", { alias: "url", describe: "File URL", type: "string", nargs: 1 })
  .option("p", { alias: "path", describe: "Directory path", type: "string", nargs: 1 })
  .option("s", { alias: "capture-snapshot", describe: "Capture snapshot", type: "boolean" }).showHelpOnFail();




function generateNMRiumURL() {
  const baseURL = process.env['BASE_NMRIUM_URL'];
  const url = new URL(baseURL)
  url.searchParams.append('workspace', "embedded")
  return url.toString()
}

async function captureSpectraViewAsBase64(nmriumState) {
  const { data: { spectra }, version } = nmriumState;
  const browser = await playwright.chromium.launch()
  const context = await browser.newContext(playwright.devices['Desktop Chrome HiDPI'])
  const page = await context.newPage()

  const url = generateNMRiumURL()

  await page.goto(url)

  await page.locator('text=Loading').waitFor({ state: 'hidden' });

  let snapshots = []

  for (const spectrum of spectra || []) {
    const spectrumObject = {
      version,
      data: {
        spectra: [{ ...spectrum }],
      }

    }

    // convert typed array to array
    const stringObject = JSON.stringify(spectrumObject, (key, value) => {
      return ArrayBuffer.isView(value) ? Array.from(value) : value
    })

    // load the spectrum into NMRium using the custom event
    await page.evaluate(
      `
      window.postMessage({ type: "nmr-wrapper:load", data:{data: ${stringObject},type:"nmrium"}}, '*');
      `
    )

    //wait for NMRium process and load spectra
    await page.locator('text=Loading').waitFor({ state: 'hidden' });

    // take a snapshot for the spectrum
    try {
      const snapshot = await page.locator('#nmrSVG .container').screenshot()

      snapshots.push({
        image: snapshot.toString('base64'),
        id: spectrum.id,
      })
    } catch (e) {
      console.log(e)
    }
  }

  await context.close()
  await browser.close()

  return snapshots;
}

async function loadSpectrumFromURL(url, enableSnapshot = false) {
  const { pathname: relativePath, origin: baseURL } = new URL(url);
  const source = {
    entries: [
      {
        relativePath,
      }
    ],
    baseURL
  };
  const fileCollection = await fileUtils.fileCollectionFromWebSource(source, {});

  const {
    nmriumState: { data, version },
  } = await loader.read(fileCollection);

  let images = []

  if (enableSnapshot) {
    images = await captureSpectraViewAsBase64({ data, version });
  }


  return { data, version, images };
}


async function loadSpectrumFromFilePath(path, enableSnapshot = false) {
  const dirPath = isAbsolute(path) ? path : join(process.cwd(), path)

  const fileCollection = await fileUtils.fileCollectionFromPath(dirPath, {});

  const {
    nmriumState: { data, version }
  } = await loader.read(fileCollection);

  let images = []

  if (enableSnapshot) {
    images = await captureSpectraViewAsBase64({ data, version });
  }


  return { data, version, images };
}


const parameters = options.argv;

if (parameters.u && parameters.p) {
  options.showHelp();
} else {

  if (parameters.u) {
    loadSpectrumFromURL(parameters.u, parameters.s).then((result) => {
      console.log(JSON.stringify(result))
    })

  }

  if (parameters.p) {
    loadSpectrumFromFilePath(parameters.p, parameters.s).then((result) => {
      console.log(JSON.stringify(result))
    })
  }

}





