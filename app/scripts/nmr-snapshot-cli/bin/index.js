#!/usr/bin/env node
const { join, isAbsolute } = require("path");
const yargs = require("yargs");
const loader = require("nmr-load-save");
const fileUtils = require("filelist-utils");
const playwright = require('playwright');

const usageMessage = "Usage: nmr-snapshot-cli -u <url> or -p <path>"

const options = yargs
  .usage(usageMessage)
  .option("u", { alias: "url", describe: "File URL", type: "string", nargs: 1 })
  .option("p", { alias: "path", describe: "Directory path", type: "string", nargs: 1 }).showHelpOnFail();


const PARSING_OPTIONS = {
  onLoadProcessing: { autoProcessing: true },
  sourceSelector: { general: { dataSelection: 'preferFT' } },
}


function generateNMRiumURL() {
  const baseURL = process.env['BASE_NMRIUM_URL'];
  const url = new URL(baseURL)
  url.searchParams.append('workspace', "embedded")
  return url.toString()
}

async function getSpectraViewAsBase64(nmriumState) {
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


async function captureSpectrumSnapshotFromURL(url) {
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
    nmriumState
  } = await loader.read(fileCollection, PARSING_OPTIONS);


  return getSpectraViewAsBase64(nmriumState);
}


async function captureSpectrumSnapshotFromFilePath(path) {
  const dirPath = isAbsolute(path) ? path : join(process.cwd(), path)

  const fileCollection = await fileUtils.fileCollectionFromPath(dirPath, {});

  const {
    nmriumState
  } = await loader.read(fileCollection, PARSING_OPTIONS);
  return getSpectraViewAsBase64(nmriumState);
}


const parameters = options.argv;

if (parameters.u && parameters.p) {
  options.showHelp();
} else {

  if (parameters.u) {
    captureSpectrumSnapshotFromURL(parameters.u).then((result) => {
      console.log(JSON.stringify(result))
    })

  }

  if (parameters.p) {
    captureSpectrumSnapshotFromFilePath(parameters.p).then((result) => {
      console.log(JSON.stringify(result))
    })
  }

}





