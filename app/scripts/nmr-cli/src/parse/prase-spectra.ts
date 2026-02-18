import { join, isAbsolute } from 'path'
import { NmriumData, ParsingOptions, type NmriumState } from '@zakodium/nmrium-core'
import init from '@zakodium/nmrium-core-plugins'
import playwright from 'playwright'
import { FileCollection } from 'file-collection'
import { FileOptionsArgs } from '..'
import { isSpectrum2D } from './data/data2d/isSpectrum2D'
import { initiateDatum2D } from './data/data2d/initiateDatum2D'
import { initiateDatum1D } from './data/data1D/initiateDatum1D'
import { detectZones } from './data/data2d/detectZones'
import { detectRanges } from './data/data1D/detectRanges'
import { Filters1DManager, Filters2DManager } from 'nmr-processing'
import yargs from 'yargs'
import { createWriteStream } from 'fs'
import { JsonStreamStringify } from 'json-stream-stringify';

type RequiredKey<T, K extends keyof T> = Omit<T, K> & Required<Pick<T, K>>;

const parsingOptions: ParsingOptions = {
  onLoadProcessing: { autoProcessing: true },
  selector: { general: { dataSelection: 'preferFT' } },
  experimentalFeatures: true,
};

interface Snapshot {
  image: string
  id: string
}

const core = init()

function generateNMRiumURL() {
  const baseURL = process.env['BASE_NMRIUM_URL'] || ''
  const url = new URL(baseURL)
  url.searchParams.append('workspace', 'embedded')
  return url.toString()
}

async function captureSpectraViewAsBase64(nmriumState: Partial<NmriumState>) {
  const { data: { spectra } = { spectra: [] }, version } = nmriumState
  const browser = await playwright.chromium.launch()
  const context = await browser.newContext(
    playwright.devices['Desktop Chrome HiDPI']
  )
  const page = await context.newPage()

  const url = generateNMRiumURL()

  await page.goto(url)

  await page.locator('text=Loading').waitFor({ state: 'hidden' })

  let snapshots: Snapshot[] = []

  for (const spectrum of spectra || []) {
    const spectrumObject = {
      version,
      data: {
        spectra: [{ ...spectrum }],
      },
    }

    // convert typed array to array
    const stringObject = JSON.stringify(
      spectrumObject,
      (key, value: unknown) => {
        return ArrayBuffer.isView(value)
          ? Array.from(value as unknown as Iterable<unknown>)
          : value
      }
    )

    // load the spectrum into NMRium using the custom event
    await page.evaluate(
      `
        window.postMessage({ type: "nmr-wrapper:load", data:{data: ${stringObject},type:"nmrium"}}, '*');
        `
    )

    //wait for NMRium process and load spectra
    await page.locator('text=Loading').waitFor({ state: 'hidden' })

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

  return snapshots

}

interface ProcessSpectraOptions {
  autoDetection: boolean; autoProcessing: boolean;
}

function processSpectra(data: NmriumData, options: ProcessSpectraOptions) {

  const { autoDetection = false, autoProcessing = false } = options

  for (let index = 0; index < data.spectra.length; index++) {
    const inputSpectrum = data.spectra[index]
    const is2D = isSpectrum2D(inputSpectrum);
    const spectrum = is2D ? initiateDatum2D(inputSpectrum) : initiateDatum1D(inputSpectrum);

    if (autoProcessing) {
      isSpectrum2D(spectrum) ? Filters2DManager.reapplyFilters(spectrum) : Filters1DManager.reapplyFilters(spectrum)
    }

    if (autoDetection && spectrum.info.isFt) {
      isSpectrum2D(spectrum) ? detectZones(spectrum) : detectRanges(spectrum);
    }

    data.spectra[index] = spectrum;
  }


}

function outputResult(result: any, outputPath?: string) {
  const stream = new JsonStreamStringify(result);

  if (outputPath) {
    const writeStream = createWriteStream(outputPath);
    stream.pipe(writeStream);
    writeStream.on('finish', () => {
      process.stderr.write(`Output written to: ${outputPath}\n`);
    });
  } else {
    stream.pipe(process.stdout);
  }
}

async function processAndSerialize(
  nmriumState: Partial<NmriumState>,
  options: FileOptionsArgs
) {
  const { s: enableSnapshot = false, p: autoProcessing = false, d: autoDetection = false, o, r } = options;

  if (nmriumState.data) {
    processSpectra(nmriumState.data, { autoDetection, autoProcessing });
  }

  const images: Snapshot[] = enableSnapshot
    ? await captureSpectraViewAsBase64(nmriumState)
    : [];

  const { data, version } = core.serializeNmriumState(
    nmriumState as NmriumState,
    { includeData: r ? 'rawData' : 'dataSource', },

  );

  // include the meta and info object in case of serialize as dataSource
  const spectra: any = data?.spectra || [];
  if (!r) {
    for (let i = 0; i < spectra.length; i++) {
      const { info = {}, meta = {} } = nmriumState.data?.spectra[i] || {};
      spectra[i] = { ...spectra[i], info, meta }
    }
  }

  outputResult({ data, version, images }, o);
}

async function loadSpectrumFromURL(options: RequiredKey<FileOptionsArgs, 'u'>) {
  const { u: url } = options;

  const { pathname: relativePath, origin: baseURL } = new URL(url)
  const source = {
    entries: [
      {
        relativePath,
      },
    ],
    baseURL,
  }

  const [nmriumState] = await core.readFromWebSource(source, parsingOptions);

  processAndSerialize(nmriumState, options)

}

async function loadSpectrumFromFilePath(options: RequiredKey<FileOptionsArgs, 'dir'>) {
  const { dir: path } = options;

  const dirPath = isAbsolute(path) ? path : join(process.cwd(), path)

  const fileCollection = await FileCollection.fromPath(dirPath, {
    unzip: { zipExtensions: ['zip', 'nmredata'] },
  })

  const {
    nmriumState
  } = await core.read(fileCollection, parsingOptions)

  processAndSerialize(nmriumState, options)

}


function parseSpectra(argv: yargs.ArgumentsCamelCase<FileOptionsArgs>
) {

  const { u, dir } = argv;
  // Handle parsing the spectra file logic based on argv options
  if (u) {
    loadSpectrumFromURL({ u, ...argv });
  }


  if (dir) {
    loadSpectrumFromFilePath({ dir, ...argv });
  }



}




export { loadSpectrumFromFilePath, loadSpectrumFromURL, parseSpectra }
