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
  id: string;
  image: string | null;
  error: string | null;
}

const core = init()

function generateNMRiumURL() {
  const baseURL = process.env['BASE_NMRIUM_URL'] || ''
  const url = new URL(baseURL)
  url.searchParams.append('workspace', 'embedded')
  return url.toString()
}

async function launchBrowser() {
  return playwright.firefox.launch();
}

async function captureSpectraViewAsBase64(nmriumState: Partial<NmriumState>): Promise<Snapshot[]> {
  const { data: { spectra } = { spectra: [] }, version } = nmriumState;

  if (!spectra?.length) return [];

  const url = generateNMRiumURL();
  const snapshots: Snapshot[] = [];
  let browser = await launchBrowser();

  for (const spectrum of spectra) {
    let context = null;

    try {
      // recreate browser if it has crashed
      if (!browser.isConnected()) {
        browser = await launchBrowser();
      }

      context = await browser.newContext(playwright.devices['Desktop Chrome HiDPI']);
      const page = await context.newPage();

      await page.goto(url);
      await page.locator('text=Loading').waitFor({ state: 'hidden' });

      const stringObject = JSON.stringify(
        { version, data: { spectra: [{ ...spectrum }] } },
        (key, value: unknown) => ArrayBuffer.isView(value) ? Array.from(value as unknown as Iterable<unknown>) : value
      );

      await page.evaluate(`
        window.postMessage({ type: "nmr-wrapper:load", data: { data: ${stringObject}, type: "nmrium" } }, '*');
      `);

      await page.locator('text=Loading').waitFor({ state: 'hidden' });

      const snapshot = await page.locator('#nmrSVG .container').screenshot();
      snapshots.push({ id: spectrum.id, image: snapshot.toString('base64'), error: null });

    } catch (e) {
      snapshots.push({
        id: spectrum.id,
        image: null,
        error: e instanceof Error ? e.message : String(e),
      });

      // browser crashed — close and recreate for next spectrum
      await browser.close().catch(() => { });
      browser = await launchBrowser();

    } finally {
      await context?.close().catch(() => { });
    }
  }

  await browser.close().catch(() => { });
  return snapshots;
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
