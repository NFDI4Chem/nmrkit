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

type RequiredKey<T, K extends keyof T> = Omit<T, K> & Required<Pick<T, K>>;

const parsingOptions: ParsingOptions = {
  onLoadProcessing: { autoProcessing: true },
  sourceSelector: { general: { dataSelection: 'preferFT' } },
  experimentalFeatures: true
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

async function loadSpectrumFromURL(options: RequiredKey<FileOptionsArgs, 'u'>) {
  const { u: url, s: enableSnapshot = false, p: autoProcessing = false, d: autoDetection = false } = options;

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
  const {
    data, version
  } = nmriumState;


  if (data) {
    processSpectra(data, { autoDetection, autoProcessing });
  }

  let images: Snapshot[] = []

  if (enableSnapshot) {
    images = await captureSpectraViewAsBase64({ data, version })
  }

  return { data, version, images }
}

async function loadSpectrumFromFilePath(options: RequiredKey<FileOptionsArgs, 'dir'>) {
  const { dir: path, s: enableSnapshot = false, p: autoProcessing = false, d: autoDetection = false } = options;

  const dirPath = isAbsolute(path) ? path : join(process.cwd(), path)

  const fileCollection = await FileCollection.fromPath(dirPath, {
    unzip: { zipExtensions: ['zip', 'nmredata'] },
  })

  const {
    nmriumState: { data, version },
  } = await core.read(fileCollection, parsingOptions)


  if (data) {
    processSpectra(data, { autoDetection, autoProcessing })
  }

  let images: Snapshot[] = []

  if (enableSnapshot) {
    images = await captureSpectraViewAsBase64({ data, version })
  }

  return { data, version, images }
}

export { loadSpectrumFromFilePath, loadSpectrumFromURL }
