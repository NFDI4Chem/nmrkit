import { join, isAbsolute } from 'path'
import { type NmriumState } from '@zakodium/nmrium-core'
import init from '@zakodium/nmrium-core-plugins'
import playwright from 'playwright'
import { FileCollection } from 'file-collection'

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

async function loadSpectrumFromURL(url: string, enableSnapshot = false) {
  const { pathname: relativePath, origin: baseURL } = new URL(url)
  const source = {
    entries: [
      {
        relativePath,
      },
    ],
    baseURL,
  }

  const [nmriumState] = await core.readFromWebSource(source);
  const {
    data, version
  } = nmriumState;

  let images: Snapshot[] = []

  if (enableSnapshot) {
    images = await captureSpectraViewAsBase64({ data, version })
  }

  return { data, version, images }
}

async function loadSpectrumFromFilePath(path: string, enableSnapshot = false) {
  const dirPath = isAbsolute(path) ? path : join(process.cwd(), path)

  const fileCollection = await FileCollection.fromPath(dirPath, {
    unzip: { zipExtensions: ['zip', 'nmredata'] },
  })

  const {
    nmriumState: { data, version },
  } = await core.read(fileCollection)

  let images: Snapshot[] = []

  if (enableSnapshot) {
    images = await captureSpectraViewAsBase64({ data, version })
  }

  return { data, version, images }
}

export { loadSpectrumFromFilePath, loadSpectrumFromURL }
