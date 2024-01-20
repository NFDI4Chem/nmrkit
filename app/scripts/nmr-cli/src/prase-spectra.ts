import { join, isAbsolute } from "path";
import { type NmriumState, read } from "nmr-load-save";
import { fileCollectionFromWebSource, fileCollectionFromPath } from "filelist-utils";
import playwright from 'playwright';


interface Snapshot {
    image: string,
    id: string;
}


function generateNMRiumURL() {
    const baseURL = process.env['BASE_NMRIUM_URL'] || '';
    const url = new URL(baseURL)
    url.searchParams.append('workspace', "embedded")
    return url.toString()
}


async function captureSpectraViewAsBase64(nmriumState: Partial<NmriumState>) {
    const { data: { spectra } = { spectra: [] }, version } = nmriumState;
    const browser = await playwright.chromium.launch()
    const context = await browser.newContext(playwright.devices['Desktop Chrome HiDPI'])
    const page = await context.newPage()

    const url = generateNMRiumURL()

    await page.goto(url)

    await page.locator('text=Loading').waitFor({ state: 'hidden' });

    let snapshots: Snapshot[] = []

    for (const spectrum of spectra || []) {
        const spectrumObject = {
            version,
            data: {
                spectra: [{ ...spectrum }],
            }

        }

        // convert typed array to array
        const stringObject = JSON.stringify(spectrumObject, (key, value: unknown) => {
            return ArrayBuffer.isView(value) ? Array.from(value as unknown as Iterable<unknown>) : value
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

async function loadSpectrumFromURL(url: string, enableSnapshot = false) {
    const { pathname: relativePath, origin: baseURL } = new URL(url);
    const source = {
        entries: [
            {
                relativePath,
            }
        ],
        baseURL
    };
    const fileCollection = await fileCollectionFromWebSource(source, {});

    const {
        nmriumState: { data, version },
    } = await read(fileCollection);

    let images: Snapshot[] = []

    if (enableSnapshot) {
        images = await captureSpectraViewAsBase64({ data, version });
    }


    return { data, version, images };
}


async function loadSpectrumFromFilePath(path: string, enableSnapshot = false) {
    const dirPath = isAbsolute(path) ? path : join(process.cwd(), path)

    const fileCollection = await fileCollectionFromPath(dirPath, {});

    const {
        nmriumState: { data, version }
    } = await read(fileCollection);

    let images: Snapshot[] = []

    if (enableSnapshot) {
        images = await captureSpectraViewAsBase64({ data, version });
    }


    return { data, version, images };
}


export {
    loadSpectrumFromFilePath,
    loadSpectrumFromURL
};

