import playwright, {
    type Browser,
    type BrowserContext,
    type Page,
} from 'playwright'

import { SpectrumWorker } from './spectrum-worker'

export interface Snapshot {
    id: string
    image: string | null
}

const SNAPSHOT_LOADING_TIMEOUT_MS = 30_000

function generateNMRiumURL(): string {
    const baseURL = process.env.BASE_NMRIUM_URL

    if (!baseURL) {
        throw new Error(
            'BASE_NMRIUM_URL environment variable is not defined',
        )
    }

    const url = new URL(baseURL)
    url.searchParams.set('workspace', 'embedded')

    return url.toString()
}

async function waitForNMRium(page: Page): Promise<void> {
    await page.locator('text=Loading').waitFor({
        state: 'hidden',
        timeout: SNAPSHOT_LOADING_TIMEOUT_MS,
    })
}

export class SpectrumSnapshot {
    private browser: Browser | null = null
    private context: BrowserContext | null = null
    private page: Page | null = null

    async start(): Promise<void> {
        if (this.page) return

        this.browser = await playwright.firefox.launch()

        this.context = await this.browser.newContext(
            playwright.devices['Desktop Chrome HiDPI'],
        )

        this.page = await this.context.newPage()

        await this.page.goto(generateNMRiumURL())

        await waitForNMRium(this.page)
    }

    async capture(
        id: string,
        spectrum: any,
        version: unknown,
        worker: SpectrumWorker,
    ): Promise<Snapshot> {
        if (!this.page) {
            throw new Error(
                'SpectrumSnapshot has not been started',
            )
        }

        const stringObject = await worker.run(
            'serialize',
            spectrum,
            version,
        )

        /*
         * Parse the serialized data here rather than injecting
         * the JSON string into JavaScript source code.
         */
        const data = JSON.parse(stringObject)

        await this.page.evaluate((nmriumData) => {
            window.postMessage(
                {
                    type: 'nmr-wrapper:load',
                    data: {
                        data: nmriumData,
                        type: 'nmrium',
                    },
                },
                '*',
            )
        }, data)

        await waitForNMRium(this.page)

        const image = await this.page
            .locator('#nmrSVG .container')
            .screenshot()

        return {
            id,
            image: image.toString('base64'),
        }
    }

    async close(): Promise<void> {
        await this.context?.close().catch(() => { })
        await this.browser?.close().catch(() => { })

        this.page = null
        this.context = null
        this.browser = null
    }
}

