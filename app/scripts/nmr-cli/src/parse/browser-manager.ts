import playwright from 'playwright';
import type { FifoLogger } from 'fifo-logger';
import type { SpectrumWorker } from './spectrum-worker';
import { Spectrum } from '@zakodium/nmr-types';
import { Snapshot } from './spectrum-snapshot';
import { toMessage } from './utility/toMessage';

// How long to wait for the NMRium "Loading" indicator to appear/disappear
// before giving up on a single spectrum's snapshot. Prevents one stuck page
// from hanging the entire pipeline.
export const SNAPSHOT_LOADING_TIMEOUT_MS = 30_000;

// A single flaky failure (a slow spectrum tipping over the loading timeout,
// a one-off page hiccup) shouldn't permanently cost a spectrum its snapshot,
// so each capture gets one retry on a fresh page before giving up.
export const SNAPSHOT_MAX_ATTEMPTS = 2;

// Owns the single shared Firefox process. Multiple lanes each get their own
// context/page from it. `reset` is guarded so that if several lanes hit a
// dead browser at once, only the first one actually relaunches — the rest
// just see `current` has already changed and pick up the fresh instance.
export class BrowserManager {
    private current: Promise<playwright.Browser> | null = null;

    async get(): Promise<playwright.Browser> {
        if (!this.current) {
            this.current = playwright.firefox.launch();
        }
        return this.current;
    }

    async reset(stale: playwright.Browser): Promise<void> {
        if (this.current && (await this.current) === stale) {
            const toClose = this.current;
            this.current = null;
            await (await toClose).close().catch(() => { });
        }
    }

    async closeAll(): Promise<void> {
        if (this.current) {
            const toClose = this.current;
            this.current = null;
            await (await toClose).close().catch(() => { });
        }
    }
}

// One snapshot lane = one browser tab, reused across every spectrum it's
// assigned. Launching the browser/context happens once per lane; every
// spectrum after the first gets a `page.reload()` (not a new context or
// browser) before its `nmr-wrapper:load` message, so each snapshot starts
// from a genuinely empty NMRium instance instead of relying on `load`
// merging vs. replacing the previous spectrum's state. A reload of an
// already-booted SPA is far cheaper than relaunching the browser/context,
// so this keeps the speed win while removing the state-leak risk.
export class SnapshotLane {
    private context: playwright.BrowserContext | null = null;
    private page: playwright.Page | null = null;
    private hasLoadedSpectrum = false;

    constructor(private manager: BrowserManager, private url: string) { }

    private async ensurePage(): Promise<playwright.Page> {
        if (this.page && !this.page.isClosed()) {
            if (this.hasLoadedSpectrum) {
                await this.page.reload();
                await this.page.locator('text=Loading').waitFor({ state: 'hidden', timeout: SNAPSHOT_LOADING_TIMEOUT_MS });
            }
            return this.page;
        }
        const browser = await this.manager.get();
        this.context = await browser.newContext(playwright.devices['Desktop Chrome HiDPI']);
        this.page = await this.context.newPage();
        await this.page.goto(this.url);
        await this.page.locator('text=Loading').waitFor({ state: 'hidden', timeout: SNAPSHOT_LOADING_TIMEOUT_MS });
        this.hasLoadedSpectrum = false;
        return this.page;
    }

    // Discards this lane's page/context so the next attempt (or the next
    // spectrum, if we're giving up) starts from a clean page instead of
    // whatever broken state caused the failure. Only escalates to a full
    // browser relaunch if the browser process itself is gone.
    private async recover(): Promise<void> {
        await this.context?.close().catch(() => { });
        this.context = null;
        this.page = null;
        this.hasLoadedSpectrum = false;

        const browser = await this.manager.get();
        if (!browser.isConnected()) {
            await this.manager.reset(browser);
        }
    }

    private async attemptCapture(
        spectrum: Spectrum,
        version: unknown,
        spectrumWorker: SpectrumWorker
    ): Promise<string> {
        const page = await this.ensurePage();

        const stringObject = await spectrumWorker.run('serialize', spectrum, version);

        // Passed as a Playwright function argument rather than spliced into an
        // evaluated script string, so a backtick or `${...}` sequence anywhere
        // in the spectrum data can't break (or hijack) the script.
        await page.evaluate(
            ({ data }) => {
                window.postMessage({ type: 'nmr-wrapper:load', data: { data, type: 'nmrium' } }, '*');
            },
            { data: JSON.parse(stringObject) }
        );

        await page.locator('text=Loading').waitFor({ state: 'hidden', timeout: SNAPSHOT_LOADING_TIMEOUT_MS });

        const snapshot = await page.locator('#nmrSVG .container').screenshot();
        this.hasLoadedSpectrum = true;
        return snapshot.toString('base64');
    }

    async capture(
        spectrum: Spectrum,
        id: string,
        version: unknown,
        spectrumWorker: SpectrumWorker,
        logger: FifoLogger
    ): Promise<Snapshot> {
        let lastError: unknown;

        for (let attempt = 1; attempt <= SNAPSHOT_MAX_ATTEMPTS; attempt++) {
            const start = Date.now();
            try {
                const image = await this.attemptCapture(spectrum, version, spectrumWorker);
                logger.info(
                    { id, stage: 'snapshot', attempt, durationMs: Date.now() - start },
                    `Captured snapshot for spectrum: ${id}`
                );
                return { id, image };
            } catch (e) {
                lastError = e;
                logger.error(
                    { id, stage: 'snapshot', attempt, durationMs: Date.now() - start, details: toMessage(e) },
                    `Snapshot attempt ${attempt}/${SNAPSHOT_MAX_ATTEMPTS} failed for spectrum: ${id}`
                );
                await this.recover();
            }
        }

        logger.error(
            { id, stage: 'snapshot', details: toMessage(lastError) },
            `Giving up on snapshot for spectrum: ${id} after ${SNAPSHOT_MAX_ATTEMPTS} attempts`
        );
        return { id, image: null };
    }

    async dispose(): Promise<void> {
        await this.context?.close().catch(() => { });
        this.context = null;
        this.page = null;
    }
}