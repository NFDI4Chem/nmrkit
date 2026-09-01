import { isSpectrum2D, type NmriumState } from '@zakodium/nmrium-core';
import type { FifoLogger } from 'fifo-logger';
import { runWithConcurrency, CPU_CONCURRENCY, SNAPSHOT_CONCURRENCY } from './run-concurrency';
import { SpectrumWorker } from './spectrum-worker';
import { BrowserManager, SnapshotLane } from './browser-manager';
import { Snapshot } from './spectrum-snapshot';
import { toMessage } from './utility/toMessage';
import { SpectrumPipelineOptions } from './type/SpectrumPipelineOptions';
import { ParsedSpectrum } from './type/ParsedSpectrum';
import { initiateDatum2D } from './data/data2d/initiateDatum2D';
import { initiateDatum1D } from './data/data1D/initiateDatum1D';

function getDurationMs(start: number): number {
    return Date.now() - start;
}

function generateNMRiumURL(): string {
    const baseURL = process.env['BASE_NMRIUM_URL'] || '';
    const url = new URL(baseURL);
    url.searchParams.append('workspace', 'embedded');
    return url.toString();
}

// Each spectrum flows through the same stages — process -> detect -> snapshot
// — but stages now run across a fixed number of concurrent "lanes" instead
// of one spectrum at a time. Every lane owns its own worker thread (CPU
// stages) and, if snapshots are enabled, its own persistent browser page.
//
// Every stage logs `durationMs` alongside its existing pass/fail log, so a
// run's logs can be aggregated afterwards to see where time actually goes
// (parse vs. process vs. detect vs. snapshot) instead of guessing.
export async function runSpectraPipeline(
    nmriumState: Partial<NmriumState>,
    options: SpectrumPipelineOptions,
    logger: FifoLogger
): Promise<Snapshot[]> {
    const data = nmriumState.data;
    const { version } = nmriumState;
    if (!data) return [];

    const parsed = parseSpectraList(data.spectra, logger);

    // Write the parsed set back so serialization later only sees spectra that
    // actually parsed successfully.
    data.spectra = parsed.map((d) => d.spectrum);
    const indexById = new Map(parsed.map((d, i) => [d.id, i]));

    const { autoProcessing, autoDetection, enableSnapshot } = options;
    const snapshots: Snapshot[] = [];
    if (parsed.length === 0) return snapshots;

    const concurrency = enableSnapshot ? SNAPSHOT_CONCURRENCY : CPU_CONCURRENCY;
    const browserManager = enableSnapshot ? new BrowserManager() : null;
    const url = enableSnapshot ? generateNMRiumURL() : '';

    const spectrumWorkers: SpectrumWorker[] = [];
    const snapshotLanes: SnapshotLane[] = [];

    const getWorker = (laneIndex: number): SpectrumWorker => {
        if (!spectrumWorkers[laneIndex]) spectrumWorkers[laneIndex] = new SpectrumWorker();
        return spectrumWorkers[laneIndex];
    };
    const getSnapshotLane = (laneIndex: number): SnapshotLane => {
        if (!snapshotLanes[laneIndex]) snapshotLanes[laneIndex] = new SnapshotLane(browserManager!, url);
        return snapshotLanes[laneIndex];
    };

    try {
        await runWithConcurrency(parsed, concurrency, async ({ id, spectrum: initial }, laneIndex) => {
            const spectrumWorker = getWorker(laneIndex);
            let spectrum = initial;

            if (autoProcessing) {
                const start = Date.now();
                try {
                    spectrum = await spectrumWorker.run('process', spectrum);
                    logger.info({ id, stage: 'processing', durationMs: getDurationMs(start) }, `Processed spectrum: ${id}`);
                } catch (e) {
                    logger.error(
                        { id, stage: 'processing', durationMs: getDurationMs(start), details: toMessage(e) },
                        `Failed to process spectrum: ${id}`
                    );
                }
            }

            if (autoDetection && spectrum.info.isFt) {
                const start = Date.now();
                try {
                    spectrum = await spectrumWorker.run('detect', spectrum);
                    logger.info({ id, stage: 'detection', durationMs: getDurationMs(start) }, `Detected peaks for spectrum: ${id}`);
                } catch (e) {
                    logger.error(
                        { id, stage: 'detection', durationMs: getDurationMs(start), details: toMessage(e) },
                        `Failed to detect peaks for spectrum: ${id}`
                    );
                }
            }

            data.spectra[indexById.get(id)!] = spectrum;

            if (enableSnapshot) {
                const lane = getSnapshotLane(laneIndex);
                // Timing + retry for this stage live inside SnapshotLane.capture
                // itself, since a retry needs its own per-attempt timing.
                const snapshot = await lane.capture(spectrum, id, version, spectrumWorker, logger);
                snapshots.push(snapshot);
            }
        });
    } finally {
        await Promise.all(spectrumWorkers.filter(Boolean).map((w) => w.terminate()));
        await Promise.all(snapshotLanes.filter(Boolean).map((l) => l.dispose()));
        await browserManager?.closeAll();
    }

    return snapshots;
}







function parseSpectraList(rawSpectra: any[], logger: FifoLogger): ParsedSpectrum[] {
    const parsed: ParsedSpectrum[] = [];
    for (const inputSpectrum of rawSpectra) {
        const id = inputSpectrum.id;
        const start = Date.now();
        try {
            const spectrum = isSpectrum2D(inputSpectrum)
                ? initiateDatum2D(inputSpectrum)
                : initiateDatum1D(inputSpectrum);
            logger.info({ id, stage: 'parsing', durationMs: Date.now() - start }, `Parsed spectrum: ${id}`);
            parsed.push({ id, spectrum });
        } catch (e) {
            logger.error(
                { id, stage: 'parsing', durationMs: Date.now() - start, details: toMessage(e) },
                `Failed to parse spectrum: ${id}`
            );
        }
    }
    return parsed;
}
