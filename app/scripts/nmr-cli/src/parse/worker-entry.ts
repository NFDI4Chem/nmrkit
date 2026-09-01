import { parentPort } from 'node:worker_threads';
import { Filters1DManager, Filters2DManager } from 'nmr-processing';
import { detectZones } from './data/data2d/detectZones';
import { detectRanges } from './data/data1D/detectRanges';
import { Spectrum } from '@zakodium/nmr-types';
import { isSpectrum2D } from '@zakodium/nmrium-core';
import { WorkerRequest } from './type/WorkerRequest';
import { WorkerResponse } from './type/WorkerResponse';

if (!parentPort) {
    throw new Error('detectWorkerEntry must be run inside a worker_threads Worker');
}

// Both mutate the spectrum in place (same as the original single-threaded
// implementation) — the return is just for a uniform call signature.
async function runAutoProcessing(spectrum: Spectrum): Promise<Spectrum> {
    if (isSpectrum2D(spectrum)) {
        Filters2DManager.reapplyFilters(spectrum);
    } else {
        Filters1DManager.reapplyFilters(spectrum);
    }
    return spectrum;
}

async function runAutoDetection(spectrum: Spectrum): Promise<Spectrum> {
    if (isSpectrum2D(spectrum)) {
        detectZones(spectrum);
    } else {
        detectRanges(spectrum);
    }
    return spectrum;
}

// Spectra hold typed-array data (Float64Array etc.), which JSON.stringify
// otherwise mangles into `{0: ..., 1: ...}` objects instead of arrays — the
// replacer below converts any ArrayBuffer view to a plain array first. No
// nmrium-core-plugins init needed here: this is a plain JSON encoding of
// { version, data: { spectra: [spectrum] } }, matching exactly what the
// browser's `nmr-wrapper:load` message expects.
async function serializeSpectrum(spectrum: Spectrum, version: unknown): Promise<string> {
    return JSON.stringify(
        { version, data: { spectra: [spectrum] } },
        (_key, value: unknown) => (ArrayBuffer.isView(value) ? Array.from(value as unknown as Iterable<unknown>) : value)
    );
}

parentPort.on('message', async (msg: WorkerRequest) => {
    const { spectrum, task, version } = msg;
    const reply = (response: WorkerResponse) => parentPort!.postMessage(response);

    try {
        switch (task) {
            case 'process':
                reply({ spectrum: await runAutoProcessing(spectrum) });
                break;
            case 'detect':
                reply({ spectrum: await runAutoDetection(spectrum) });
                break;
            case 'serialize':
                reply({ stringObject: await serializeSpectrum(spectrum, version) });
                break;
            default: {
                const exhaustive: never = task;
                throw new Error(`Unknown task: ${exhaustive}`);
            }
        }
    } catch (e) {
        reply({ error: e instanceof Error ? e.message : String(e) });
    }
});