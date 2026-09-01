import { Spectrum } from '@zakodium/nmr-types';
import { join } from 'node:path';
import { Worker } from 'node:worker_threads';
import { WorkerResponse } from './type/WorkerResponse';

// Runs processing/detection/serialization in a separate worker thread (its
// own V8 heap) so a heap-OOM there only terminates the worker instead of the
// whole CLI process. One instance is created per concurrency lane (see
// run-pipeline.ts).

export class SpectrumWorker {
    private worker: Worker | null = null;

    private spawn(): Worker {
        const worker = new Worker(join(__dirname, 'worker-entry.js'), {
            resourceLimits: { maxOldGenerationSizeMb: 3072 },
        });
        worker.on('error', () => { this.worker = null; });
        worker.on('exit', () => { this.worker = null; });
        return worker;
    }

    private get(): Worker {
        if (!this.worker) this.worker = this.spawn();
        return this.worker;
    }

    run(task: 'process' | 'detect', spectrum: Spectrum): Promise<Spectrum>;
    run(task: 'serialize', spectrum: Spectrum, version: unknown): Promise<string>;
    run(task: 'process' | 'detect' | 'serialize', spectrum: Spectrum, version?: unknown): Promise<any> {
        return new Promise((resolve, reject) => {
            const worker = this.get();
            let settled = false;

            const onMessage = (msg: WorkerResponse) => {
                settled = true;
                cleanup();
                if (msg.error) reject(new Error(msg.error));
                else resolve(task === 'serialize' ? msg.stringObject : msg.spectrum);
            };
            const onError = (err: Error) => {
                settled = true;
                this.worker = null;
                cleanup();
                reject(err);
            };
            const onExit = (code: number) => {
                this.worker = null;
                if (!settled && code !== 0) {
                    cleanup();
                    reject(new Error(`Worker exited with code ${code} (likely out of memory)`));
                }
            };
            const cleanup = () => {
                worker.off('message', onMessage);
                worker.off('error', onError);
                worker.off('exit', onExit);
            };

            worker.on('message', onMessage);
            worker.on('error', onError);
            worker.on('exit', onExit);
            worker.postMessage({ spectrum, task, version });
        });
    }

    async terminate() {
        await this.worker?.terminate().catch(() => { });
        this.worker = null;
    }
}