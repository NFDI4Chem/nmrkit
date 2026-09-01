import os from 'node:os';

// --- Concurrency tuning -----------------------------------------------
// CPU-bound work (auto-processing / peak detection) is parallelized across
// worker threads: one worker per lane, reused for every spectrum that lane
// handles (spawning a worker reloads the whole nmr-processing module graph,
// so lanes are long-lived, not spawned per spectrum).
//
// Snapshot capture is parallelized across browser tabs, but each tab is far
// more memory-hungry than a worker thread, so it gets its own (lower) cap
// regardless of CPU count.
//
// Both are overridable via env vars for tuning on a given machine.
export const CPU_CONCURRENCY = Number(process.env['NMR_CLI_CPU_CONCURRENCY']) || Math.max(1, os.cpus().length - 1);
export const SNAPSHOT_CONCURRENCY = Number(process.env['NMR_CLI_SNAPSHOT_CONCURRENCY']) || 3;

// Runs `handler` over `items` using a fixed number of long-lived "lanes"
// rather than firing off one promise per item. `laneIndex` is stable for the
// lifetime of a lane, so handlers can lazily attach an expensive, reusable
// resource (a worker thread, a browser page) to a given lane instead of
// creating one per item.
export async function runWithConcurrency<T>(
    items: T[],
    concurrency: number,
    handler: (item: T, laneIndex: number) => Promise<void>
): Promise<void> {
    if (items.length === 0) return;
    let cursor = 0;
    const laneCount = Math.max(1, Math.min(concurrency, items.length));
    const lanes = Array.from({ length: laneCount }, (_, laneIndex) =>
        (async () => {
            while (cursor < items.length) {
                const item = items[cursor++];
                await handler(item, laneIndex);
            }
        })()
    );
    await Promise.all(lanes);
}