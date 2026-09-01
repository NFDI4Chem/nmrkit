import { Spectrum } from "@zakodium/nmr-types";

// Message contract shared with detectWorkerEntry.ts — keep both in sync.
export interface WorkerRequest {
    spectrum: Spectrum;
    task: 'process' | 'detect' | 'serialize';
    version?: unknown;
}