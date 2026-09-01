import { Spectrum } from "@zakodium/nmr-types";

export interface WorkerResponse {
    spectrum?: Spectrum;
    stringObject?: string;
    error?: string;
}