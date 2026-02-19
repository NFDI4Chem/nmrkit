import { Experiment } from "../../base";

export interface SpectraTypeMapItem {
    type: string; nucleus: string
}

export const spectraTypeMap: Partial<Record<Experiment, SpectraTypeMapItem>> = {
    proton: { type: 'nmr;1H;1d', nucleus: '1H' },
    carbon:

        { type: 'nmr;13C;1d', nucleus: '13C' },
}
