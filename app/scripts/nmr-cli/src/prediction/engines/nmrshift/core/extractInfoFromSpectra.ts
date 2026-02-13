import { Experiment } from "../../base";
import { spectraTypeMap, SpectraTypeMapItem } from "./spectraTypeMap";



export function extractInfoFromSpectra(spectra: Experiment[]) {
    const info: SpectraTypeMapItem[] = [];
    for (const experiment of spectra) {
        const data = spectraTypeMap[experiment];
        if (!data) continue;

        info.push(data)
    }
    return info;
}