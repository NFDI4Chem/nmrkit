import { Spectrum2D } from "@zakodium/nmrium-core";
import { isFt2DSpectrum } from "./isSpectrum2D";
import { mapZones } from "nmr-processing";
import { getDetectionZones } from "./getDetectionZones";
import { Zone } from "@zakodium/nmr-types";

export function detectZones(spectrum: Spectrum2D) {

    if (!isFt2DSpectrum(spectrum)) return;

    const { data } = spectrum;

    const { rr: { minX, maxX, minY, maxY } } = data;
    const detectionOptions = {
        selectedZone: { fromX: minX, toX: maxX, fromY: minY, toY: maxY },
        thresholdFactor: 1,
        maxPercentCutOff: 0.03,
    };

    const zones = getDetectionZones(spectrum, detectionOptions);
    spectrum.zones.values = mapZones(zones as Zone[], spectrum);


}