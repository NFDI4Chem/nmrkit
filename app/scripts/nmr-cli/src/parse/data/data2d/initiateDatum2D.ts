import type {
  Spectrum2D,
} from '@zakodium/nmrium-core';
import { Filters2DManager } from 'nmr-processing';


import { initiateZones } from './initiateZones.js';
import { initiateFilters } from '../initiateFilters.js';

const defaultMinMax = { z: [], minX: 0, minY: 0, maxX: 0, maxY: 0 };


function initiateDisplay(spectrum: any) {
  return {
    isPositiveVisible: true,
    isNegativeVisible: true,
    isVisible: true,
    dimension: 2,
    ...spectrum.display,
  };
}

function initiateInfo(spectrum: any) {
  return {
    nucleus: ['1H', '1H'],
    isFt: true,
    isFid: false,
    isComplex: false, // if isComplex is true that mean it contains real/ imaginary  x set, if not hid re/im button .
    dimension: 2,
    ...spectrum.info,
  };
}


export function initiateDatum2D(
  spectrum: any,
): Spectrum2D {
  const datum: any = { ...spectrum };

  datum.id = spectrum.id || crypto.randomUUID();

  datum.display = initiateDisplay(spectrum);

  datum.info = initiateInfo(spectrum);

  datum.originalInfo = datum.info;

  datum.meta = { ...spectrum.meta };

  datum.customInfo = { ...spectrum.customInfo };

  datum.data = getData(datum, spectrum);
  datum.originalData = datum.data;
  datum.filters = initiateFilters(spectrum?.filters);

  datum.zones = initiateZones(spectrum, datum as Spectrum2D);

  //reapply filters after load the original data

  return datum;
}

function getData(datum: any, options: any) {
  if (datum.info.isFid) {
    const { re = defaultMinMax, im = defaultMinMax } = options.data;
    return { re, im };
  }
  return { rr: defaultMinMax, ...options.data };
}
