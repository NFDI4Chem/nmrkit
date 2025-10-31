import type {
  Spectrum1D,
} from '@zakodium/nmrium-core';
import { Filters1DManager } from 'nmr-processing';

import { initSumOptions } from './initSumOptions.js';
import { initiateRanges } from './initiateRanges.js';
import { convertDataToFloat64Array } from './convertDataToFloat64Array.js';
import { initiateFilters } from '../initiateFilters.js';
import { MoleculeExtended } from '../../type/MoleculeExtended.js';
import { initiatePeaks } from './initiatePeaks.js';
import { initiateIntegrals } from './initiateIntegrals.js';


interface InitiateDatum1DOptions {
  molecules?: MoleculeExtended[];
}

export function initiateDatum1D(
  spectrum: any,
  options: InitiateDatum1DOptions = {},
): Spectrum1D {
  const { molecules = [] } = options;

  const { integrals, ranges, ...restSpectrum } = spectrum;
  const spectrumObj: Spectrum1D = { ...restSpectrum };
  spectrumObj.id = spectrum.id || crypto.randomUUID();

  spectrumObj.display = {
    isVisible: true,
    isRealSpectrumVisible: true,
    ...spectrum.display,
  };

  spectrumObj.info = {
    nucleus: '1H', // 1H, 13C, 19F, ...
    isFid: false,
    isComplex: false, // if isComplex is true that mean it contains real/ imaginary  x set, if not hid re/im button .
    dimension: 1,
    ...spectrum.info,
  };

  spectrumObj.originalInfo = spectrumObj.info;

  spectrumObj.meta = { ...spectrum.meta };

  spectrumObj.customInfo = { ...spectrum.customInfo };

  spectrumObj.data = convertDataToFloat64Array(spectrum.data);

  spectrumObj.originalData = spectrumObj.data;

  spectrumObj.filters = initiateFilters(spectrum?.filters); //array of object {name: "FilterName", options: FilterOptions = {value | object} }

  const { nucleus } = spectrumObj.info;

  spectrumObj.peaks = initiatePeaks(spectrum, spectrumObj);

  const integralsOptions = initSumOptions(integrals?.options || {}, {
    nucleus,
    molecules,
  });
  spectrumObj.integrals = initiateIntegrals(
    spectrum,
    spectrumObj,
    integralsOptions,
  );

  const rangesOptions = initSumOptions(ranges?.options || {}, {
    nucleus,
    molecules,
  });
  spectrumObj.ranges = initiateRanges(spectrum, spectrumObj, rangesOptions);


  return spectrumObj;
}
