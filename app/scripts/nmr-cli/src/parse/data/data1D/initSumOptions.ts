import type { SumOptions } from '@zakodium/nmr-types';
import { MF } from 'mf-parser';
import getAtom from '../../utility/getAtom';
import { MoleculeExtended } from '../../type/MoleculeExtended';

export {
  updateIntegralsRelativeValues,
  updateRangesRelativeValues,
} from 'nmr-processing';


export interface SumParams {
  nucleus: string;
  molecules: MoleculeExtended[];
}

export type SetSumOptions = Omit<SumOptions, 'isSumConstant'>;

export function initSumOptions(
  options: Partial<SumOptions>,
  params: SumParams,
) {
  let newOptions: SumOptions = {
    sum: undefined,
    isSumConstant: true,
    sumAuto: true,
    ...options,
  };
  const { molecules, nucleus } = params;

  if (options.sumAuto && Array.isArray(molecules) && molecules.length > 0) {
    const { mf, id } = molecules[0];
    newOptions = { ...newOptions, sumAuto: true, mf, moleculeId: id };
  } else {
    const { mf, moleculeId, ...resOptions } = newOptions;
    newOptions = { ...resOptions, sumAuto: false };
  }
  if (!newOptions.sum) {
    newOptions.sum = getSum(newOptions.mf || null, nucleus);
  }

  return newOptions;
}

export function getSum(mf: string | null | undefined, nucleus: string) {
  const defaultSum = 100;

  if (!mf || !nucleus) return defaultSum;

  const atom = getAtom(nucleus);
  const atoms = new MF(mf).getInfo().atoms;

  return atoms[atom] || defaultSum;
}


