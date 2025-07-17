import {
  resurrect,
  rangesToXY,
  type NMRRangeWithIntegration,
} from 'nmr-processing'
import { v4 } from '@lukeed/uuid'
import { CURRENT_EXPORT_VERSION } from '@zakodium/nmrium-core'
import { castToArray } from './utilities/castToArray'

interface Info {
  nucleus: string
  solvent: string
  name: string
}

function generateSpectrumFromRanges(
  ranges: NMRRangeWithIntegration[],
  info: Info
) {
  const { nucleus, solvent, name = null } = info

  const frequency = 400
  try {
    const { x, y } = rangesToXY(ranges, {
      nucleus,
      frequency,
      nbPoints: 2 ** 17,
    })

    const info = {
      isFid: false,
      isComplex: false,
      dimension: 1,
      nucleus,
      originFrequency: frequency,
      baseFrequency: frequency,
      pulseSequence: '',
      solvent,
      isFt: true,
      name,
    }

    const spectrum = {
      id: v4(),
      data: { x: castToArray(x), im: undefined, re: castToArray(y) },
      info,
      ranges: {
        values: ranges,
        options: {
          sum: 100,
          isSumConstant: false,
          sumAuto: false,
        },
      },
    }

    return { data: { spectra: [spectrum] }, version: CURRENT_EXPORT_VERSION }
  } catch (error) {
    console.log(error)
  }
}

function generateSpectrumFromPublicationString(publicationString: string) {
  const {
    ranges,
    info: { nucleus, solvent = '' },
    parts,
  } = resurrect(publicationString)
  return generateSpectrumFromRanges(ranges, {
    nucleus,
    solvent,
    name: parts[0],
  })
}

export { generateSpectrumFromPublicationString }
