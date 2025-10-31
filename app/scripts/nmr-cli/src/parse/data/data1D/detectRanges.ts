import { xFindClosestIndex } from "ml-spectra-processing";
import { isProton } from "../../utility/isProton";
import { Spectrum1D } from "@zakodium/nmrium-core";
import { mapRanges, OptionsXYAutoPeaksPicking, updateRangesRelativeValues, xyAutoRangesPicking } from "nmr-processing";


//TODO expose OptionsPeaksToRanges from nmr-processing
interface OptionsPeaksToRanges {
    /**
     * Number of hydrogens or some number to normalize the integration data. If it's zero return the absolute integration value
     * @default 100
     */
    integrationSum?: number;
    /**
     * if it is true, it will join any overlaped ranges.
     * @default true
     */
    joinOverlapRanges?: boolean;
    /**
     * If exits it remove all the signals with integration < clean value
     * @default 0.4
     */
    clean?: number;
    /**
     * If true, the Janalyzer function is run over signals to compile the patterns.
     * @default true
     */
    compile?: boolean;
    /**
     * option to chose between approx area with peaks or the sum of the points of given range ('sum', 'peaks')
     * @default 'sum'
     */
    integralType?: string;
    /**
     * Observed frequency
     * @default 400
     */
    frequency?: number;
    /**
     * distance limit to clustering peaks.
     * @default 16
     */
    frequencyCluster?: number;
    /**
     * If true, it will keep the peaks for each signal
     */
    keepPeaks?: boolean;
    /**
     * Nucleus
     * @default '1H'
     */
    nucleus?: string;
    /**
     * ratio of heights between the extreme peaks
     * @default 1.5
     */
    symRatio?: number;
}


interface AutoDetectOptions {
    from?: number;
    to?: number;
    minMaxRatio?: number;
    lookNegative?: number;
}



export function detectRanges(
    spectrum: Spectrum1D,
    options: AutoDetectOptions = {},
) {


    const { from, to, minMaxRatio = 0.05, lookNegative = false } = options
    const { info: { nucleus, solvent, originFrequency }, data } = spectrum;
    let { x, re } = data
    const windowFromIndex = from ? xFindClosestIndex(x, from) : undefined;
    const windowToIndex = to ? xFindClosestIndex(x, to) : undefined;

    const isProtonic = isProton(nucleus);

    const peakPickingOptions: OptionsXYAutoPeaksPicking = {
        ...defaultPeakPickingOptions,
        smoothY: undefined,
        sensitivity: 100,
        broadWidth: 0.05,
        thresholdFactor: 8,
        minMaxRatio,
        direction: lookNegative ? 'both' : 'positive',
        frequency: originFrequency,
        sgOptions: undefined,

    };

    const rangesOptions: OptionsPeaksToRanges = {
        nucleus,
        compile: isProtonic,
        frequency: originFrequency,
        integrationSum: isProtonic ? spectrum.ranges.options.sum : 100,
        frequencyCluster: isProtonic ? 16 : 0,
        clean: 0.5,
        keepPeaks: true,
        joinOverlapRanges: isProtonic,
    };

    if (windowFromIndex !== undefined && windowToIndex !== undefined) {
        x = x.slice(windowFromIndex, windowToIndex);
        re = re.slice(windowFromIndex, windowToIndex);
    }


    const ranges = xyAutoRangesPicking(
        { x, y: re },
        {
            impurities: nucleus === '13C' ? { solvent: solvent || '' } : undefined,
            peakPicking: peakPickingOptions,
            ranges: rangesOptions,
        },
    );


    spectrum.ranges.values = spectrum.ranges.values.concat(
        mapRanges(ranges, spectrum),
    );

    updateRangesRelativeValues(spectrum);
}


const defaultPeakPickingOptions: OptionsXYAutoPeaksPicking = {
    minMaxRatio: 1,
    shape: { kind: 'lorentzian' },
    realTopDetection: true,
    maxCriteria: true,
    smoothY: true,
    sensitivity: 100,
    broadWidth: 0.25,
    broadRatio: 0.0025,
    thresholdFactor: 5,
    sgOptions: { windowSize: 7, polynomial: 3 },
    frequency: 0
};

