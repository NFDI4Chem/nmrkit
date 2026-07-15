export function generateName(
    name: string,
    options: { frequency: number | number[]; experiment: string },
) {
    const { frequency, experiment } = options
    const freq = Array.isArray(frequency) ? frequency[0] : frequency
    return name || `${experiment.toUpperCase()}_${freq}MHz`
}