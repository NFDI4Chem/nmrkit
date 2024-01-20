export function castToArray(value: Float64Array | number[]): number[] {
    return ArrayBuffer.isView(value) ? Array.from(value) : value
}