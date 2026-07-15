function percentToHex(p: number): string {
    const percent = Math.max(0, Math.min(100, p));
    const intValue = Math.round((percent / 100) * 255);
    const hexValue = intValue.toString(16);
    return percent === 100 ? '' : hexValue.padStart(2, '0');
}

export function adjustAlpha(color: string, factor: number): string {
    return color + percentToHex(factor);
}