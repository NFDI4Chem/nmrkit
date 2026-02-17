import { Peak2D, Signal2D, Zone } from "@zakodium/nmr-types"
import { NMRZone } from "nmr-processing"

export function mapZones(zones: NMRZone[]): Zone[] {
    return zones.map((zone): Zone => {
        const { signals, ...resZone } = zone
        const newSignals = signals.map((signal): Signal2D => {
            const { x, y, id, peaks, kind, ...resSignal } = signal
            return {
                ...resSignal,
                id: id || crypto.randomUUID(),
                kind: kind || 'signal',
                x: { ...x, originalDelta: x.delta || 0 },
                y: { ...y, originalDelta: y.delta || 0 },
                peaks: peaks?.map(
                    (peak): Peak2D => ({
                        ...peak,
                        id: peak.id || crypto.randomUUID(),
                    }),
                ),
            }
        })
        return {
            ...resZone,
            id: crypto.randomUUID(),
            signals: newSignals,
            kind: 'signal',
        }
    })
}