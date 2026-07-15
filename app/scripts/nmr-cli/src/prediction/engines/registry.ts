import type { Engine } from './base'

/**
 * Auto-discovered engine registry
 * Engines are automatically registered when imported
 */
class EngineRegistry {
    private engines = new Map<string, Engine>()

    /**
     * Register an engine
     * Called automatically when engine files are imported
     */
    register(engine: Engine): void {
        if (this.engines.has(engine.id)) {
            console.warn(`Engine ${engine.id} is already registered, overwriting...`)
        }
        this.engines.set(engine.id, engine)
    }

    /**
     * Get an engine by ID
     */
    get(id: string): Engine | undefined {
        return this.engines.get(id)
    }

    /**
     * Get all registered engines
     */
    getAll(): Engine[] {
        return Array.from(this.engines.values())
    }

    /**
     * Get all engine IDs
     */
    getIds(): string[] {
        return Array.from(this.engines.keys())
    }

    /**
     * Check if an engine exists
     */
    has(id: string): boolean {
        return this.engines.has(id)
    }
}

// Singleton instance
export const engineRegistry = new EngineRegistry()

/**
 * Helper function to create and auto-register an engine
 * Just call this at the bottom of your engine file!
 */
export function defineEngine(engine: Engine): Engine {
    engineRegistry.register(engine)
    return engine
}