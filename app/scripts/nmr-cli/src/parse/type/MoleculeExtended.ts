import { StateMolecule } from "@zakodium/nmrium-core";

export interface MoleculeExtended
    extends Required<Pick<StateMolecule, 'id' | 'molfile' | 'label'>>,
    Omit<StateMolecule, 'id' | 'molfile' | 'label'> {
    mf: string;
    em: number;
    mw: number;
    svg: string;
    atoms: Record<string, number>;
}