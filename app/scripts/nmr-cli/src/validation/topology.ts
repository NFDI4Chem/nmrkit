import { Molecule } from 'openchemlib'

import type { PreparedStructure } from './molfile'

export interface Topology {
  /** Implicit H count by 1-based original heavy-atom index. */
  hydrogenCounts: Map<number, number>
  /** Diastereotopic class ID by 1-based original heavy-atom index. */
  classes: Map<number, string>
  /** Servlet H atom number -> 1-based original index of the carrying heavy atom. */
  servletHydrogenOwners: Map<number, number>
}

/**
 * Topology of the H-free structure sent to nmrshiftdb.
 *
 * openchemlib keeps the atom order of an H-free V2000 molfile, so its indices
 * line up with the prepared numbering. The servlet appends implicit hydrogens
 * after the heavy atoms, heavy atom by heavy atom, which is what
 * `servletHydrogenOwners` reproduces.
 */
export function analyseTopology(structure: PreparedStructure): Topology {
  const molecule = Molecule.fromMolfile(structure.molfile)
  if (molecule.getAllAtoms() !== structure.heavyAtomCount) {
    throw new Error('Unexpected atom count after parsing the H-free molfile')
  }

  const diaIDs = molecule.getDiastereotopicAtomIDs()
  const hydrogenCounts = new Map<number, number>()
  const classes = new Map<number, string>()
  const servletHydrogenOwners = new Map<number, number>()

  let nextHydrogen = structure.heavyAtomCount + 1
  for (let index = 0; index < structure.heavyAtomCount; index++) {
    const original = structure.toOriginal[index + 1]
    const count = molecule.getImplicitHydrogens(index)
    hydrogenCounts.set(original, count)
    classes.set(original, diaIDs[index])
    for (let h = 0; h < count; h++) {
      servletHydrogenOwners.set(nextHydrogen++, original)
    }
  }

  return { hydrogenCounts, classes, servletHydrogenOwners }
}
