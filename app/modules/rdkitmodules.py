from hosegen import HoseGenerator
from rdkit import Chem
from typing import List


async def getRDKitHOSECodes(smiles: str, noOfSpheres: int) -> List[str]:
    """
    Generate RDKit HOSE codes for a given SMILES string.

    Parameters:
        smiles (str): The input SMILES string representing the molecular structure.
        noOfSpheres (int): The number of spheres for generating HOSE codes.

    Returns:
        List[str]: A list of generated HOSE codes for each atom in the molecule.
    """
    if any(char.isspace() for char in smiles):
        smiles = smiles.replace(" ", "+")
    mol = Chem.MolFromSmiles(smiles)
    gen = HoseGenerator()
    hosecodes = []
    for i in range(0, len(mol.GetAtoms()) - 1):
        hosecode = gen.get_Hose_codes(mol, i, noOfSpheres)
        hosecodes.append(hosecode)
    return hosecodes
