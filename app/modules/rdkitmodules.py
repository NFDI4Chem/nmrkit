from hosegen import HoseGenerator
from rdkit import Chem

async def getRDKitHOSECodes(smiles: str, noOfSpheres: int):
    if any(char.isspace() for char in smiles):
        smiles = smiles.replace(" ", "+")
    mol = Chem.MolFromSmiles(smiles)
    gen = HoseGenerator()
    hosecodes = []
    for i in range(0, len(mol.GetAtoms()) - 1):
        hosecode = gen.get_Hose_codes(mol, i, noOfSpheres)
        hosecodes.append(hosecode)
    return hosecodes