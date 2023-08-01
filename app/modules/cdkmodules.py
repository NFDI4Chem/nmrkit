import os
import pystow
from jpype import startJVM, getDefaultJVMPath
from jpype import JClass, JVMNotFoundException, isJVMStarted

# Start JVM to use CDK in python
try:
    jvmPath = getDefaultJVMPath()
except JVMNotFoundException:
    print(
        "If you see this message, for some reason JPype",
        "cannot find jvm.dll.",
        "This indicates that the environment varibale JAVA_HOME",
        "is not set properly.",
        "You can set it or set it manually in the code",
    )
    jvmPath = "Define/path/or/set/JAVA_HOME/variable/properly"
if not isJVMStarted():
    cdk_path = "https://github.com/cdk/cdk/releases/download/cdk-2.8/cdk-2.8.jar"
    cdkjar_path = str(pystow.join("CDK")) + "/cdk-2.8.jar"

    if not os.path.exists(cdkjar_path):
        jar_path = pystow.ensure("CDK", url=cdk_path)

    startJVM("-ea", classpath=[cdkjar_path])
    cdk_base = "org.openscience.cdk"


def getCDKSDG(smiles: str):
    """This function takes the user input SMILES and Creates a
       Structure Diagram Layout using the CDK.
    Args:
            smiles (string): SMILES string given by the user.
    Returns:
            mol object : mol object with CDK SDG.
    """
    if any(char.isspace() for char in smiles):
        smiles = smiles.replace(" ", "+")
    SCOB = JClass(cdk_base + ".silent.SilentChemObjectBuilder")
    SmilesParser = JClass(cdk_base + ".smiles.SmilesParser")(SCOB.getInstance())
    molecule = SmilesParser.parseSmiles(smiles)
    StructureDiagramGenerator = JClass(cdk_base + ".layout.StructureDiagramGenerator")()
    StructureDiagramGenerator.generateCoordinates(molecule)
    molecule_ = StructureDiagramGenerator.getMolecule()

    return molecule_


def getCDKSDGMol(smiles: str):
    """This function takes the user input SMILES and returns a mol
       block as a string with Structure Diagram Layout.
    Args:
            smiles (string): SMILES string given by the user.
    Returns:
            mol object (string): CDK Structure Diagram Layout mol block.
    """
    if any(char.isspace() for char in smiles):
        smiles = smiles.replace(" ", "+")
    StringW = JClass("java.io.StringWriter")()

    moleculeSDG = getCDKSDG(smiles)
    SDFW = JClass(cdk_base + ".io.SDFWriter")(StringW)
    SDFW.write(moleculeSDG)
    SDFW.flush()
    mol_str = str(StringW.toString())
    return mol_str


async def getCDKHOSECodes(smiles: str, noOfSpheres: int, ringsize: bool):
    """This function takes the user input SMILES and returns a mol
       block as a string with Structure Diagram Layout.
    Args:
            smiles (string): SMILES string given by the user.
    Returns:
            mol object (string): CDK Structure Diagram Layout mol block.
    """
    if any(char.isspace() for char in smiles):
        smiles = smiles.replace(" ", "+")
    SCOB = JClass(cdk_base + ".silent.SilentChemObjectBuilder")
    SmilesParser = JClass(cdk_base + ".smiles.SmilesParser")(SCOB.getInstance())
    molecule = SmilesParser.parseSmiles(smiles)
    HOSECodeGenerator = JClass(cdk_base + ".tools.HOSECodeGenerator")()
    HOSECodes = []
    atoms = molecule.atoms()
    for atom in atoms:
        moleculeHOSECode = HOSECodeGenerator.getHOSECode(
            molecule, atom, noOfSpheres, ringsize
        )
        HOSECodes.append(str(moleculeHOSECode))
    return HOSECodes
