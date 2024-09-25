import unittest

import numpy as np
import Bio

from src.Prediction import Prediction

class TestPrediction(unittest.TestCase):

    def _test_reset_bfactors(self, structure):
        new_bfactor_to_set = 0

        infileIDtoBfactor = dict()
        for chain in structure[0]:
            for residue in chain:
                for atom in residue:
                    infileID = atom.get_serial_number()
                    infileIDtoBfactor[infileID] = new_bfactor_to_set

        structure_mod = Prediction.reset_bfactors(structure, infileIDtoBfactor)

        for chain in structure_mod[0]:
            for residue in chain:
                for atom in residue:
                    bfac = atom.get_bfactor()
                    self.assertEqual(
                        bfac, new_bfactor_to_set,
                        f"Resetting the B-Factor should be {new_bfactor_to_set} instead if {bfac}."
                    )

    def _test_get_infile_ID_to_normed_bfactor(self, structure):
        pdb_file = "../data/pdb/1ab1.pdb"
        structure = Prediction.read_pbd_file(pdb_file)

        infileIDtoBfactor = Prediction.get_infile_ID_to_normed_bfactor(structure)

        values = np.array([v for v in infileIDtoBfactor.values()])

        mean = values.mean()
        self.assertAlmostEqual(mean, 0.0, 10)
        stddev = values.std()
        self.assertAlmostEqual(stddev, 1.0, 10)

    def _test_keep_highest_occupancy_atoms(self, structure):
        Prediction.keep_highest_occupancy_atoms(structure)
        for chain in structure:
            for residue in chain:
                for atom in residue:
                    self.assertFalse(
                        isinstance(atom,
                                   Bio.PDB.Atom.DisorderedAtom),
                        f"Atom should be of type 'Bio.PDB.Atom.Atom' instead of 'Bio.PDB.Atom.DisorderedAtom'"
                    )

    def _test_remove_hydrogen_atoms(self, structure):
        Prediction.remove_hydrogen_atoms(structure)

        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        self.assertNotEqual(
                            atom.element, "H",
                            f"There should be not atom with element '{atom.element}'"
                        )

    def _test_remove_non_standard_amino_acids(self, structure):
        Prediction.remove_non_standard_amino_acids(structure)

        for model in structure:
            for chain in model:
                for residue in chain:
                    self.assertTrue(
                        Bio.PDB.is_aa(residue, standard=True),
                        f"Residue should be standard amino acid but is '{residue.get_resname()}'."
                    )


    def test_prediction(self):
        self.assertRaises(
            FileNotFoundError,
            Prediction.read_pbd_file,
            "this_is_an_unknown_file_path_wew28764234523472")

        pdb_file = "../data/pdb/1ab1.pdb"
        structure = Prediction.read_pbd_file(pdb_file)
        self.assertTrue(
            isinstance(structure, Bio.PDB.Structure.Structure),
            f"Should be 'Bio.PDB.Structure.Structure' but is {type(structure)}"
        )

        self._test_reset_bfactors(structure)
        self._test_get_infile_ID_to_normed_bfactor(structure)
        self._test_keep_highest_occupancy_atoms(structure)
        self._test_remove_hydrogen_atoms(structure)
        self._test_remove_non_standard_amino_acids(structure)


if __name__ == '__main__':
    unittest.main()