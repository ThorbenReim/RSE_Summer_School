from src.Plotter import Plotter
from src.Utils import Utils
from src.Ensemble import Ensemble
from src.Parser import Parser
import numpy as np
import Bio.PDB
import os

class Prediction:
    @staticmethod
    def _analyse_difference(prediction, graph):
        result = []
        all_diffs = []
        res_diff, res_pred, res_rmsf, res_id_to_name = {}, {}, {}, {}
        for pred, node, fluct in zip(prediction, graph.infile_IDs, graph.y):
            # skip backbone atoms:
            split = node.split("_")
            atomname = split[1]
            if Utils.atom_name_is_backbone(atomname):
                continue
            resName = split[2]
            resID = int(split[4])

            diff = abs(pred - fluct)
            all_diffs.append(diff)
            result.append((node, pred, fluct, diff))

            if resID not in res_diff:
                res_diff[resID] = []
                res_pred[resID] = []
                res_rmsf[resID] = []
            res_diff[resID].append(diff)
            res_pred[resID].append(pred)
            res_rmsf[resID].append(fluct)
            res_id_to_name[resID] = resName

        # result.sort(key=lambda x: x[3])
        header = ["Node,Prediction,RMSF,Difference"]
        print(",".join(header))
        for node, pred, fluct, diff in result:
            liste = [node, f"{pred:.2f}", f"{fluct:.2f}", f"{diff:.2f}"]
            print(",".join(liste))
        print()
        print(f"MAE:{np.mean(all_diffs):.2f}")

        print()
        print("ResID,ResName,MeanPred,MeanRMSF,MeanDiff,MaxDiff")
        for resID in sorted(res_diff.keys()):
            mean = np.mean(res_diff[resID])
            maxi = np.max(res_diff[resID])
            name = res_id_to_name[resID]
            mean_pred = np.mean(res_pred[resID])
            mean_rmsf = np.mean(res_rmsf[resID])
            print(f"{resID},{name},{mean_pred:.2f},{mean_rmsf:.2f},{mean:.2f},{maxi:.2f}")

    @staticmethod
    def _print_output(prediction, graph):
        print("AtomID,Predicted-RMSF")
        sep = "\t"
        for pred, node in zip(prediction, graph.infile_IDs):
            pdb_id, res_id, res_name, atom_name, chain_name, atom_id = Utils.extract_node_id(node)
            node_str = f"{pdb_id}{sep}{res_name}{res_id:>5}{sep}{atom_name:<4}{atom_id:>6}"
            string = f"{node_str}{sep}{pred:.2f}"
            print(string)

    @staticmethod
    def read_pbd_file(pdb_file):
        parser = Bio.PDB.PDBParser(QUIET=True)
        pdb_file = os.path.abspath(pdb_file)
        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"The PDB file '{pdb_file}' does not exist.")

        structure = parser.get_structure('protein', pdb_file)
        return structure

    @staticmethod
    def reset_bfactors(structure, infileIDtoBfactor):
        for chain in structure[0]:
            for residue in chain:
                for atom in residue:
                    infileID = atom.get_serial_number()
                    if infileID in infileIDtoBfactor:
                        bfac = infileIDtoBfactor[infileID]
                        atom.set_bfactor(bfac)
        return structure

    @staticmethod
    def get_infile_ID_to_normed_bfactor(structure):
        infileIDtoBfactor = dict()
        bfactors = []
        for chain in structure[0]:
            for residue in chain:
                if Bio.PDB.is_aa(residue, True):
                    for atom in residue:
                        bfac = atom.get_bfactor()
                        ID = atom.get_serial_number()
                        bfactors.append(bfac)
                        infileIDtoBfactor[ID] = bfac

        mean = np.mean(bfactors)
        stddev = np.std(bfactors)

        for k, v in infileIDtoBfactor.items():
            normed = (v - mean) / stddev
            infileIDtoBfactor[k] = normed

        return infileIDtoBfactor

    @staticmethod
    def get_infile_ID_to_prediction(prediction, infile_IDs):
        infileIDtoBfactor = dict()
        for id, pred in zip(infile_IDs, prediction):
            pdb_id, res_id, res_name, atom_name, chain_name, atom_id = Utils.extract_node_id(id)
            infileIDtoBfactor[int(atom_id)] = pred
        return infileIDtoBfactor

    @staticmethod
    def calc_diff(infileIDtoBfactorReal, infileIDtoBfactorPred):
        # Initialize the dictionary to store absolute differences
        infileIDtoBfactorAbsDiff = {}

        # Iterate over the keys in the first dictionary
        for key in infileIDtoBfactorReal.keys():
            if key in infileIDtoBfactorPred:
                # Calculate the absolute difference between the values from both dictionaries
                abs_diff = np.abs(infileIDtoBfactorReal[key] - infileIDtoBfactorPred[key])
                # Store the result in the new dictionary
                infileIDtoBfactorAbsDiff[key] = abs_diff
            else:
                # Handle the case where the key is not present in the second dictionary
                infileIDtoBfactorAbsDiff[key] = None  # or some other appropriate value or action

        return infileIDtoBfactorAbsDiff

    @staticmethod
    def remove_hydrogen_atoms(structure):
        for model in structure:
            for chain in model:
                for residue in chain:
                    atoms_to_delete = []
                    for atom in residue.get_atoms():
                        if atom.element == 'H':
                            atoms_to_delete.append(atom)

                    # Remove collected hydrogen atoms from the chain
                    for atom in atoms_to_delete:
                        atom.get_parent().detach_child(atom.id)

    @staticmethod
    def remove_non_standard_amino_acids(structure):
        # Iterate over all models in the structure
        for model in structure:
            # Iterate over all chains in the model
            for chain in model:

                # Collect residues to delete
                residues_to_delete = []

                # Check and collect non-standard residues
                for residue in chain:
                    if not Bio.PDB.is_aa(residue, standard=True):
                        residues_to_delete.append(residue)

                # Remove collected non-standard residues from the chain
                for residue in residues_to_delete:
                    chain.detach_child(residue.id)

                    
    @staticmethod
    def keep_highest_occupancy_atoms(structure):
        # Iterate over all models in the structure
        for model in structure:
            # Iterate over all chains in the model
            for chain in model:
                # Iterate over all residues in the chain
                for residue in chain:
                    # Check if the residue has disordered atoms
                    if isinstance(residue, Bio.PDB.Residue.Residue):
                        disordered_atoms = [atom for atom in residue if isinstance(atom, Bio.PDB.Atom.DisorderedAtom)]
                        for atom in disordered_atoms:
                            atom_to_delete = atom
                            atom_to_add = None

                            for child in atom:
                                if atom_to_add is None:
                                    atom_to_add = child
                                elif atom_to_add.get_occupancy() < child.get_occupancy():
                                    atom_to_add = child

                            atom_to_add.set_occupancy(1.0)
                            atom_to_add.disordered_flag = 0
                            residue.detach_child(atom_to_delete.id)
                            atom_to_add.set_parent(residue)
                            residue.add(atom_to_add)


    @staticmethod
    def run(pdb_file_path, desc_file_path, ml_model_path, features_keep, features_strip, resuts_path, user_info):
        parser = Parser()
        header, graph = parser.readGraphDescriptorFile(desc_file_path, features_keep, features_strip, user_info)
        ensemble_model = Ensemble.loadModel(ml_model_path, user_info)

        graph.sortNodeFeatures()

        y_pred = ensemble_model.predict(graph.node_features)
        y_real = graph.y

        Prediction._print_output(y_pred, graph)

        if pdb_file_path:
            structure = Prediction.read_pbd_file(pdb_file_path)
            Prediction.remove_hydrogen_atoms(structure)
            Prediction.remove_non_standard_amino_acids(structure)
            Prediction.keep_highest_occupancy_atoms(structure)
            infileIDtoBfactorReal = Prediction.get_infile_ID_to_normed_bfactor(structure)
            infileIDtoBfactorPred = Prediction.get_infile_ID_to_prediction(y_pred, graph.infile_IDs)
            infileIDtoBfactorAbsDiff = Prediction.calc_diff(infileIDtoBfactorReal, infileIDtoBfactorPred)
            model = Prediction.reset_bfactors(structure, infileIDtoBfactorAbsDiff)

            Plotter.plotValidationScatter(y_pred, y_real, f"{structure.header["idcode"].lower()}_scatter", user_info)

            # Save the updated model to a PDB file
            output_file_name = f'{structure.header["idcode"].lower()}.pdb'  # Specify the path where you want to save the PDB file
            output_file_path = os.path.join(resuts_path, output_file_name)
            pdb_io = Bio.PDB.PDBIO()
            pdb_io.set_structure(model)
            pdb_io.save(output_file_path)
