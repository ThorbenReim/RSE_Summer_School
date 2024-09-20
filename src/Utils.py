import os
import numpy as np

class Utils:
    @staticmethod
    def get_all_files(directory: str, file_paths: str, extension: str):
        all_file_paths = []
        if file_paths:
            all_file_paths += [f for f in file_paths if f.endswith(extension)]
        if directory:
            files = [os.path.join(directory, f)
                     for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
            all_file_paths += [f for f in files if f.endswith(extension)]
        return all_file_paths

    @staticmethod
    def atom_name_is_backbone(atom_name):
        backbone_atoms = {"CA" , "C", "N", "O", "OT1", "OT2"}
        if atom_name in backbone_atoms:
            return True
        return False

    @staticmethod
    def atom_name_is_sidechain(atom_name):
        if not Utils.atom_name_is_backbone(atom_name):
                return True
        return False

    @staticmethod
    def extract_node_id(node):
        split = node.split("_")
        pdb_id = split[0]
        atom_id = split[1]
        atom_name = split[2]
        res_name = split[3]
        chain_name = split[4]
        res_id = split[5]
        return pdb_id, res_id, res_name, atom_name, chain_name, atom_id

    @staticmethod
    def get_atom_id(node_id):
        pdb_id, res_id, res_name, atom_name, chain_name, atom_id = Utils.extract_node_id(node_id)
        return atom_id


