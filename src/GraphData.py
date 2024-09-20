import numpy as np
import pandas as pd

from src.Utils import Utils

def extract_info_from_graphs(graphs, useDataframe: bool = True):
    node_ids = pd.concat([g.infile_IDs for g in graphs], axis=0, ignore_index=True)
    x = pd.concat([g.node_features for g in graphs], axis=0, ignore_index=True)
    if not useDataframe:
         x = x.to_numpy()
    y = np.concatenate([g.y for g in graphs], axis=0)
    return node_ids, x, y

class Data:
    def __init__(self, train, test, validation):
        self.train = train
        self.test = test
        self.validation = validation

class GraphData:
    def __init__(self, y, node_features, adj_matrix, edge_features, infile_IDs, name):
        self.y = None
        if isinstance(y, list):
            self.y = np.array(y)
        elif isinstance(y, (np.ndarray, np.generic) ):
            self.y = y # already is np.ndarray
        else:
            raise TypeError("'y' must be of type 'list' or 'numpy.ndarray'!")
        self.node_features = node_features
        self.adj_matrix = np.array(adj_matrix)
        self.edge_features = np.array(edge_features)
        if isinstance(infile_IDs, list):
            self.infile_IDs = pd.Series(infile_IDs)
        else:
            self.infile_IDs = infile_IDs
        self.name = name

    def sortNodeFeatures(self):
        self.node_features = self.node_features.sort_index(axis=1)

    def keep_sidechain_atoms(self):
        node_ids_stripped, X_stripped, y_stripped = [], [], []
        for node_id, desc, rmsf in zip(self.infile_IDs, self.node_features.itertuples(index=False), self.y):
            pdb_id, res_id, res_name, atom_name, chain_name, atom_id = Utils.extract_node_id(node_id)
            if not Utils.atom_name_is_backbone(atom_name):
                node_ids_stripped.append(node_id)
                X_stripped.append(desc)
                y_stripped.append(rmsf)
        df = pd.DataFrame(X_stripped)
        return GraphData(y_stripped, df, [], [], node_ids_stripped, self.name)

    def _keep_by_atom_name(self, which_atom_name_to_keep):
        node_ids_stripped, X_stripped, y_stripped = [], [], []
        for node_id, desc, rmsf in zip(self.infile_IDs, self.node_features.itertuples(index=False), self.y):
            pdb_id, res_id, res_name, atom_name, chain_name, atom_id = Utils.extract_node_id(node_id)
            if atom_name == which_atom_name_to_keep:
                node_ids_stripped.append(node_id)
                X_stripped.append(desc)
                y_stripped.append(rmsf)
        df = pd.DataFrame(X_stripped)
        return GraphData(y_stripped, df, [], [], node_ids_stripped, self.name)

    def keep_beta_carbons_only(self):
        return self._keep_by_atom_name("CB")

    def keep_alpha_carbons_only(self):
        return self._keep_by_atom_name("CA")

    def keep_most_flexible_atom_descriptors_per_residue(self):
        def _append(list_a, list_b, list_c, a, b, c):
            list_a.append(a)
            list_b.append(b)
            list_c.append(c)

        node_ids_stripped, X_stripped, y_stripped = [], [], []
        for node_id, desc, rmsf in zip(self.infile_IDs, self.node_features.itertuples(index=False), self.y):
            pdb_id, res_id, res_name, atom_name, chain_name, atom_id = Utils.extract_node_id(node_id)
            if res_name == "ALA" and atom_name == "CB":
                _append(node_ids_stripped, X_stripped, y_stripped, node_id, desc, rmsf)
            if res_name == "ARG" and atom_name == "NH1":
                _append(node_ids_stripped, X_stripped, y_stripped, node_id, desc, rmsf)
            if res_name == "ASN" and atom_name == "ND2":
                _append(node_ids_stripped, X_stripped, y_stripped, node_id, desc, rmsf)
            if res_name == "ASP" and atom_name == "OD2":
                _append(node_ids_stripped, X_stripped, y_stripped, node_id, desc, rmsf)
            if res_name == "CYS" and atom_name == "SG":
                _append(node_ids_stripped, X_stripped, y_stripped, node_id, desc, rmsf)
            if res_name == "GLN" and atom_name == "OE1":
                _append(node_ids_stripped, X_stripped, y_stripped, node_id, desc, rmsf)
            if res_name == "GLU" and atom_name == "OE1":
                _append(node_ids_stripped, X_stripped, y_stripped, node_id, desc, rmsf)
            if res_name == "GLY":
                pass
            if res_name == "HIS" and atom_name == "NE2":
                _append(node_ids_stripped, X_stripped, y_stripped, node_id, desc, rmsf)
            if res_name == "ILE" and atom_name == "CD":
                _append(node_ids_stripped, X_stripped, y_stripped, node_id, desc, rmsf)
            if res_name == "LEU" and atom_name == "CD1":
                _append(node_ids_stripped, X_stripped, y_stripped, node_id, desc, rmsf)
            if res_name == "LYS" and atom_name == "NZ":
                _append(node_ids_stripped, X_stripped, y_stripped, node_id, desc, rmsf)
            if res_name == "MET" and atom_name == "CE":
                _append(node_ids_stripped, X_stripped, y_stripped, node_id, desc, rmsf)
            if res_name == "PHE" and atom_name == "CZ":
                _append(node_ids_stripped, X_stripped, y_stripped, node_id, desc, rmsf)
            if res_name == "PRO" and atom_name == "CG":
                _append(node_ids_stripped, X_stripped, y_stripped, node_id, desc, rmsf)
            if res_name == "SER" and atom_name == "OG":
                _append(node_ids_stripped, X_stripped, y_stripped, node_id, desc, rmsf)
            if res_name == "THR" and atom_name == "CG2":
                _append(node_ids_stripped, X_stripped, y_stripped, node_id, desc, rmsf)
            if res_name == "TRP" and atom_name == "CH2":
                _append(node_ids_stripped, X_stripped, y_stripped, node_id, desc, rmsf)
            if res_name == "TYR" and atom_name == "OH":
                _append(node_ids_stripped, X_stripped, y_stripped, node_id, desc, rmsf)
            if res_name == "VAL" and atom_name == "CG1":
                _append(node_ids_stripped, X_stripped, y_stripped, node_id, desc, rmsf)
        df = pd.DataFrame(X_stripped)

        return GraphData(y_stripped, df, [], [], node_ids_stripped, self.name)