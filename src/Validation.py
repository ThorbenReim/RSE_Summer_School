import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from src.Utils import Utils

def print_validation(models_dict, node_ids, X_test, y_test):
    for name, model in models_dict.items():
        pred_y = model.predict(X_test)
        valid = Validation(model_name=name, node_ids=node_ids, features=X_test, predicted_y=pred_y, real_y=y_test)
        #print(valid)
        #print(valid.get_per_residue_validation())
        #print(valid.get_per_atom_validation())
        #print(valid.get_per_atom_residue_validation())
        print(valid.get_top_N_worst_predictions(10))

def same_nodes(src_res_name, src_atom_name, node_ids, features, all_errors):
    X, Y = [], []
    for n, x, y in zip(node_ids, features, all_errors):
        pdb_id, res_id, res_name, atom_name, chain_name, atom_id = Utils.extract_node_id(n)
        if src_res_name == res_name and src_atom_name == atom_name:
            X.append(x)
            Y.append(y)
    return X, Y

def node_distance(features_a, features_b):
    summe = 0
    for a, b in zip(features_a, features_b):
        diff = abs(a-b)/max(a, b)
        if not np.isnan(diff):
            summe += diff
    return abs(summe / len(features_a))

def get_similar_node(src_node, src_node_features, node_error,node_ids, features, all_errors):
    liste = [(src_node, src_node_features, 0.0, node_error)]
    for n, f, e in zip(node_ids, features, all_errors):
        distance = node_distance(src_node_features, f)
        liste.append((n, f, e, distance))
    liste.sort(key=lambda x: x[-1])
    for tup in liste[:11]:
        n, f, e, distance = tup
        pdb_id, res_id, res_name, atom_name, chain_name, atom_id = Utils.extract_node_id(n)
        string = f"{pdb_id}-{res_name}{res_id}-{atom_name}{atom_id}\t{distance:.2f}\t{e:.2f}"
        print(string)

def find_most_similar_node(src_node, src_node_features, node_error, node_ids, features, all_errors):
    pdb_id, res_id, res_name, atom_name, chain_name, atom_id = Utils.extract_node_id(src_node)
    features, all_errors = same_nodes(res_name, atom_name, node_ids, features, all_errors)
    get_similar_node(src_node, src_node_features, node_error, node_ids, features, all_errors)


class Validation:
    def __init__(self, model_name, node_ids, features, predicted_y, real_y, n_params = -1):
        self.model_name = model_name
        self.features = features
        self._node_ids = node_ids
        self._predicted_y = predicted_y
        self._real_y = real_y
        self._measures = {
            "MAE": mean_absolute_error(predicted_y, real_y) if len(predicted_y) >= 2 else float('nan'),
            "MSE": mean_squared_error(predicted_y, real_y) if len(predicted_y) >= 2 else float('nan'),
            "R2": r2_score(real_y, predicted_y) if len(predicted_y) >= 2 else float('nan'),
            "Pearson": pearsonr(predicted_y, real_y).correlation if len(predicted_y) >= 2 else float('nan'),
            "n_params": n_params
        }
        self.data_rate = 1.00

    def getNofParams(self):
        return self._measures["n_params"]

    def get_top_N_worst_predictions(self, n):
        liste = []
        for this_node, feat, predicted_y, real_y in zip(self._node_ids, self.features, self._predicted_y, self._real_y):
            tup = this_node, feat, predicted_y, real_y, abs(predicted_y - real_y)
            liste.append(tup)
        liste.sort(key=lambda x: x[-1], reverse=True)
        sep = "\t"
        print()
        print(f"Top {n} worst predictions:")
        space = " " * (20 - 4)
        print(f"Rank{sep}Node{space}{sep}AbsErr{sep}Real{sep}Predicted")
        for i, tup in enumerate(liste):
            this_node, feat, predicted_y, real_y, abs_err = tup
            pdb_id, res_id, res_name, atom_name, chain_name, atom_id = Utils.extract_node_id(this_node)
            node_str = f"{pdb_id}-{res_name}{res_id}-{atom_name}{atom_id}"
            print(f"{i}{sep}{node_str:<20}{sep}{abs_err:.2f}{sep}{real_y:.2f}{sep}{predicted_y:.2f}")
            if i == n:
                break



    def set_data_rate(self, rate):
        self.data_rate = rate

    def get_most_errorness_node(self):
        mae = 0
        node_err = None
        node_features = None
        real = None
        pred = None
        all_errors = []
        for this_node, feat, predicted_y, real_y in zip(self._node_ids, self.features, self._predicted_y, self._real_y):
            err = abs(predicted_y - real_y)
            all_errors.append(err)
            if err > mae:
                node_features = feat
                mae = err
                node_err = this_node
                real = real_y
                pred = predicted_y
        pdb_id, res_id, res_name, atom_name, chain_name, atom_id = Utils.extract_node_id(node_err)
        node_str = f"{pdb_id}-{res_name}{res_id}-{atom_name}{atom_id}"
        string = f"Most errorness node: {node_str} - MAE: {mae:.3f}, Real: {real:.3f}, Predicted: {pred:.3f}"
        find_most_similar_node(node_err, node_features, mae, self._node_ids, self.features, all_errors)
        return string

    def get_string(self):
        liste = [f"{key}: {val:.3f}" for key, val in self._measures.items()]
        if 1.00 - self.data_rate > 0.001:
            liste.append(f"Rate: {self.data_rate:.3f}")
        return " ".join(liste)

    def __str__(self):
        return f"Model: {self.model_name} - {self.get_string()}"

    def get_measures(self):
        return self._measures

    def _get_node_id_property_validation(self, node_functor, what):
        #"18_C_MET_-_1"
        my_dict = dict()
        for node, feat, predicted_y, real_y in zip(self._node_ids, self.features, self._predicted_y, self._real_y):
            typ = node_functor(node)
            if typ not in my_dict:
                my_dict[typ] = dict()
                my_dict[typ]["Features"] = []
                my_dict[typ]["Predicted"] = []
                my_dict[typ]["Real"] = []
                my_dict[typ]["IDs"] = []
            my_dict[typ]["Features"].append(feat)
            my_dict[typ]["Predicted"].append(predicted_y)
            my_dict[typ]["Real"].append(real_y)
            my_dict[typ]["IDs"].append(node)

        # for key, val in my_dict.items():
        #     my_dict[key]["Validation"] = Validation(what, val["IDs"], my_dict[typ]["Features"], val["Predicted"], val["Real"])
        #     rate = len(val["Predicted"]) / len(self._predicted_y)
        #     my_dict[key]["Validation"].set_data_rate(rate)
        #
        # for key in sorted(my_dict.keys()):
        #     string = f"Model: {self.model_name} - {what}: {key} - {my_dict[key]["Validation"].get_string()} - {my_dict[key]["Validation"].get_most_errorness_node()}"
        #     print(string)

    def get_per_residue_validation(self):
        def get_residue_name(node_id):
            pdb_id, res_id, res_name, atom_name, chain_name, atom_id = Utils.extract_node_id(node_id)
            return res_name
        self._get_node_id_property_validation(get_residue_name, "Residue")

    def get_per_atom_validation(self):
        def get_atom_name(node_id):
            pdb_id, res_id, res_name, atom_name, chain_name, atom_id = Utils.extract_node_id(node_id)
            return atom_name
        self._get_node_id_property_validation(get_atom_name, "Atom")

    def get_per_atom_residue_validation(self):
        def get_atom_residue_name(node_id):
            pdb_id, res_id, res_name, atom_name, chain_name, atom_id = Utils.extract_node_id(node_id)
            return res_name + " " + atom_name
        self._get_node_id_property_validation(get_atom_residue_name, "Residue-Atom")

class ValidationAllSplits:
    def __init__(self, model_name, validation_list):
        self.model_name = model_name
        self._measures = {
            "MAE": float('nan'),
            "MSE": float('nan'),
            "R2": float('nan'),
            "Pearson": float('nan'),
        }
        self.stddev = {key: float('nan') for key in self._measures.keys()}
        self.mean = {key: float('nan') for key in self._measures.keys()}
        self.median = {key: float('nan') for key in self._measures.keys()}

        for key in self._measures.keys():
            measures = [val._measures[key] for val in validation_list]
            self._measures[key] = measures
            self.stddev[key] = np.std(measures)
            self.mean[key] = np.mean(measures)
            self.median[key] = np.median(measures)

    def __str__(self):
        string = f"{self.model_name}:"
        string += f"\n{10 * ' '}\tMean\tMedian\tStddev\tAll values"
        for key in self._measures.keys():
            values_string = ", ".join([f"{val:.2f}" for val in self._measures[key]])
            string += (f"\n{key:<10}\t{self.mean[key]:.2f}\t{self.median[key]:.2f}\t{self.stddev[key]:.2f}"
                       f"\t{values_string}")
        return string

class ValidationSingleGraph:
    def __init__(self, model, graph):
        self.y_pred = model.predict(graph.node_features)
        self.y_real = graph.y
        self.name = graph.name
        self.measures = {
            "MAE": mean_absolute_error(self.y_real, self.y_pred) if len(self.y_pred) >= 2 else float('nan'),
            "MSE": mean_squared_error(self.y_real, self.y_pred) if len(self.y_pred) >= 2 else float('nan'),
            "R2": r2_score(self.y_real, self.y_pred) if len(self.y_pred) >= 2 else float('nan'),
            "Pearson": pearsonr(self.y_real, self.y_pred).correlation if len(self.y_pred) >= 2 else float('nan')
        }


class ValidationGraphs:
    def __init__(self, model_name, number_to_print_per_split: int = 100):
        self.model_name = model_name
        self.singleGraphValidations = []
        self.number_to_print_per_split = number_to_print_per_split

    def addGraphs(self, model, graphs):
        new_split = [ValidationSingleGraph(model, g) for g in graphs]
        self.singleGraphValidations.append(new_split)

    def __str__(self):
        string = f"{self.model_name}:"
        string += f"\n{10 * ' '}\tPDB\tSplit\tIndex\tMAE\tMSE\tR2\tPearson"
        for i, split in enumerate(self.singleGraphValidations):
            for j, g in enumerate(sorted(split, key=lambda g: g.measures["R2"])):
                string += f"\n{10 * ' '}\t{g.name}\t{i}\t{j}\t{g.measures['MAE']:.2f}\t{g.measures['MSE']:.2f}\t{g.measures['R2']:.2f}\t{g.measures['Pearson']:.2f}"
                if j == self.number_to_print_per_split - 1:
                    break
            string += "\n"
        return string
