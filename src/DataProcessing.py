import pandas as pd
import numpy as np
from sklearn.impute import MissingIndicator
from sklearn.model_selection import KFold

from src.Utils import Utils
from src.UserInfo import UserInfo
from src.GraphData import Data, GraphData, extract_info_from_graphs


def _reconstruct_graphs(graphs, tramsformed_df):
    graph_sizes = [len(g.node_features) for g in graphs]
    scaled_graphs = []
    size_sum = 0
    for size, graph in zip(graph_sizes, graphs):
        max_idx = size_sum + size
        scaled_features = tramsformed_df[size_sum: max_idx]
        size_sum = max_idx
        if isinstance(scaled_features, pd.DataFrame):
            scaled_features = scaled_features.to_numpy()
        graph.node_features = scaled_features
        scaled_graphs.append(graph)
    return scaled_graphs

class DataPreprocessing:

    @staticmethod
    def get_numercial_scaler():
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, QuantileTransformer, PowerTransformer
        scaler_dict = {
            "Drop": None,
            "No": False,
            "None": False,
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler(),
            "MaxAbsScaler": MaxAbsScaler(),
            "Normalizer": Normalizer(),
            "QuantileTransformer": QuantileTransformer(),
            "PowerTransformer": PowerTransformer()
        }
        return scaler_dict

    @staticmethod
    def get_categorical_scaler():
        from sklearn.preprocessing import OneHotEncoder, LabelEncoder
        scaler_dict = {
            "Drop": None,
            "No": False,
            "None": False,
            "OneHotEncoder": OneHotEncoder(),
            "LabelEncoder": LabelEncoder()
        }
        return scaler_dict


    @staticmethod
    def _get_used_numerical_features(feature_names):
        return [name for name in feature_names if name[:2] == "N_"]

    @staticmethod
    def _get_used_categorical_features(feature_names):
        return [name for name in feature_names if name[:2] == "C_"]

    @staticmethod
    def _check_for_feature_prefix(feature_names, user_info):
        unknown_features = [name for name in feature_names if name[:2] not in {"C_", "N_"}]
        if unknown_features:
            user_info.log(f"In DataPreprocessing there are the following unknown features: '{' '.join(unknown_features)}'. "
                          f"These are excluded in the next steps!",
                          UserInfo.WARNING)
            user_info.log(f"Add a prefix for the feature name, 'C_' for categorical or 'N_' for numerical.",
                          UserInfo.WARNING)
        used_features = [name for name in feature_names if name not in unknown_features]
        return used_features

    @staticmethod
    def split_for_numercial_and_categorical_features(graph, user_info):
        known_feature_names = DataPreprocessing._check_for_feature_prefix(graph.node_features.columns, user_info)
        used_numerical_feature_names = DataPreprocessing._get_used_numerical_features(known_feature_names)

        used_categorical_feature_names = DataPreprocessing._get_used_categorical_features(known_feature_names)

        numerical_features_df = graph.node_features[used_numerical_feature_names]
        categorical_features_df = graph.node_features[used_categorical_feature_names]

        return numerical_features_df, categorical_features_df

    @staticmethod
    def scale(graphs, num_scaler_string, cat_scaler_string, polynomial_degree, user_info):
        if polynomial_degree > 1:
            from sklearn.preprocessing import PolynomialFeatures
            user_info.log(f"Apply feature polynomialization with degree {polynomial_degree} ... ")
            poly = PolynomialFeatures(degree=polynomial_degree, interaction_only=True, include_bias=True)
            poly.set_output(transform="pandas")

        num_scl = DataPreprocessing.get_numercial_scaler()[num_scaler_string]
        if num_scl:
            num_scl.set_output(transform="pandas")
            user_info.log(f"Apply numerical scaler '{num_scaler_string}'...")

        cat_scl = DataPreprocessing.get_categorical_scaler()[cat_scaler_string]
        if cat_scl:
            cat_scl.set_output(transform="pandas")
            cat_scl.sparse_output=False
            user_info.log(f"Apply categorical scaler '{cat_scaler_string}'...")

        new_graphs = []
        for g in graphs:
            num_df, cat_df = DataPreprocessing.split_for_numercial_and_categorical_features(g, user_info)
            if polynomial_degree > 1:
                poly.fit(num_df)
                num_df = poly.fit_transform(num_df)

            if num_scaler_string == "Drop":
                num_df = pd.DataFrame()
            elif num_scl:
                num_scl.fit(num_df)
                num_df = num_scl.transform(num_df)

            if cat_scaler_string == "Drop":
                cat_df = pd.DataFrame()
            elif cat_scl:
                cat_scl.fit(cat_df)
                cat_df = cat_scl.transform(cat_df)
            g.node_features = pd.concat([num_df, cat_df], axis=1)
            new_graphs.append(g)
        return new_graphs




class FeatureStripping:
    @staticmethod
    def applyFlexlevelStripping(graphs, flex_level, user_info):
        new_graphs = []
        old_size = sum(len(feature) for g in graphs for feature in g.infile_IDs)
        if flex_level == "sidechain":
            new_graphs = [g.keep_sidechain_atoms() for g in graphs]
        elif flex_level == "betacarbon":
            new_graphs = [g.keep_beta_carbons_only() for g in graphs]
        elif flex_level == "alphacarbon":
            new_graphs = [g.keep_alpha_carbons_only() for g in graphs]
        elif flex_level == 'mostflexibleresidueatom':
            new_graphs = [g.keep_most_flexible_atom_descriptors_per_residue() for g in graphs]
        elif flex_level == 'allatoms':
            new_graphs = graphs
        else:
            user_info.log(f"Flexlevel '{flex_level}' not yet implemented. Using all descriptors...",
                          UserInfo.WARNING)
            new_graphs = graphs
        new_size = sum(len(feature) for g in new_graphs for feature in g.infile_IDs)
        user_info.log(
            f"Using Flexlevel '{flex_level}' keeped {new_size/old_size*100:.2f}% ({new_size}) from {old_size} atom descriptors.",
            UserInfo.INFO)
        return new_graphs

class Decomposition:
    @staticmethod
    def PCA(df, n_components, user_info):
        from sklearn.decomposition import PCA

        user_info.log(f"Applying PCA with {n_components} components ...",
                      UserInfo.INFO)
        pca = PCA(n_components=n_components)
        pca.fit(df)
        array = pca.fit_transform(df)
        feature_names_reduced = pca.get_feature_names_out()
        df_reduced = pd.DataFrame(array, columns=feature_names_reduced)
        return df_reduced, feature_names_reduced

class PreprocessGraphData:
    @staticmethod
    def removeGraphsWithNaNFeatures(graphs):
        newGraphs = []
        removedGraphs = 0
        indicator = MissingIndicator()
        for g in graphs:
            data = indicator.fit_transform(g.node_features)
            at_least_one_value_is_nan = np.any(data)
            if at_least_one_value_is_nan:
                removedGraphs += 1
            else:
                newGraphs.append(g)
        return newGraphs, removedGraphs

    @staticmethod
    def _drop_feature_columns(graph, columns_to_drop):
        graph.node_features = graph.node_features.drop(columns=columns_to_drop)
        return graph

    @staticmethod
    def dropAtomFeatures(graphs):
        name = 'Atom'
        columns_to_drop = [col for col in graphs[0].node_features.columns if col[2:6] == name]
        graphs = [PreprocessGraphData._drop_feature_columns(g, columns_to_drop) for g in graphs]
        return graphs, columns_to_drop

    @staticmethod
    def dropResidueFeatures(graphs):
        name = 'Res_'
        columns_to_drop = [col for col in graphs[0].node_features.columns if col[2:6] == name]
        graphs = [PreprocessGraphData._drop_feature_columns(g, columns_to_drop) for g in graphs]
        return graphs, columns_to_drop


    @staticmethod
    def run(graphs, flex_level, num_scaler, cat_scaler, drop_res_features, drop_atom_features, polynomial_degree, n_components, user_info):
        user_info.log("Start with preprocessing of input data ...",
                      UserInfo.INFO)
        graphs = FeatureStripping.applyFlexlevelStripping(graphs, flex_level, user_info)
        graphs, n_removedGraphs = PreprocessGraphData.removeGraphsWithNaNFeatures(graphs)
        user_info.log(f"Removed {n_removedGraphs} graphs due to NaN values. {len(graphs)} graphs are left.",
                      UserInfo.INFO)

        feature_names = graphs[0].node_features.columns

        if drop_atom_features:
            graphs, droped_cols = PreprocessGraphData.dropAtomFeatures(graphs)
            user_info.log(f"Removed all Atom features: {droped_cols}")
        elif drop_res_features:
            graphs, droped_cols = PreprocessGraphData.dropResidueFeatures(graphs)
            user_info.log(f"Removed all Residue features: {droped_cols}")

        graphs = DataPreprocessing.scale(graphs, num_scaler, cat_scaler, polynomial_degree, user_info)

        # if n_components:
        #     numerical_features_df, numerical_feature_names = Decomposition.PCA(numerical_features_df, n_components, user_info)
        #
        # all_features_df = pd.concat([numerical_features_df, categorical_features_df], axis=1)
        # all_features_df = all_features_df.sort_index(axis=1)
        # all_feature_names = all_features_df.columns.tolist()
        # graphs = _reconstruct_graphs(graphs, all_features_df)

        all_feature_names = graphs[0].node_features.columns

        user_info.log(f"Done with preprocessing. Started with {len(feature_names)} and ended with {len(all_feature_names)} features.",
                      UserInfo.INFO)

        #sort feature names
        for g in graphs:
            g.sortNodeFeatures()

        return graphs, all_feature_names

class DataSpliter:

    @staticmethod
    def getSplittingTypes():
        types = {
            "Random": DataSpliter.createRandomSplits,
            "GraphRandom": DataSpliter.createRandomGraphsSplits,
            "FarestPointSearch": DataSpliter.createFarestPointsSplits,
            "FarestGraphSearch": DataSpliter.createFarestGraphSplits,
        }
        return types

    @staticmethod
    def _create_random_single_train_test_validate_split(df, ratios, seed):
        sample = df.sample(frac=1, random_state=seed)
        train, test, validate = \
            np.split(
                sample,
                [int(ratios[0] * len(df)), int((ratios[0] + ratios[1]) * len(df))]
            )
        return train[0].tolist(), test[0].tolist(), validate[0].tolist()


    @staticmethod
    def createRandomGraphsSplits(graphs, n_splits, ratios, seed, user_info):
        assert len(ratios) == 3, "Three ratios must be provided"
        assert sum(ratios) == 1, "The sum of the ratios must be 1"

        # To make results reproducable, sort the graphs!
        graphs.sort(key=lambda graph: graph.name)

        user_info.log(
            f"Split graphs in {n_splits} groups: train ({ratios[0]:.1f}, {len(graphs) * ratios[0]:.0f}), "
            f"test ({ratios[1]:.1f}, {len(graphs) * ratios[1]:.0f}), "
            f"validation ({ratios[2]:.1f}, {len(graphs) * ratios[2]:.0f}).",
            UserInfo.INFO)

        df = pd.DataFrame(graphs)
        sample = df.sample(frac=1, random_state=seed)
        split_size = len(df) / n_splits
        sections = [int(split_size * i) for i in range(1, n_splits)]
        splits = np.split(sample, sections)
        enum = [i for i in range(len(splits))]
        liste = []
        for i in range(n_splits):
            permutation = enum[:]
            np.random.shuffle(permutation)
            #train = splits[permutation[0:3]][0].tolist()
            #test = splits[[permutation[3]][0].tolist()
            #vali = splits[permutation[4]][0].tolist()


        splits = []
        for i in range(n_splits):
            train, test, validate = DataSpliter._create_random_single_train_test_validate_split(df, ratios, seed + i)
            splits.append(Data(train, test, validate))

        return splits

    @staticmethod
    def createRandomSplits(graphs, n_splits, ratios, seed, user_info):
        node_ids, x, y = extract_info_from_graphs(graphs)
        pass


    @staticmethod
    def createFarestPointsSplits(graphs, n_splits, ratios, seed, user_info):
        pass

    @staticmethod
    def createFarestGraphSplits(graphs, n_splits, ratios, seed, user_info):
        pass
