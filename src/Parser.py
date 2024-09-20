import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from src.UserInfo import UserInfo
from src.GraphData import GraphData
from src.Utils import Utils

class ParserError(Exception):
    pass

class Parser:
    def __init__(self):
        pass

    def _convertToTorchCompatibleGraph(self, feature_names, df, edges, name):
        node_ids = df['InfileID']
        df['InfileID'] = df['InfileID'].apply(Utils.get_atom_id).astype(int)
        infile_ID_to_index = df['InfileID'].reset_index().set_index('InfileID')
        series = infile_ID_to_index['index']

        # Filter and map edges efficiently
        valid_edges = edges[
            (edges['InfileID1'].isin(infile_ID_to_index.index)) & (edges['InfileID2'].isin(infile_ID_to_index.index))]

        # Use .loc to access using label-based indexing
        def get_index(idx, series):
            res = series.loc[idx] if idx in series.index else None
            return res

        edges_index = valid_edges[['InfileID1', 'InfileID2']].apply(
            lambda x: [get_index(x['InfileID1'], series), get_index(x['InfileID2'], series)],
            axis=1
        ).dropna().tolist()  # Drop rows where index mapping wasn't successful

        edge_features = valid_edges[['CovalentBond', 'HBond', 'Distance']].values.tolist()

        features_df_sorted = df[feature_names].sort_index(axis=1)

        y = df['RMSF'].to_numpy()

        return GraphData(
            y,
            features_df_sorted,
            edges_index,
            edge_features,
            node_ids,
            name
        )

    def readGraphDescriptorFile(self, file_path, features_keep, features_strip, user_info):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File path '{file_path}' does not exist.")

        name = os.path.basename(file_path).split(".")[0]

        with open(file_path, 'r') as myfile:
            lines = myfile.readlines()
        edge_start_line_list = [i for i, line in enumerate(lines) if 'Edges_v3' in line]
        if not edge_start_line_list:
            raise ParserError("The file does not contain entry 'Edges_v3' or any edge data.")
        edge_start_line = edge_start_line_list[0]
        data_lines = lines[:edge_start_line]
        edge_lines = lines[edge_start_line + 1:]

        df = pd.read_csv(StringIO("\n".join(data_lines)))

        required_columns = list(features_keep + ['InfileID', 'RMSF']) if features_keep else list(
            set(df.columns) - set(features_strip))
        df = df[required_columns]

        edges = pd.read_csv(StringIO("\n".join(edge_lines)))

        feature_names = [col for col in df.columns if col != 'InfileID' and col != 'RMSF']

        graph = self._convertToTorchCompatibleGraph(feature_names, df, edges, name)
        return feature_names, graph

    def parseDescriptorFiles(self, file_paths, directory, features_keep, features_strip, n_threads,
                             user_info: UserInfo = None):
        all_file_paths = Utils.get_all_files(directory, file_paths, ".txt")
        last_feature_names = None
        graphs = []

        def process_file(file_path):
            if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
                return None, None

            try:
                feature_names, graph = self.readGraphDescriptorFile(file_path, features_keep, features_strip, user_info)
                if graph is None or not graph.node_features.size or not graph.adj_matrix.size or not graph.y.size:
                    user_info.log(f"The file '{file_path}' has missing data.", UserInfo.WARNING)
                    return None, None
                return feature_names, graph
            except Exception as e:
                user_info.log(f"Error processing file '{file_path}': {e}", UserInfo.ERROR)
                return None, None

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            future_to_file = {executor.submit(process_file, file_path): file_path for file_path in
                              sorted(all_file_paths)}

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                feature_names, graph = future.result()
                if graph is None:
                    continue

                # Ensure last_feature_names is assigned properly
                if last_feature_names is None:
                    last_feature_names = feature_names
                elif feature_names != last_feature_names:
                    user_info.log(f"Inconsistent headers in '{file_path}'.", UserInfo.ERROR)
                    continue

                graphs.append(graph)

        if user_info and last_feature_names:
            user_info.log(f"Number of kept features: {len(last_feature_names)}.")
            user_info.log(f"Features kept: {' '.join(last_feature_names)}.", UserInfo.DETAILED_INFO)

        return graphs, last_feature_names