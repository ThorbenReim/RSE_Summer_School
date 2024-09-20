import numpy as np
import pandas as pd
from itertools import product

from src.Ensemble import Ensemble
from src.UserInfo import UserInfo
from src.GraphData import extract_info_from_graphs
from src.Validation import ValidationAllSplits
from src.Plotter import Plotter


class MachineLearning:
    @staticmethod
    def get_methods_string_types():
        methods = [
            'all',
            'ensemble',
            'linear',
        ]
        return methods

    @staticmethod
    def get_flexibility_level_types():
        levels = [
            'allatoms',
            'mostflexibleresidueatom',
            'betacarbon',
            'alphacarbon',
            'sidechain',
            'backbone',
            'transition'
        ]
        return levels

    @staticmethod
    def runEnsemble(data, seed, feature_names, plot_name, calc_feature_importance: bool, n_threads: int = 1, user_info: UserInfo = None):
        Ensemble.run(data, seed, feature_names, n_threads, plot_name, calc_feature_importance, user_info=user_info)

    @staticmethod
    def runLinear(data, seed, feature_names, plot_name, n_threads: int = 1, user_info: UserInfo = None):
        LinearRegression.run(data, seed, feature_names, n_threads, plot_name, user_info=user_info)
