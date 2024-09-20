import os.path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import sklearn.ensemble as SKLearnEnsemble
import sklearn.tree as SKLearnTree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.utils import all_estimators

from sklearn.base import is_regressor
from src.Plotter import Plotter
from src.UserInfo import UserInfo
from src.Validation import Validation, print_validation, ValidationAllSplits, ValidationGraphs
from src.GraphData import extract_info_from_graphs


class Ensemble:

    @staticmethod
    def saveModel(model, path, user_info):
        import joblib
        joblib.dump(model, path)
        user_info.log(f'Saved ensemble model in {os.path.abspath(path)}',
                      UserInfo.INFO)

    @staticmethod
    def loadModel(path, user_info):
      if os.path.exists(path):
        import joblib
        loaded_model = joblib.load(path)
        user_info.log(f'Loaded ensemble model from {os.path.abspath(path)}',
                      UserInfo.INFO)
        return loaded_model
      else:
        user_info.log(f'Could not load ensemble model from {os.path.abspath(path)}',
                      UserInfo.ERROR)
        exit(1)
      return

    @staticmethod
    def _create_mini_batches(array, batch_size):
        num_splits = len(array) // batch_size + 1
        split_arrays = [array[i * batch_size:(i + 1) * batch_size] for i in range(num_splits)]
        return split_arrays

    @staticmethod
    def _fit_mini_batch(ensemble_model, X_train_batches, y_train_batches):
        for _ in range(10):  # 10 passes through the data
            for X, y in zip(X_train_batches, y_train_batches):
                ensemble_model.fit(X, y)
                ensemble_model.n_estimators += 1  # increment by one so next  will add 1 tree

    @staticmethod
    def run(data, seed, feature_names, n_threads, plot_name, calc_feature_importance: bool, user_info: UserInfo = None):
        #rf_opt = RandomForestOptimizer.opimizeRandomForest(data, seed, n_threads, user_info)
        rf_opt = SKLearnEnsemble.RandomForestRegressor(
            n_estimators=100,
            max_features='sqrt'
        )
        #criterion{“squared_error”, “absolute_error”, “friedman_mse”, “poisson”}
        opt_params = rf_opt.get_params()
        new_rf = SKLearnEnsemble.RandomForestRegressor()
        new_rf.set_params(**opt_params)
        new_rf.warm_start = True
        new_rf.random_state = seed
        new_rf.n_jobs = n_threads

        batch_size = 20000
        user_info.log("Start training 'Ensemble' models ...")

        models = {
            # "RandomForestRegressor Optimized":
            #     new_rf,
            "RandomForestRegressor Default":
                SKLearnEnsemble.RandomForestRegressor(
                    warm_start=True,
                    random_state=seed,
                    n_jobs=n_threads
                ),
            # "HistGradientBoostingRegressor":
            #     SKLearnEnsemble.HistGradientBoostingRegressor(
            #         warm_start=True,
            #         random_state=seed,
            #     ),
            # "GradientBoostingRegressor":
            #     SKLearnEnsemble.GradientBoostingRegressor(
            #         warm_start=True,
            #         random_state=seed,
            #     ),
            # "AdaBoostRegressor N Esti 100":
            #     SKLearnEnsemble.AdaBoostRegressor(
            #         estimator=SKLearnTree.DecisionTreeRegressor(),
            #         random_state=seed,
            #         n_estimators=100,  # default 50
            #     ),
            # "BaggingRegressor":
            #     SKLearnEnsemble.BaggingRegressor(
            #         estimator=SKLearnTree.DecisionTreeRegressor(),
            #         warm_start=True,
            #         random_state=seed,
            #         n_estimators=20, #default 10
            #         n_jobs=n_threads
            #     ),
        }

        names_to_poop = []
        predictions_dict = dict()
        validations_test = dict()
        validations_train = dict()
        for name, model in models.items():
            vali_graphs = ValidationGraphs(name)
            vals_train = []
            vals_test = []
            predictions = []
            real =  []
            nodes = []
            features = []
            for i, split in enumerate(data):
                train_node_ids, train_X, train_Y = extract_info_from_graphs(split.train)
                test_node_ids, test_X, test_Y = extract_info_from_graphs(split.test)
                train_X_batches = Ensemble._create_mini_batches(train_X, batch_size)
                train_Y_batches = Ensemble._create_mini_batches(train_Y, batch_size)
                try:
                    if hasattr(model, 'warm_start') and model.warm_start:
                        Ensemble._fit_mini_batch(model, train_X_batches, train_Y_batches)
                    else:
                        model.fit(train_X, train_Y)
                    user_info.log(f"Done fitting model {name}, Split ({i+1}/{len(data)})",
                                  UserInfo.INFO)
                    predicted_y = model.predict(test_X)
                    predictions.extend(predicted_y.tolist())
                    real.extend(test_Y.tolist())
                    nodes.extend(test_node_ids)
                    features.extend(test_X)
                    vals_train.append(Validation(name, train_node_ids, train_X, model.predict(train_X), train_Y))
                    vals_test.append(Validation(name, test_node_ids, test_X, predicted_y, test_Y))
                    vali_graphs.addGraphs(model, (split.test + split.validation))
                except Warning as w:
                    user_info.log(f"A warning occurred while fitting the model {name}: {w}",
                                  UserInfo.WARNING)
                    break
                except Exception as e:
                    user_info.log(f"An error occurred while fitting the model {name}: {e}",
                                  UserInfo.ERROR)
                    names_to_poop.append(name)
                    break
                #break
            if vals_train:
                predictions = np.array(predictions)
                real = np.array(real)
                validations_test[name] = ValidationAllSplits(name, vals_test)
                validations_train[name] = ValidationAllSplits(name, vals_train)
                predictions_dict[name] = predictions
                user_info.log(f"Train:\n{validations_train[name]}")
                user_info.log(f"Test:\n{validations_test[name]}")
                newValiadtion = Validation(name, nodes, features, predictions, real)
                newValiadtion.get_top_N_worst_predictions(100)
                user_info.log(f"{vali_graphs}")

        for name in names_to_poop:
            models.pop(name)

        all_test_y = np.concatenate([graph.y for split in data for graph in split.test])

        Plotter.plotModelsValidationScatter(
            models,
            predictions_dict,
            real,
            f"{plot_name}_trained_ensemble_model_results_validation_scatter.png",
            user_info=user_info
        )


        if calc_feature_importance:
            user_info.log("Start calculating feature importance ...")
            all_X = np.concatenate([graph.node_features for split in data for graph in split.test + split.train + split.validation]).tolist()
            all_Y = np.concatenate([graph.y for split in data for graph in split.test + split.train + split.validation]).tolist()

            rf_model = models["RandomForestRegressor Default"]
            if "RandomForestRegressor Optimized" in models:
                rf_model = models["RandomForestRegressor Optimized"]

            Plotter.plotFeatureImportanceImpurityBased(
                rf_model,
                f'default_RF_{plot_name}',
                feature_names,
                user_info=user_info
            )

            Plotter.plotFeatureImportancePermutationBased(
                rf_model,
                all_X,
                all_Y,
                f'default_RF_{plot_name}',
                feature_names,
                n_threads,
                user_info=user_info
            )

        Ensemble.saveModel(
            models["RandomForestRegressor Default"],
            f"{plot_name}_random_forest_model.joblib",
            user_info
        )
