import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

from src.UserInfo import UserInfo

def get_string_for_estimator(estimator):
    #params = estimator.get_params()
    #print(params)
    string = estimator.__str__().replace("(", "_").replace(",", "-").replace(")", "").replace("\n", "")
    return string.replace("'","").replace(" ", "")

def print_sorted_features_importances(name, sorted_series):
    print(name)
    print(sorted_series.to_string(float_format='%.8f'))


class Plotter:
    @staticmethod
    def plotFeatureImportanceImpurityBased(randomForestModel, plot_name, feature_names, user_info):
        user_info.log("Plotting feature importance impurity based.")
        importances = randomForestModel.feature_importances_
        #std = np.std([tree.feature_importances_ for tree in randomForestModel.estimators_], axis=0)

        forest_importances = pd.Series(importances, index=feature_names)
        sorted_importances = forest_importances.sort_values()
        print_sorted_features_importances("Feature importance impurity based:", sorted_importances)

        #plt.figure()
        fig, ax = plt.subplots(figsize=(10, 15))
        #sorted_importances.plot.barh(yerr=std, ax=ax)
        sorted_importances.plot.barh(ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_xlabel("Mean decrease in impurity")
        #ax.set_yticklabels(labels=feature_names ,ha="left", va='center', position=(-0.75,0))
        fig.tight_layout()
        model_string = get_string_for_estimator(randomForestModel)
        path = os.path.abspath(f"{plot_name}_feature_importance_impurity_random_forest_{model_string}.svg")
        plt.savefig(path)
        user_info.log(f'Saved feature importance impurity-based figure in {path}',
                      UserInfo.INFO)
        plt.close()

    @staticmethod
    def plotFeatureImportancePermutationBased(randomForestModel, X_test, y_test, plot_name, feature_names, n_threads, user_info):
        user_info.log("Plotting feature importance permutation based.")
        from sklearn.inspection import permutation_importance
        result = permutation_importance(
            randomForestModel,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            n_jobs=1, # THIS MUST BE 1 IF THE MODEL USES N_JOBS=N_THREAD!!!
        )
        importances = result.importances_mean
        #std = result.importances_std

        forest_importances = pd.Series(importances, index=feature_names)
        sorted_importances = forest_importances.sort_values()
        print_sorted_features_importances("Feature importance permutation based:", sorted_importances)

        #plt.figure()
        fig, ax = plt.subplots(figsize=(10, 15))
        #sorted_importances.plot.barh(yerr=std, ax=ax)
        sorted_importances.plot.barh(ax=ax)
        ax.set_title("Feature importances using permutation on full model")
        ax.set_xlabel("Mean accuracy decrease")
        #ax.set_yticklabels(labels=feature_names, ha="left", va='center', position=(-0.75,0))
        fig.tight_layout()
        model_string = get_string_for_estimator(randomForestModel)
        path = os.path.abspath(f"{plot_name}_feature_importance_permutation_random_forest_{model_string}.svg")
        plt.savefig(path)
        user_info.log(f'Saved feature importance permutation-based figure in {path}',
                      UserInfo.INFO)
        plt.close()

    @staticmethod
    def plotModelsValidationScatter(models_dict, predictions_dict, y_test, fig_name, user_info=None):
        # Determine number of plots (models)
        num_plots = len(models_dict)

        # Calculate the optimal grid size for (num_plots)
        if num_plots == 1:
            nof_rows, nof_cols = 1, 1
        elif num_plots == 2:
            nof_rows, nof_cols = 1, 2
        elif num_plots <= 4:
            nof_rows, nof_cols = 2, 2
        elif num_plots <= 9:
            nof_rows, nof_cols = 3, 3
        elif num_plots <= 16:
            nof_rows, nof_cols = 4, 4
        elif num_plots <= 20:
            nof_rows, nof_cols = 5, 4
        else:
            nof_rows = 5
            nof_cols = (num_plots + nof_rows - 1) // nof_rows  # Ensuring enough columns to fit all models

        size = 6
        plt.figure(figsize=(size * nof_cols, size * nof_rows))

        for i, (name, model) in enumerate(models_dict.items()):
            if name == "RadiusNeighborsRegressor":
                break
            mse = mean_squared_error(y_test, predictions_dict[name])
            mae = mean_absolute_error(y_test, predictions_dict[name])
            r2 = r2_score(y_test, predictions_dict[name])
            pearson = pearsonr(y_test, predictions_dict[name]).correlation

            plt.subplot(nof_rows, nof_cols, i + 1)
            plt.scatter(y_test, predictions_dict[name], alpha=0.5,
                        label=f'R2: {r2:.2f}, Pearson: {pearson:.2f}, MAE: {mae:.2f}, MSE: {mse:.2f}')

            # Add a reference line
            range_vals = [np.min(y_test), np.max(y_test)]
            plt.plot(range_vals, range_vals, '--k')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(name)
            plt.legend()
            plt.grid(True)

            # Logging information
            user_info.log(f'{name:<35}\tR2: {r2:.2f}\tPearson: {pearson:.2f}\tMAE: {mae:.2f}\tMSE: {mse:.2f}',
                          UserInfo.INFO)

        plt.tight_layout()
        path = os.path.abspath(fig_name)
        plt.savefig(path)
        user_info.log(f'Saved scatter validation figure in {path}', UserInfo.INFO)
        plt.close()

    @staticmethod
    def plotValidationScatter(predictions, y_test, fig_name, user_info: UserInfo = None):
        # plt.figure()
        plt.figure()
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        pearson = pearsonr(y_test, predictions).correlation
        r2 = r2_score(y_test, predictions)
        plt.scatter(y_test, predictions, alpha=0.5, label=f'R2: {r2:.2f}, Pearson: {pearson:.2f}, MAE: {mae:.2f}, MSE: {mse:.2f}')
        range = [np.min(y_test), np.max(y_test)]
        plt.plot(range, range, '--k')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        name = ""
        plt.title(name)
        plt.legend()
        plt.grid(True)
        user_info.log(f'{name:<35}\tR2: {r2:.2f}\tPearson: {pearson:.2f}\tMAE: {mae:.2f}\tMSE: {mse:.2f}',
                      UserInfo.INFO)
        plt.tight_layout()
        path = os.path.abspath(fig_name)
        plt.savefig(path)
        plt.close()

        plt.figure()
        diff = abs(y_test - predictions)
        plt.violinplot(diff)
        plt.savefig(f'{path}_violinplot.png')
        plt.close()

        plt.figure()
        diff = abs(y_test - predictions)
        plt.boxplot(diff)
        plt.savefig(f'{path}_boxplot.png')
        plt.close()

        user_info.log(f'Saved scatter-, box- nad violin-plot figures in {path}',
                      UserInfo.INFO)


    @staticmethod
    def plot_losses(train_losses, test_losses, val_losses, loss_name: str = "MSE", plot_name: str = "train_vs_test"):
        plt.figure()
        max_y = np.mean(train_losses) + np.std(train_losses)
        plt.ylim(0, max_y)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel(f'Loss {loss_name}')
        plt.title('Train, Test and Validation Loss')
        plt.legend()
        plt.savefig(f'{plot_name}.png')
        plt.close()

    @staticmethod
    def plot_correlations(x, y, name, correlation, plot_name, user_info):
        path = os.path.abspath(f'{plot_name}.png')
        plt.close()
        plt.figure()
        plt.scatter(x, y, label=f'Correlation: {correlation:.2f}')
        plt.xlabel(f'{name}')
        plt.ylabel(f'Real RMSF')
        plt.title(f'Feature correlation: {name}')
        plt.legend()
        plt.savefig(path)
        user_info.log(f'Saved scatter validation figure in {path}',
                      UserInfo.INFO)
        plt.close()