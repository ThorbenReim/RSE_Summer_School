import argparse
import sys
import os
import numpy as np
import time


from src.MachineLearning import MachineLearning
from src.DataProcessing import PreprocessGraphData, DataPreprocessing, DataSpliter
from src.UserInfo import UserInfo
from src.Parser import Parser
from src.Ensemble import Ensemble
from src.Utils import Utils
from src.Prediction import Prediction

class Train:
    @staticmethod
    def run(descriptor_files, directory, methods, feature_keep, feature_strip, n_threads, n_split, split_ratios, plot_name, flex_level, num_scaler, cat_scaler, drop_res_features, drop_atom_features, polynomial_degree, n_components, calc_feature_importance, seed, user_info):
        user_info.log(
            f"Parsing descriptor files ... ")
        p = Parser()
        start_time = time.time()
        # Call the function you want to measure
        graphs, feature_names = p.parseDescriptorFiles(
            descriptor_files,
            directory,
            feature_keep,
            feature_strip,
            n_threads,
            user_info
        )

        user_info.log(
            f"Parsing descriptor files executed in {time.time() - start_time:.2f} seconds.")

        graphs, feature_names = PreprocessGraphData.run(
            graphs, flex_level, num_scaler, cat_scaler, drop_res_features, drop_atom_features, polynomial_degree, n_components, user_info)

        data_splits = DataSpliter.createRandomGraphsSplits(graphs, n_split, split_ratios, seed, user_info)

        MachineLearning.runEnsemble(
            data_splits,
            seed,
            feature_names,
            plot_name,
            calc_feature_importance,
            n_threads,
            user_info)
        
        return 0


################################################################################


class ProteinFlexibilityPredicion(object):

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Framework to train and predict protein flexibility',
            usage='''main.py <command> [<args>]

Commands:
   train    Train a machine learning model
   predict  Predict protein flexibility for an input structure

''')
        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            sys.stderr.write('Unrecognized command\n')
            parser.print_help()
            exit(1)
        getattr(self, args.command)(sys.argv[2:])

    def _add_general_options(self, parser):
        parser.add_argument(
            '-v', '--verbosity',
            type=str,
            choices=UserInfo.get_verbosity_string_types(),
            default='INFO',
            help=f"Set the verbosity level: {' '.join(UserInfo.get_verbosity_string_types())}")
        parser.add_argument(
            '-t', '--threads',
            help="Number of threads",
            default=1,
            type=int
        )

    def _argument_parsing(self, parser, arguments):
        self._add_general_options(parser)
        args = parser.parse_args(arguments)
        return args


    def train(self, arguments):
        parser = argparse.ArgumentParser(
            description='Train, test and validate machine learning methods')
        parser.add_argument(
            '-f', '--descfiles',
            nargs='+',
            help="Atomic flexibility descriptor file, each line containing the real value (y, first column) and the features (X)."
        )
        parser.add_argument(
            '-d', '--directory',
            help="Directory with flexibility descriptor files, each line containing the real value (y, first column) and the features (X)."
        )
        parser.add_argument(
            '-m', '--methods',
            type=str,
            choices=MachineLearning.get_methods_string_types(),
            default=MachineLearning.get_methods_string_types()[0],
            nargs='+',
            help="Methods to chose to train"
        )
        parser.add_argument('--flexlevel',
                            choices=MachineLearning.get_flexibility_level_types(),
                            help='Choose an option from the list, flexibility type.',
                            required=True)

        parser.add_argument(
            '-n', '--plotname',
            help="Base name for the plots created during training."
        )
        parser.add_argument(
            '--features_keep',
            help="Feature names to KEEP from descriptor file",
            nargs='+',
            default=[]
        )
        parser.add_argument(
            '--features_strip',
            help="Feature names to STRIP from descriptor file",
            nargs='+',
            default=[]
        )

        parser.add_argument('-ns','--numericalscaler',
                            choices=DataPreprocessing.get_numercial_scaler().keys(),
                            help='Choose an option from the list, numerical scaler.',
                            required=True)
        parser.add_argument('-pd', '--polynomial_degree',
                            type=int,
                            default=0,
                            help='Polynomial degree for feature preprocessing, if zero, it is not applied.')
        parser.add_argument('-cs', '--categoricalscaler',
                            choices=DataPreprocessing.get_categorical_scaler().keys(),
                            help='Choose an option from the list, categorical scaler.',
                            required=True)
        parser.add_argument('-pca', '--n_components',
                            type=int,
                            default=0,
                            help='Number of components from PCA to keep, if zero, it is not applied.')
        parser.add_argument(
            '-fi', '--featureimportance',
            action='store_true',
            help="Calc feature importance for ensemble method (Random forest)."
        )
        parser.add_argument(
            '-dr', '--dropresidue',
            action='store_true',
            help="Drop all residue features from descriptor."
        )
        parser.add_argument(
            '-da', '--dropatom',
            action='store_true',
            help="Drop all atom features from descriptor."
        )
        parser.add_argument(
            '-s', '--seed',
            type=int,
            default=0,
            help="Set seed."
        )
        parser.add_argument(
            '--nsplits',
            type=int,
            default=5,
            help="Set number of dataset splits for training."
        )
        parser.add_argument(
            '--splitratios',
            type=float,
            nargs='+',
            default=[0.6, 0.2, 0.2],
            help="Dataset train, test, and validation ration."
        )


        # now that we're inside a subcommand, ignore the first
        args = self._argument_parsing(parser, arguments)

        user_info = UserInfo(args.verbosity)

        if len(args.splitratios) != 3:
            sys.stderr.write("Please provide 3 floats with sum of 1 for '--splitratios'.\n")
            exit(1)

        if sum(args.splitratios) != 1:
            sys.stderr.write("The sum for '--splitratios' must be 1.\n")
            exit(1)

        if not args.descfiles and not args.directory:
            sys.stderr.write("Please provide at least option '-d' or '-f'.\n")
            exit(1)

        if args.features_keep and args.features_strip:
            sys.stderr.write("Please provide only one option, either '--features_keep' or '--features_strip'.\n")
            exit(1)

        if args.dropresidue and args.dropatom:
            sys.stderr.write("Please provide only one option, either '--dropresidue' or '--dropatom'.\n")
            exit(1)

        retCode = Train.run(args.descfiles,
                            args.directory,
                            args.methods,
                            args.features_keep,
                            args.features_strip,
                            args.threads,
                            args.nsplits,
                            args.splitratios,
                            args.plotname,
                            args.flexlevel,
                            args.numericalscaler,
                            args.categoricalscaler,
                            args.dropresidue,
                            args.dropatom,
                            args.polynomial_degree,
                            args.n_components,
                            args.featureimportance,
                            args.seed,
                            user_info)

        exit(retCode)

    def predict(self, arguments):
        parser = argparse.ArgumentParser(
            description='Predict protein flexibility for an input structure')
        parser.add_argument(
            "-p","--pdb",
            help="PDB file of the structure, for error plotting"
        )
        parser.add_argument(
            "-d", "--descriptor",
            help="Flexibility descriptor file of the structure"
        )
        parser.add_argument(
            "-m", "--model",
            help="Path to the trained machine learning model",
            type=str,
            required=True
        )
        parser.add_argument(
            '--features_keep',
            help="Feature names to KEEP from descriptor file",
            nargs='+',
            default=[]
        )
        parser.add_argument(
            '--features_strip',
            help="Feature names to STRIP from descriptor file",
            nargs='+',
            default=[]
        )

        args = self._argument_parsing(parser, arguments)
        user_info = UserInfo(args.verbosity)
        if not args.pdb and not args.descriptor:
            user_info.log("Please only provide EITHER option '--pdb' OR '--descriptor'.",
                          UserInfo.ERROR)
            exit(1)
        if args.features_keep and args.features_strip:
            sys.stderr.write("Please provide only one option, either '--features_keep' or '--features_strip'.\n")
            exit(1)

        retCode = Prediction.run(
            args.pdb,
            args.descriptor,
            args.model,
            args.features_keep,
            args.features_strip,
            user_info)
        exit(retCode)

if __name__ == "__main__":
    os.environ["XDG_SESSION_TYPE"] = "xcb"
    #if I do not do this -> Warning: Ignoring XDG_SESSION_TYPE=wayland on Gnome. Use QT_QPA_PLATFORM=wayland to run on Wayland anyway.

    ProteinFlexibilityPredicion()
