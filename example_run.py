from src.Prediction import Prediction
from src.UserInfo import UserInfo
from main import Train
import os
import sys

if len(sys.argv) != 2:
    sys.stderr.write(f"{sys.argv[0]}: path_for_results\n")
    exit(1)

resuts_path = os.path.abspath(sys.argv[1])
if not os.path.exists(resuts_path):
    sys.stderr.write(f"Your path does not exists: {resuts_path}\n")
    exit(1)

user_info = UserInfo('DETAILED_INFO')

file_prefix_name = "trained_random_forest"
file_prefix_path = os.path.join(resuts_path, file_prefix_name)
model_path = os.path.join(resuts_path, file_prefix_name + "_random_forest_model.joblib")

Train.run(
    [],
    "data/descriptors/",
    "ensemble",
    [],
    [],
    10,
    5,
    [0.6, 0.2, 0.2],
    file_prefix_path,
    "allatoms",
    "No",
    "No",
    False,
    False,
    0,
    0,
    False,
    0,
    user_info
)

Prediction.run(
    "data/pdb/1ab1.pdb",
    "data/descriptors/1ab1.txt",
    model_path,
    [],
    [],
    resuts_path,
    user_info
)