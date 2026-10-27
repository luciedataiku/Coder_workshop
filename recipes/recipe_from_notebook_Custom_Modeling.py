# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import pandas as pd
import mlflow
import warnings

from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from dataikuapi.dss.ml import DSSPredictionMLTaskSettings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Replace these constants by your own values
EXPERIMENT_TRACKING_FOLDER_NAME = "tracking"
EXPERIMENT_TRACKING_FOLDER_CONNECTION = "s3-managed"
EXPERIMENT_NAME = "custom-modeling"

MLFLOW_CODE_ENV_NAME = "38"
SAVED_MODEL_NAME = "custom-model"
DATASET_TRAINING = "train"

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Some utils
def now_str() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Experiment tracking (scikit-learn)
# 
# This notebook contains a simple example to showcase the new Experiment Tracking capabilities of Dataiku. It explains how to perform several runs with different parameters, select the best run and promote it as a Saved Model version in a Dataiku Flow. It leverages:
# * the scikit-learn package

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Loading the training data
# 
# Our training data lives in the `labeled` Dataset, let's load it in a pandas DataFrame and see what it looks like:

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
client = dataiku.api_client()
project = client.get_default_project()
training_dataset = dataiku.Dataset(DATASET_TRAINING)
df = training_dataset.get_dataframe()
df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# We are working on a *binary classification* problem here, which is to predict whether or not a given customer is high value. This outcome is reflected by the `high_value` column which can either take the "0.0" or "1.0" values.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
target_name = "high_value"
target = df[target_name]
data = df.drop(columns=[target_name])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Get-or-create Managed Folder (WIP)
project_folders = project.list_managed_folders()
folder = None
if len(project_folders) > 0:
    for mf in project_folders:
        if mf["name"] == EXPERIMENT_TRACKING_FOLDER_NAME:
            folder_id = mf["id"]
            print(f"Found experiment tracking folder {EXPERIMENT_TRACKING_FOLDER_NAME} with id {mf['id']}")
            folder = project.get_managed_folder(odb_id=folder_id)
            break
        else:
            continue
    # -- If you reach this point, you didn't find the experiment tracking folder among the existing ones.
    if not folder:
        print("Experiment tracking folder not found. Creating it...")
        folder = project.create_managed_folder(EXPERIMENT_TRACKING_FOLDER_NAME,
                                   connection_name=EXPERIMENT_TRACKING_FOLDER_CONNECTION)
else:
    print("No folder found in project. Creating one for experiment tracking...")
    # Write the creation of the mf code here.
    folder = project.create_managed_folder(EXPERIMENT_TRACKING_FOLDER_NAME,
                                       connection_name=EXPERIMENT_TRACKING_FOLDER_CONNECTION)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Preparing the experiment
# 
# To prepare the grounds for our experiments, we need to create a few handles and define which MLFlow experiment we'll collect our runs into:

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Create a mlflow_extension object to easily collect information for the promotion step
mlflow_extension = project.get_mlflow_extension()

# Create a handle for the mlflow client
mlflow_handle = project.setup_mlflow(managed_folder=folder)

# Set the experiment
experiment = mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Experimenting
# 
# The goal of experiment tracking is to *instrument the iterative process of ML model training* by collecting all parameters and results of each trial. To be more specific, within an **experiment**, you perform multiple **runs**, each run being different from the others because of the **parameters** you use for it. You also need to specific which **metrics** to track, they will reflect the performance of the model for a given set of parameters.
# 
# In this notebook example, if you want to produce experiment runs:
# * edit the parameters in the 3.1 cell and run it
# * run the 3.2 cell to effectively... perform the run 🙂

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Defining the parameters of our run

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Create run name
run_params = {}
run_metrics = {}

# Define run parameters
# -- Which columns to retain ?
categorical_cols = ["gender", "ip_country_code"]
run_params["categorical_cols"] = categorical_cols
numerical_cols = ["age", "price_first_item_purchased", "pages_visited", "campaign"]
run_params["numerical_cols"] = numerical_cols

# --Which algorithm to use? Which hyperparameters for this algo to try?
# --- Example: Random Forest
hparams = {"n_estimators": 50,
           "criterion": "gini",
           "max_depth": 5,
           "min_samples_split": 3,
           "random_state": 42}
clf = RandomForestClassifier(**hparams)
model_algo = type(clf).__name__
run_params["model_algo"] = model_algo
for hp in hparams.keys():
    run_params[hp] = hparams[hp]

# --Which cross-validation settings to use?
n_cv_folds = 5
cv = StratifiedKFold(n_splits=n_cv_folds)
run_params["n_cv_folds"] = n_cv_folds
metrics = ["f1_macro", "roc_auc"]

# --Let's print all of that to get a recap:
print(f"Parameters to log:\n {run_params}")
print(100*'-')
print(f"Metrics to log:\n {metrics}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Performing the run and logging parameters, metrics and the model

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
run_ts = now_str()
run_name = f"run-{run_ts}"
with mlflow.start_run(run_name=run_name) as run:
    run_id = run.info.run_id
    print(f"Starting run {run_name} (id: {run_id})...")
    # --Preprocessing
    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer([
        ('categorical', categorical_preprocessor, categorical_cols),
        ('numerical', 'passthrough', numerical_cols)])

    # --Pipeline definition (preprocessing + model)
    pipeline = make_pipeline(preprocessor, clf)

    # --Cross-validation
    print(f"Running cross-validation...")
    scores = cross_validate(pipeline, data, target, cv=cv, scoring=metrics)
    for m in [f"test_{mname}" for mname in metrics]:
        run_metrics[f"mean_{m}"] = scores[m].mean()
        run_metrics[f"std_{m}"] = scores[m].std()

    # --Pipeline fit
    pipeline.fit(X=data, y=target)
    # --Log the order of the class label
    run_params["class_labels"] = [str(c) for c in pipeline.classes_.tolist()]

    # --Log parameters, metrics and model
    mlflow.log_params(params=run_params)
    mlflow.log_metrics(metrics=run_metrics)
    artifact_path = f"{model_algo}-{run_id}"
    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path=artifact_path)

    # --Set useful information to faciliate run promotion
    mlflow_extension.set_run_inference_info(run_id=run_id,
                                            prediction_type="BINARY_CLASSIFICATION",
                                            classes=run_params["class_labels"],
                                            code_env_name=MLFLOW_CODE_ENV_NAME,
                                            target="high_value")
    print(f"DONE! Your artifacts are available at {run.info.artifact_uri}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
tjdsem0b = dataiku.Folder("TjDsem0B")
tjdsem0b_info = tjdsem0b.get_info()