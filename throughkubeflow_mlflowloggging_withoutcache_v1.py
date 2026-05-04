from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset, Model

# 🔥 IMPORTANT: use this for local MLflow
MLFLOW_URI = "http://host.docker.internal:30007"


# =========================
# Step 1: Load Data
# =========================
@dsl.component(base_image="python:3.9")
def load_data(output_csv: Output[Dataset]):
    import subprocess
    subprocess.run(["pip", "install", "pandas", "scikit-learn"], check=True)

    from sklearn.datasets import load_iris
    import pandas as pd

    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    df.to_csv(output_csv.path, index=False)


# =========================
# Step 2: Preprocess (NO SCALING HERE ❌)
# =========================
@dsl.component(base_image="python:3.9")
def preprocess_data(
    input_csv: Input[Dataset],
    X_train_out: Output[Dataset],
    X_test_out: Output[Dataset],
    y_train_out: Output[Dataset],
    y_test_out: Output[Dataset]
):
    import subprocess
    subprocess.run(["pip", "install", "pandas", "scikit-learn"], check=True)

    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_csv.path)

    X = df.drop(columns=["target"])
    y = df["target"].values

    # ✅ Only split (NO scaling)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pd.DataFrame(X_train).to_csv(X_train_out.path, index=False)
    pd.DataFrame(X_test).to_csv(X_test_out.path, index=False)
    pd.DataFrame(y_train).to_csv(y_train_out.path, index=False)
    pd.DataFrame(y_test).to_csv(y_test_out.path, index=False)


# =========================
# Step 3: Train + MLflow (WITH PIPELINE ✅)
# =========================
@dsl.component(base_image="python:3.9")
def train_model(
    X_train: Input[Dataset],
    y_train: Input[Dataset],
    X_test: Input[Dataset],
    y_test: Input[Dataset],
    model_out: Output[Model]
):
    import subprocess
    subprocess.run(["pip", "install", "mlflow", "pandas", "scikit-learn", "joblib"], check=True)

    import pandas as pd
    import mlflow
    import mlflow.sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from joblib import dump

    # ✅ MLflow setup
    mlflow.set_tracking_uri("http://host.docker.internal:30007")
    mlflow.set_experiment("kubeflow-iris")

    X_train = pd.read_csv(X_train.path)
    y_train = pd.read_csv(y_train.path).values.ravel()
    X_test = pd.read_csv(X_test.path)
    y_test = pd.read_csv(y_test.path).values.ravel()

    with mlflow.start_run():

        # ✅ FULL PIPELINE (Scaler + Model)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=200))
        ])

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # ✅ Logs
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)

        # ✅ Log FULL pipeline
        mlflow.sklearn.log_model(pipeline, "model")

        # ✅ Register model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, "iris-classifier")

        # ✅ Save model artifact
        dump(pipeline, model_out.path)


# =========================
# Pipeline
# =========================
@dsl.pipeline(name="mlflow-kubeflow-pipeline")
def pipeline():

    load = load_data().set_caching_options(False)

    preprocess = preprocess_data(
        input_csv=load.outputs["output_csv"]
    ).set_caching_options(False)

    train = train_model(
        X_train=preprocess.outputs["X_train_out"],
        y_train=preprocess.outputs["y_train_out"],
        X_test=preprocess.outputs["X_test_out"],
        y_test=preprocess.outputs["y_test_out"]
    ).set_caching_options(False)


# =========================
# Compile YAML
# =========================
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="mlflow_kubeflow_pipeline.yaml"
    )