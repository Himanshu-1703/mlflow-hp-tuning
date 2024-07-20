import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from mlflow.sklearn import autolog

# start the mlflow autolog
autolog(log_input_examples=True)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# load the data
X, y = load_breast_cancer(return_X_y=True, as_frame=True)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=30)

# set tracking server
mlflow.set_tracking_uri("http://127.0.0.1:8000")

# mlflow set experiment name
mlflow.set_experiment("Autologging")

# # model parameters
params = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": 2
}

# train the model
rf = RandomForestClassifier(**params)

model_pipe = Pipeline(steps=[
    ('scale',StandardScaler()),
    ('clf',rf)
])

# fit the model pipeline
model_pipe.fit(X_train,y_train)

# get the model predictions
y_pred = model_pipe.predict(X_test)

# get the metrics
metrics = {
    "accuracy": accuracy_score(y_test,y_pred),
    "precision": precision_score(y_test,y_pred),
    "recall": recall_score(y_test,y_pred),
    "f1": f1_score(y_test,y_pred)
}


# confusion matrix figure
cm_figure = (
    ConfusionMatrixDisplay
    .from_estimator(model_pipe,X_test,y_test)
    .figure_
)