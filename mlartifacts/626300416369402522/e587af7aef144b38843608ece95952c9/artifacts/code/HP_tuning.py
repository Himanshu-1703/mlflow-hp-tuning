import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay
from mlflow.sklearn import log_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# load the data
X, y = load_breast_cancer(return_X_y=True, as_frame=True)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=30)

# set tracking server
mlflow.set_tracking_uri("http://127.0.0.1:8000")

# mlflow set experiment name
mlflow.set_experiment("HP Tuning Grid Search")

# parameters grid
params = {
    "clf__n_estimators": [50,100,150,200],
    "clf__max_depth": [3,4,5,6,7,8],
    "clf__random_state": [42],
}

# train the model
model_pipe = Pipeline(steps=[
    ('scale',StandardScaler()),
    ('clf',RandomForestClassifier())
])

# fit the grid search object
grid_search = GridSearchCV(estimator=model_pipe,
                           param_grid=params,
                           cv=5,
                           scoring=['accuracy','recall'],
                           n_jobs=-1,
                           verbose=3,
                           refit='recall',
                           return_train_score=True)


# fit the model
grid_search.fit(X_train,y_train)


# log with mlflow
with mlflow.start_run(run_name='best model') as parent:
    # log the best model score
    mlflow.log_metric("best recall score",grid_search.best_score_)
    
    # log the best parameters
    mlflow.log_params(grid_search.best_params_)
    
    # model signature
    signature = mlflow.models.infer_signature(model_input=X_test,
                                              model_output=grid_search.predict(X_test))
    
    # log the best model
    log_model(sk_model=grid_search.best_estimator_,
              artifact_path="Grid Search",
              input_example=X_train.head(),
              signature=signature,
              registered_model_name='Cancer-RandomForest')
    
    # log the code
    mlflow.log_artifact(__file__,"code")
    
    # log the confusion matrix
    cm = (
        ConfusionMatrixDisplay
        .from_estimator(grid_search.best_estimator_,X_test,y_test)
        .figure_
    )
    mlflow.log_figure(cm,"confusion_matrix.png")
    
    # log the individual hyperparameter runs
    with mlflow.start_run(nested=True) as child:
        results = grid_search.cv_results_
        for ind in range(len(results['mean_test_accuracy'])):
            # test metrics
            test_metrics = {
                "test_accuracy": results["mean_test_accuracy"][ind],
                "test_recall": results["mean_test_recall"][ind]
            }
            
            # train metrics
            train_metrics = {
                "train_accuracy": results["mean_train_accuracy"][ind],
                "train_recall": results["mean_train_recall"][ind]
            }

            # log the train metrics
            mlflow.log_metrics(train_metrics)
            # log the test metrics
            mlflow.log_metrics(test_metrics)
            
            # log the run hyperparameters
            mlflow.log_params(results['params'][ind])
            
    # get predictions from the best model on the test data
    y_pred = grid_search.predict(X_test)

    # get the test metrics
    metrics = {
    "accuracy": accuracy_score(y_test,y_pred),
    "precision": precision_score(y_test,y_pred),
    "recall": recall_score(y_test,y_pred),
    "f1": f1_score(y_test,y_pred)
    }
    
    # log the test metrics 
    mlflow.log_metrics(metrics)
    
    # log the train and test data
    # log training data
    train_df = X_train.join(y_train)
    mlflow.log_input(mlflow.data.from_pandas(train_df),context='training')
    
    val_df = X_test.join(y_test)
    mlflow.log_input(mlflow.data.from_pandas(val_df),context='validation')
    