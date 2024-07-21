# ML Flow Advanced Workflows

## Autolog

The Autolog feature in MLflow is great for low code / no code scenarios where the autologging feature automatically tracks the parameters and data from the model whenever we call the `fit()` method. During the training process the model parameters, model metrics on the training data and training dataset along with the model signatures are logged.

This also logs the model and its artifacts and dependencies along with it.

During calling the `predict()` method, the test data and the output schema is logged.

To log any test metrics just import the metrics after the autolog statement and they are automatically logged wherever they are used in the code.

Autologging also creates necessary plots according to the model type used - "regression" or "classification".
Any plots created in the code after the autolog statement are also logged.

>[!Imp]  
> There is no need to specifically call the `mlflow.start_run()` while autologging.

## HyperParameter Tuning

The Hyperparameter tuning runs are captured on mlflow through parent child runs.

The **Parent Run** captures the information of the best parameters and the best estimator that has resulted from the tuning process.
The **Child Runs** are all `nested` runs which are logged under the parent run and capture the information of the parameter combination and the associated metrics for that run

>[!Imp]  
>The parent run also logs the model, model artifacts and data.
>It also captures the evaluation scores run on test data by the best estimator.

```python
import mlflow

with mlflow.start_run() as parent:
    ... # code for best estimator logging
    for trial in trials:
        with mlflow.start_run(nested=True) as child:
            ... # code for nested run logging
```
