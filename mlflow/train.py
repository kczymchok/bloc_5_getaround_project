import argparse
import pandas as pd
import time
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.preprocessing import  StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor 
from sklearn.svm import SVR, LinearSVR
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor


if __name__ == "__main__":

    # Setting experiment
    experiment_name = "GetAround_car_rental_price_predictor"
    client = mlflow.tracking.MlflowClient()

    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)


    # Parse arguments given in shell script 'run.sh' or in Command-Line
    parser = argparse.ArgumentParser()
    parser.add_argument("--regressor", default='LR', choices = ['LR', 'Ridge', 'RF', 'Catboost', 'XGboost', 'DecisionTree'])
    parser.add_argument("--cv", type = int, default = None)
    parser.add_argument("--alpha", type = float, nargs = "*")
    parser.add_argument("--max_depth", type = int, nargs="*")
    parser.add_argument("--min_samples_leaf", type = int, nargs="*")
    parser.add_argument("--min_samples_split", type = int, nargs="*")
    parser.add_argument("--n_estimators", type = int, nargs="*")
    parser.add_argument("--learning_rate", type = float, nargs="*")
    args = parser.parse_args()
    print(args)

    print("training model...")

    # Time execution
    start_time = time.time()

    # Call mlflow autolog
    mlflow.sklearn.autolog(log_models=False)


    # Import dataset
    df = pd.read_csv("https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_pricing_project.csv", index_col = 0)

    # Drop irrelevant rows
    df = df[(df['mileage'] >= 0) & (df['engine_power'] > 0)]

    # X, y split 
    target_col = 'rental_price_per_day'
    y = df[target_col]
    X = df.drop(target_col, axis = 1)

    # Features categorization
    numerical_features = []
    binary_features = []
    categorical_features = []
    for i,t in X.dtypes.items():
        if ('float' in str(t)) or ('int' in str(t)) :
            numerical_features.append(i)
        elif ('bool' in str(t)):
            binary_features.append(i)
        else :
            categorical_features.append(i)

    # Regroup fewly-populated category labels in label 'other'
    for feature in categorical_features:
        label_counts = X[feature].value_counts()
        fewly_populated_labels = list(label_counts[label_counts < 0.5 / 100 * len(X)].index)
        for label in fewly_populated_labels:
            X.loc[X[feature]==label,feature] = 'other'

    # Train / test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    # Features preprocessing 
    categorical_transformer = OneHotEncoder(drop='first')
    numerical_transformer = StandardScaler()
    binary_transformer = FunctionTransformer(None, feature_names_out = 'one-to-one') #identity function
    feature_preprocessor = ColumnTransformer(
            transformers=[
                ("categorical_transformer", categorical_transformer, categorical_features),
                ("numerical_transformer", numerical_transformer, numerical_features),
                ("binary_transformer", binary_transformer, binary_features)
            ]
        )

    # Model definition
    grid_search_done = False
    if args.regressor == 'LR':
        model = LinearRegression()
    else: # If a model can have hyperparameters to be tuned, allow a GridSearch with cross-validation
        if args.regressor == 'Ridge':
            regressor = Ridge()
        elif args.regressor == 'RF':
            regressor = RandomForestRegressor()
        elif args.regressor == 'Catboost':
            regressor = CatBoostRegressor()
        elif args.regressor == 'XGboost':
            regressor = XGBRegressor()
        elif args.regressor == 'DecisionTree':
            regressor = DecisionTreeRegressor()

        regressor_args = {option: parameters for option, parameters in vars(args).items() if parameters is not None and option not in ['cv', 'regressor']}
        regressor_params = {param_name: values for param_name, values in regressor_args.items()}
        
        if regressor_params:
            model = GridSearchCV(regressor, param_grid=regressor_params, cv=args.cv, verbose=3)
            grid_search_done = True
        else:
            model = regressor  # If no hyperparameters to tune, use the default regressor

    # Pipeline 
    predictor = Pipeline(steps=[
        ('features_preprocessing', feature_preprocessor),
        ("model", model)
    ])


    # # Log experiment to MLFlow
    with mlflow.start_run() as run:
        
        # Fit the model on train set
        predictor.fit(X_train, y_train)

        # Log GridSearch's best_params_ attribute if a grid search has been performed
        if grid_search_done:
            mlflow.log_params({"best_param_" + k: v for k, v in model.best_params_.items()})

        # Make predictions
        y_train_pred = predictor.predict(X_train)
        y_test_pred = predictor.predict(X_test)

        # Log MAE score expressed as % of the target median as new metric for train set 
        mlflow.log_metric(
            "training_MAE_percent_of_target_median", round(mean_absolute_error(y_train, y_train_pred) / y.median() * 100, 2)
        )


        # Log MAE score expressed as % of the target median as new metric for test set 
        mlflow.log_metric(
            "test_MAE_percent_of_target_median", round(mean_absolute_error(y_test, y_test_pred) / y.median() * 100, 2)
        )

        # Fill 'description' field of the run with model info and main metric
        mlflow.set_tags({'mlflow.note.content':f"{model}\ntest MAE: {round(mean_absolute_error(y_test, y_test_pred) / y.median() * 100, 2)}% of target median"})

        # End mlflow autolog for retraining model on whole dataset (train + test) 
        mlflow.sklearn.autolog(disable=True)
        predictor.fit(X, y)

        # Log model seperately to have more flexibility on setup 
        mlflow.sklearn.log_model(
            sk_model=predictor,
            artifact_path="car_rental_price_predictor",
            registered_model_name=f"{args.regressor}_car_rental_price_predictor",
            signature=infer_signature(X, y)
        )
        
    print("...Done!")
    print(f"---Total training time: {time.time()-start_time}")

