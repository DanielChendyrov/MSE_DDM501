import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os

# Set tracking URI to local mlruns directory
mlflow.set_tracking_uri("file:./mlruns")

# Create experiment
experiment_name = "Classification_Experiment"
try:
    experiment_id = mlflow.create_experiment(experiment_name)
except:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# Set the experiment
mlflow.set_experiment(experiment_name)

def generate_data(n_samples=1000, n_features=20, n_classes=2, random_state=42):
    """Generate synthetic classification data"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_features // 2,
        random_state=random_state
    )
    return X, y

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train logistic regression model with hyperparameter tuning"""
    with mlflow.start_run(run_name="Logistic_Regression"):
        # Log data characteristics
        mlflow.log_params({
            "data_size": X_train.shape[0],
            "n_features": X_train.shape[1]
        })
        
        # Define parameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga'],
            'max_iter': [100, 500, 1000]
        }
        
        # Grid search
        grid_search = GridSearchCV(
            LogisticRegression(random_state=42),
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Train the model
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Log best parameters
        mlflow.log_params(grid_search.best_params_)
        
        # Evaluate model
        metrics = evaluate_model(best_model, X_test, y_test)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        return best_model, metrics

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train random forest model with hyperparameter tuning"""
    with mlflow.start_run(run_name="Random_Forest"):
        # Log data characteristics
        mlflow.log_params({
            "data_size": X_train.shape[0],
            "n_features": X_train.shape[1]
        })
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        
        # Grid search
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Train the model
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Log best parameters
        mlflow.log_params(grid_search.best_params_)
        
        # Evaluate model
        metrics = evaluate_model(best_model, X_test, y_test)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        return best_model, metrics

def train_svm(X_train, y_train, X_test, y_test):
    """Train SVM model with hyperparameter tuning"""
    with mlflow.start_run(run_name="SVM"):
        # Log data characteristics
        mlflow.log_params({
            "data_size": X_train.shape[0],
            "n_features": X_train.shape[1]
        })
        
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        }
        
        # Grid search
        grid_search = GridSearchCV(
            SVC(random_state=42, probability=True),
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Train the model
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Log best parameters
        mlflow.log_params(grid_search.best_params_)
        
        # Evaluate model
        metrics = evaluate_model(best_model, X_test, y_test)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        return best_model, metrics

def find_best_run():
    """Find the best model based on accuracy metric"""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs.empty:
        return None
    
    # Find the run with the highest accuracy
    best_run = runs.loc[runs['metrics.accuracy'].idxmax()]
    return best_run

def register_best_model(best_run):
    """Register the best model in the MLflow model registry"""
    model_uri = f"runs:/{best_run.run_id}/model"
    model_name = "best_classification_model"
    
    # Register the model
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
    
    # Save the model locally in the models directory
    if not os.path.exists('models'):
        os.makedirs('models')
    
    best_model = mlflow.sklearn.load_model(model_uri)
    joblib.dump(best_model, 'models/best_model.joblib')
    
    return registered_model

if __name__ == "__main__":
    # Generate data
    X, y = generate_data(n_samples=1000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train different models
    print("Training Logistic Regression model...")
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_test, y_test)
    
    print("Training Random Forest model...")
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    
    print("Training SVM model...")
    svm_model, svm_metrics = train_svm(X_train, y_train, X_test, y_test)
    
    # Compare models
    print("\nModel Comparison:")
    print(f"Logistic Regression - Accuracy: {lr_metrics['accuracy']:.4f}, F1-Score: {lr_metrics['f1_score']:.4f}")
    print(f"Random Forest - Accuracy: {rf_metrics['accuracy']:.4f}, F1-Score: {rf_metrics['f1_score']:.4f}")
    print(f"SVM - Accuracy: {svm_metrics['accuracy']:.4f}, F1-Score: {svm_metrics['f1_score']:.4f}")
    
    # Find and register the best model
    print("\nFinding best model from MLflow tracking...")
    best_run = find_best_run()
    
    if best_run is not None:
        print(f"Best model: {best_run['tags.mlflow.runName']} with accuracy: {best_run['metrics.accuracy']:.4f}")
        registered_model = register_best_model(best_run)
        print(f"Best model registered as: {registered_model.name} version {registered_model.version}")
    else:
        print("No models found in MLflow tracking.")