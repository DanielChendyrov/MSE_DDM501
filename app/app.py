from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os
import sys
import mlflow
from sklearn.datasets import make_classification

# Add parent directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

app = Flask(__name__)

# Load the best model
def load_model():
    try:
        # Try to load the model from the local file first
        model_path = os.path.join(parent_dir, 'models', 'best_model.joblib')
        if os.path.exists(model_path):
            print(f"Loading model from local file: {model_path}")
            return joblib.load(model_path)
        else:
            # If local file doesn't exist, try MLflow registry
            print("Local model file not found, attempting to load from MLflow registry")
            mlflow_dir = os.path.join(parent_dir, 'mlruns')
            mlflow.set_tracking_uri(f"file:{mlflow_dir}")
            
            # List available models to debug
            client = mlflow.tracking.MlflowClient()
            registered_models = client.search_registered_models()
            print(f"Available registered models: {[rm.name for rm in registered_models]}")
            
            # Try to get the run ID of the best model
            experiment = client.get_experiment_by_name("Classification_Experiment")
            if experiment:
                runs = mlflow.search_runs([experiment.experiment_id])
                if not runs.empty:
                    best_run_id = runs.loc[runs['metrics.accuracy'].idxmax()]['run_id']
                    print(f"Best run ID: {best_run_id}")
                    model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
                    return model
            
            raise Exception("No model found in MLflow registry")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if we're using demo data
        if request.form.get('use_demo') == 'true':
            # Generate demo data
            X, _ = make_classification(
                n_samples=1,
                n_features=20,
                n_classes=2,
                n_informative=10,
                random_state=np.random.randint(0, 100)
            )
            features = X[0]
        else:
            # Get manual input features
            features = []
            for i in range(20):  # Assuming 20 features as in our training data
                feature_value = request.form.get(f'feature_{i}')
                features.append(float(feature_value) if feature_value else 0.0)
            features = np.array(features)
        
        if model is not None:
            # Make prediction
            prediction = model.predict([features])[0]
            probability = model.predict_proba([features])[0].tolist()
            
            return jsonify({
                'status': 'success',
                'prediction': int(prediction),
                'probability': probability,
                'features': features.tolist()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded. Please train the model first.'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/model-info')
def model_info():
    if model is not None:
        # Get model type and parameters if possible
        try:
            model_type = type(model).__name__
            params = model.get_params()
        except:
            model_type = "Unknown"
            params = {}
        
        return render_template('model_info.html', model_type=model_type, params=params)
    else:
        return render_template('model_info.html', error="Model not loaded")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)