# MSE_DDM501 - MLflow Classification Project

This project demonstrates how to create and deploy a machine learning classification model using MLflow for experiment tracking and model registry, with a Flask web application for serving predictions.

## Project Structure

```text
.
├── .github/workflows/   # CI/CD GitHub Actions workflows
├── app/                 # Flask web application
│   ├── templates/       # HTML templates
│   └── app.py           # Flask app code
├── models/              # Directory for saved models
├── mlruns/              # MLflow tracking data (created on first run)
├── Dockerfile           # Docker configuration
├── requirements.txt     # Project dependencies
├── train_model.py       # Script for training and registering models
└── README.md            # This file
```

## Features

- Synthetic data generation using scikit-learn's make_classification
- Training multiple classification models (Logistic Regression, Random Forest, SVM)
- Hyperparameter tuning for each model
- MLflow experiment tracking and model registry
- Flask web application for model inference
- CI/CD with GitHub Actions and Docker

## Getting Started

### Local Development

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Train the models:

   ```bash
   python train_model.py
   ```

3. Run the Flask application:

   ```bash
   python app/app.py
   ```

4. Access the web application at <http://localhost:5000>

### Using Docker

1. Build the Docker image:

   ```bash
   docker build -t mse_ddm501 .
   ```

2. Run the Docker container:

   ```bash
   docker run -p 5000:5000 mse_ddm501
   ```

3. Access the web application at <http://localhost:5000>

### CI/CD Setup

This project is configured with GitHub Actions for Continuous Integration and Continuous Deployment. When you push to the main branch:

1. The code is automatically tested
2. A Docker image is built
3. The image is pushed to Docker Hub

To set up CI/CD:

1. Create a GitHub repository named `MSE_DDM501`
2. Create a Docker Hub account if you don't have one
3. Generate a Docker Hub access token
4. Add these secrets to your GitHub repository:
   - DOCKER_USERNAME: Your Docker Hub username
   - DOCKER_TOKEN: Your Docker Hub access token

## Using the Web Application

The web application provides two main features:

1. **Making Predictions**: You can either use randomly generated demo data or input your own feature values to get a prediction.

2. **Model Information**: View details about the best performing model, including its type and parameters.

## MLflow Tracking

You can view detailed MLflow experiment tracking results by running:

```bash
mlflow ui
```

This will start the MLflow UI server, which you can access at <http://localhost:5000> to see detailed metrics, parameters, and artifacts for all your experiments.
