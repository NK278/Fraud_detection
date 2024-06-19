# Fraud_detection

## Project Overview
The Fraud_detection project aims to detect fraudulent transactions using various machine learning techniques. It includes data processing, model training, evaluation, and deployment components to build a robust fraud detection system.

## File Structure

```
Fraud_detection/
├── notebooks/
│   ├── Untitled1.ipynb
│   ├── .DS_Store
│   ├── Untitled.ipynb
│   ├── Random_forest_clf.pkl
│   ├── test.csv
│   ├── correlation_heatmap.png
│   ├── Untitled1 copy.ipynb
│   ├── Untitled1 copy 2.ipynb
│   ├── .ipynb_checkpoints/
│   │   ├── Untitled1-checkpoint.ipynb
│   │   ├── creditCardFraud_28011964_120214-checkpoint.csv
│   │   ├── Untitled-checkpoint.ipynb
│   ├── Untitled copy.ipynb
│   ├── fraudTrain.csv
│   ├── fraudTest.csv
│   ├── ts.ipynb
│   ├── creditCardFraud_28011964_120214.csv
│   ├── Random_forest_clf_d2.pkl
│   └── transformed_pip.pkl
├── src/
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   ├── __init__.py
│   │   ├── __pycache__/
│   │   │   ├── prediction_pipeline.cpython-311.pyc
│   │   │   ├── train_pipeline.cpython-311.pyc
│   │   │   ├── __init__.cpython-311.pyc
│   │   ├── prediction_pipeline.py
│   ├── configuration/
│   │   ├── __init__.py
│   │   ├── mongodb_connection.py
│   ├── exception.py
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── __pycache__/
│   │   │   ├── main_utils.cpython-311.pyc
│   │   │   ├── __init__.cpython-311.pyc
│   │   ├── main_utils.py
│   ├── __pycache__/
│   │   ├── exception.cpython-311.pyc
│   │   ├── logger.cpython-311.pyc
│   │   ├── __init__.cpython-311.pyc
│   ├── logger.py
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── model_trainer.py
│   │   ├── __init__.py
│   │   ├── __pycache__/
│   │   │   ├── data_ingestion.cpython-311.pyc
│   │   │   ├── model_trainer.cpython-311.pyc
│   │   │   ├── data_transformation.cpython-311.pyc
│   │   │   ├── __init__.cpython-311.pyc
│   │   ├── data_transformation.py
│   ├── constant/
│   │   ├── __init__.py
│   │   ├── __pycache__/
│   │   │   ├── __init__.cpython-311.pyc
├── templates/
│   └── upload_file.html
├── upload_data_to_db/
│   └── upload.ipynb
├── app.py
├── requirements.txt
└── setup.py
```

### Description of Files and Directories

- **notebooks/**
  - Contains various Jupyter notebooks for data analysis, model training, and testing.
  - `Random_forest_clf.pkl`: Serialized random forest classifier model.
  - `test.csv`: Test dataset for evaluation.
  - `correlation_heatmap.png`: Heatmap image showing correlation among features.
  - `fraudTrain.csv`: Training dataset for the model.
  - `fraudTest.csv`: Testing dataset for model evaluation.
  - `transformed_pip.pkl`: Serialized transformation pipeline.

- **src/**
  - **pipeline/**: Scripts for training and prediction pipelines.
    - `train_pipeline.py`: Script for training the model.
    - `prediction_pipeline.py`: Script for making predictions using the trained model.
  - **configuration/**: Configuration and connection settings.
    - `mongodb_connection.py`: Script to handle MongoDB connections.
  - **components/**: Contains core components for data ingestion, transformation, and model training.
    - `data_ingestion.py`: Script for ingesting data.
    - `data_transformation.py`: Script for data transformation.
    - `model_trainer.py`: Script for training the model.
  - **utils/**: Utility functions used across the project.
    - `main_utils.py`: Main utility functions.
  - **constant/**: Constant definitions used throughout the project.
  - `exception.py`: Custom exception handling.
  - `logger.py`: Logging configuration and functions.

- **templates/**
  - `upload_file.html`: HTML template for file upload interface in the web application.

- **upload_data_to_db/**
  - `upload.ipynb`: Notebook for uploading data to the database.

- **app.py**: Flask web application script for deploying the model and serving the interface.

- **requirements.txt**: List of Python dependencies required for the project.

- **setup.py**: Setup script for installing the project as a package.

## Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/NK278/Fraud_detection.git
    cd Fraud_detection
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the web application**:
    ```bash
    python app.py
    ```

5. **Access the web application**:
    Open a web browser and go to `http://127.0.0.1:5000`.

## Usage

- **Web Application**: The web application allows users to upload transaction data and get predictions on whether the transactions are fraudulent or not.

- **Jupyter Notebooks**: The notebooks in the `notebooks/` directory provide detailed steps for data analysis, model training, and evaluation.

## Dependencies

The project requires the following Python packages:

- Flask
- pandas
- scikit-learn
- Jupyter
- pymongo

All required packages are listed in `requirements.txt` and can be installed using `pip`.

## Contact

For any questions or issues, please contact [nishchalgaur2003@gmail.com](mailto:nishchalgaur2003@gmail.com).


---

Feel free to contribute to this project by submitting issues or pull requests. We hope this project helps in detecting and preventing fraudulent transactions effectively.
