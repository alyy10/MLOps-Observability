# MLOps-Observability:Classification with Drift Monitoring

This project presents an end-to-end MLOps pipeline focused on monitoring and addressing model and data degradation in a loan eligibility classification system. The core objective is to predict whether a loan should be granted or rejected, based on data sourced from a PostgreSQL database. A key highlight of this project is its robust implementation of Model and Drift Monitoring, ensuring the continuous reliability and performance of the machine learning model in a dynamic data environment. Furthermore, the entire ML pipeline is orchestrated using Docker and Apache Airflow, demonstrating best practices in MLOps for automation, scalability, and reproducibility.

In essence, this repository provides a comprehensive solution for deploying, monitoring, and maintaining a machine learning model in a production-like setting, specifically addressing the critical challenges of concept drift, data drift, and model drift to sustain predictive accuracy over time.

<img width="1208" height="287" alt="monitored_pipeline" src="https://github.com/user-attachments/assets/ae2e2cbf-03e3-42f7-b48a-b9a3cdf87fbf" />


## Aim

This project aims to achieve the following:

* **Fetch data from PostgreSQL server**: Establish robust data extraction mechanisms to retrieve loan-related data from a PostgreSQL database.
* **Build a Loan Eligibility classification model**: Develop a machine learning model capable of predicting whether a loan should be given or refused.
* **Monitor Concept, Data, and Model drift**: Implement comprehensive monitoring solutions to detect changes in data distribution, relationships between features and target, and model performance degradation over time.
* **Orchestrate monitoring pipeline with Airflow**: Automate the entire ML pipeline, including data ingestion, model training, drift detection, and model deployment, using Apache Airflow for efficient and reliable operations.

## Tech Stack

This project leverages a variety of technologies and libraries to build a robust and observable MLOps pipeline:

* **Language**: Python
* **Libraries**:
  * `pandas`: For data manipulation and analysis.
  * `numpy`: For numerical operations.
  * `matplotlib`: For data visualization.
  * `scikit-learn`: For machine learning model development (Random Forest, Gradient Boosting).
  * `deepchecks`: For comprehensive data and model drift detection, and data quality checks.
  * `sqlalchemy`: For interacting with SQL databases.
  * `psycopg2-binary`: PostgreSQL adapter for Python.
* **Services**:
  * `Airflow`: For orchestrating and automating the ML pipeline.
  * `Docker`: For containerizing applications and ensuring reproducible environments.
  * `PostgreSQL`: As the primary data source for loan-related information.

## Approach

The project follows a structured approach to build and monitor the loan eligibility classification model:

1. **Extracting Data from PostgreSQL**: Data is fetched from a PostgreSQL database, which serves as the primary data source for loan details, customer information, and credit history. This process is managed by the `etl.py` script and demonstrated in `data collection.ipynb`.
2. **Data Preprocessing**: Raw data undergoes several preprocessing steps to ensure quality and suitability for model training. This includes:
   * **Data Cleaning**: Handling missing values and ensuring correct data types.
   * **Feature Engineering**: Creating new, informative features from existing ones (e.g., extracting temporal features from application time, calculating credit balance ratios).
   * **Encoding Categorical Variables**: Converting categorical features into numerical representations using various encoding strategies (e.g., ranking, weighted ranking) as implemented in `preprocess.py`.
3. **Train-Test Split**: The preprocessed data is split into training and testing datasets, typically using a time-based split to simulate real-world data flow and ensure that the model is evaluated on unseen, future data.
4. **Model Training and Evaluation**: Two classification models, Random Forest and Gradient Boosting, are trained on the prepared dataset. Their performance is rigorously evaluated using metrics such as AUC, accuracy, precision, recall, and F1-score. The `train.py` module handles this process, including model selection based on predefined performance and degradation thresholds.
5. **Model Monitoring**: A crucial aspect of this project is the continuous monitoring of the deployed model and the incoming data for any signs of degradation. This involves:
   * **Concept Drift**: Monitoring changes in the relationship between input features and the target variable.
   * **Data Drift**: Detecting shifts in the distribution of input data over time.
   * **Model Drift**: Assessing the degradation of the model's predictive performance. The `drifts.py` module, leveraging the `deepchecks` library, is central to these monitoring efforts, generating detailed reports and triggering alerts or retraining processes when drifts are detected.
6. **Orchestration with Airflow**: The entire MLOps pipeline, from data extraction and preprocessing to model training, drift detection, and potential retraining/redeployment, is automated and managed using Apache Airflow. The `dag_pipeline.py` defines the DAG, ensuring a robust and reproducible workflow, while `docker-compose.yaml` sets up the necessary services for the Airflow environment and the inference API.

## Project Structure

This repository is organized to facilitate an end-to-end MLOps pipeline. Below is a detailed breakdown of the key files and directories:

* **`EDA.ipynb`**: This Jupyter Notebook is used for Exploratory Data Analysis. It contains initial data loading, inspection of data types, descriptive statistics, and visualizations to understand the dataset's characteristics and identify potential issues before preprocessing. It also includes preliminary data cleaning and feature engineering steps.
* **`airflow.sh`**: A shell script designed to run Airflow commands within the Dockerized environment. It simplifies the execution of Airflow CLI commands by wrapping `docker-compose run --rm airflow-cli`.
* **`app.py`**: This Flask application serves as the inference endpoint for the deployed machine learning model. It exposes a `/predict` endpoint that receives input data, preprocesses it using the deployed model's preprocessing steps, and returns predictions. It also includes a `/ping` endpoint for health checks.
* **`config.py`**: This file centralizes all configuration parameters for the project. It defines paths for data, models, and results, as well as various thresholds for model performance and drift detection. It also lists categorical and numerical variables, predictors, and stages of the ML pipeline.
* **`dag_pipeline.py`**: This is the main Apache Airflow DAG definition file. It orchestrates the entire MLOps pipeline, including data extraction, data quality checks, data drift detection, preprocessing, model training, model drift detection, and model deployment. It uses PythonOperators and BranchPythonOperators to define the workflow and conditional task execution.
* **`dag_test.py`**: A simple Airflow DAG used for testing Slack integration. It demonstrates how to send test messages to a Slack channel using the `SlackAPIPostOperator`.
* **`data collection.ipynb`**: This Jupyter Notebook focuses on the data collection process from PostgreSQL. It contains SQL queries for creating temporary tables and extracting the necessary data, demonstrating the `etl.py` functionalities in an interactive manner.
* **`docker-compose.yaml`**: This Docker Compose file defines the multi-container Docker environment for the project. It sets up Airflow services (webserver, scheduler, worker), a PostgreSQL database, Redis as a message broker, and the Flask inference application (`predictor`). It ensures that all services are properly linked and configured for local development and testing.
* **`drifts.py`**: This Python module contains functions for detecting data quality issues, data drift, and model drift using the `deepchecks` library. It includes `check_data_quality`, `check_data_drift`, and `check_model_drift` functions, which generate HTML reports and determine whether retraining is necessary.
* **`etl.py`**: This module handles the Extract, Transform, Load (ETL) process. It contains functions to connect to the PostgreSQL database, execute SQL queries (defined in `queries.py`), extract raw data, and save it to the designated data directory. It is responsible for fetching the initial dataset for the pipeline.
* **`helpers.py`**: A collection of utility functions used across the project. This includes functions for saving and loading datasets and models (pickle and JSON), interacting with the PostgreSQL database (creating tables, logging activity, retrieving job statuses), generating UUIDs, and managing file paths.
* **`inference.py`**: This module contains functions for making predictions using the deployed model. It loads the latest deployed model and applies it to new data. It also includes a `batch_inference` function for processing data in batches and a `make_predictions` function for real-time inference via the Flask app.
* **`missing_values_model.pkl`**: A pickled Python object (likely a scikit-learn imputer or a custom object) used for handling missing values during preprocessing. This model is trained on the reference data and then used consistently for both training and inference to ensure consistent imputation.
* **`ml-monitoring.sql`**: This SQL script defines the schema for the monitoring database. It includes DDL (Data Definition Language) for tables like `ml_model`, `model_metric`, `alert_rule`, and `alert`, along with views and triggers to support real-time monitoring and alerting for machine learning models.
* **`modelDrifts.ipynb`**: This Jupyter Notebook provides an interactive environment for exploring and demonstrating model drift detection. It uses the `deepchecks` library to compare model predictions on different datasets and visualize any observed drifts, complementing the automated checks in `drifts.py`.
* **`preprocess.py`**: This module contains functions for data preprocessing, including enforcing data types, engineering new features, encoding categorical variables, and imputing missing values. It ensures that raw data is transformed into a format suitable for model training and inference.
* **`preprocessing.ipynb`**: A Jupyter Notebook that demonstrates the data preprocessing steps defined in `preprocess.py`. It allows for interactive exploration and validation of each preprocessing stage.
* **`purpose_to_int_model.json`**: A JSON file that likely stores the mapping or encoding scheme for the 'purpose' categorical variable. This is used to ensure consistent encoding of this feature across different stages of the pipeline.
* **`queries.py`**: This file stores all SQL queries used by the `etl.py` module to interact with the PostgreSQL database. It includes queries for creating temporary tables and selecting data based on specified criteria.
* **`rf.pkl`**: A pickled Python object representing a trained Random Forest Classifier model. This is one of the potential models that can be deployed for inference.

## Setup/Installation Guide

To set up and run this project locally, follow these steps:

### Prerequisites

* **Docker and Docker Compose**: Ensure Docker and Docker Compose are installed on your system. These are essential for running the containerized Airflow and PostgreSQL services.
  * [Install Docker](https://docs.docker.com/get-docker/)
  * [Install Docker Compose](https://docs.docker.com/compose/install/)

### 1. Clone the Repository

First, clone this GitHub repository to your local machine:

```bash
git clone https://github.com/alyy10/MLOps-Observability.git
cd MLOps-Observability
```

### 2. Environment Setup

The project uses a `docker-compose.yaml` file to set up all necessary services. This includes PostgreSQL, Redis, and various Airflow components.

#### Build and Start Docker Containers

Navigate to the root directory of the cloned repository and run the following command to build the Docker images and start the services:

```bash
docker-compose up --build -d
```

* `--build`: This flag ensures that Docker images are built before starting the containers. This is important for the first time setup or if you make changes to the Dockerfile.
* `-d`: This flag runs the containers in detached mode, meaning they will run in the background.

This command will:

* Create and start the `postgres` container (for the Airflow metadata database and project data).
* Create and start the `redis` container (as a message broker for Airflow CeleryExecutor).
* Create and start `airflow-webserver`, `airflow-scheduler`, `airflow-worker`, and `airflow-triggerer` containers.
* Create and start the `predictor` container, which runs the Flask inference application.

### 3. Initialize Airflow

After the containers are up and running, you need to initialize the Airflow environment. This involves creating the Airflow database and setting up the necessary users.

#### Run Airflow Database Migrations

```bash
docker-compose run airflow-cli airflow db migrate
```

#### Create an Airflow User

Create an admin user for the Airflow UI. Replace `admin`, `admin`, `admin@example.com`, `admin`, and `admin` with your desired username, first name, last name, email, and password respectively.

```bash
docker-compose run airflow-cli airflow users create \
    --username admin \
    --password admin \
    --firstname admin \
    --lastname admin \
    --role Admin \
    --email admin@example.com
```

### 4. Configure Airflow Connections

This project requires a PostgreSQL connection for data extraction and a Slack connection for notifications. You can configure these via the Airflow UI.

#### Access Airflow UI

Open your web browser and navigate to `http://localhost:8080`. Log in with the admin credentials you created in the previous step.

#### Create PostgreSQL Connection

1. In the Airflow UI, go to `Admin` -> `Connections`.
2. Click on the `+` button to add a new connection.
3. Fill in the details:
   * **Conn Id**: `postgres_default` (or any other ID, but ensure it matches what's used in `config.py` if you change it)
   * **Conn Type**: `PostgreSQL`
   * **Host**: `postgres` (this is the service name defined in `docker-compose.yaml`)
   * **Schema**: `airflow`
   * **Login**: `airflow`
   * **Password**: `airflow`
   * **Port**: `5432`
4. Click `Save`.

#### Create Slack Connection

1. In the Airflow UI, go to `Admin` -> `Connections`.
2. Click on the `+` button to add a new connection.
3. Fill in the details:
   * **Conn Id**: `slack_connection` (this ID is used in `dag_pipeline.py`)
   * **Conn Type**: `Slack`
   * **Token**: Your Slack Bot User OAuth Token (starts with `xoxb-`). You will need to create a Slack app and add a bot to your workspace to get this token. Ensure the bot has the necessary permissions (e.g., `chat:write`, `files:write`).
   * **Login**: Your Slack channel name (e.g., `#mlops-alerts`). This will be the default channel for notifications.
4. Click `Save`.

### 5. Upload Creds.json

The `etl.py` and `helpers.py` scripts use a `Creds.json` file to connect to the PostgreSQL database. You need to create this file and place it in the `dags` directory.

Create a file named `Creds.json` inside the `dags` directory with the following content:

```json
{
    "host": "postgres",
    "database": "airflow",
    "user": "airflow",
    "password": "airflow",
    "port": "5432"
}
```

### 6. Prepare Data Directories

Ensure the necessary data directories exist within the `dags` folder. These are mounted as volumes in `docker-compose.yaml`.

```bash
mkdir -p dags/data/raw dags/data/preprocessed dags/models dags/results
```

### 7. Unpause the DAG

In the Airflow UI, navigate to the `DAGs` page. Find the `ml_pipeline_monitoring` DAG and toggle the switch to unpause it. The DAG should start running according to its schedule (`@daily`).

Your MLOps Observability pipeline is now set up and ready to run!

## Usage Instructions

Once the Airflow environment is set up and the `ml_pipeline_monitoring` DAG is unpaused, the pipeline will automatically run daily to perform data extraction, preprocessing, model training, and drift checks. However, you can also manually trigger the DAG or interact with the inference service.

### 1. Running the Airflow DAG Manually

To manually trigger the `ml_pipeline_monitoring` DAG:

1. Open the Airflow UI (`http://localhost:8080`).
2. Navigate to the `DAGs` page.
3. Find the `ml_pipeline_monitoring` DAG.
4. Click on the `Trigger DAG` button (the play icon) in the `Actions` column.
5. You can optionally provide a JSON configuration for the DAG run, for example, to specify a custom `start_date` and `end_date` for data extraction:

   ```json
   {
       "start_date": "2023-01-01",
       "end_date": "2023-03-31"
   }
   ```

   If no configuration is provided, the DAG will use its default date range (typically the last day or a predefined period).
6. Monitor the progress of the DAG run in the `Graph View` or `Gantt Chart` sections of the Airflow UI.

### 2. Interacting with the Inference Service

The Flask application (`app.py`) runs as a Docker service named `predictor` and is exposed on port `5100`. You can send prediction requests to this service.

#### Ping Endpoint

To check if the inference service is running, you can send a GET request to the `/ping` endpoint:

```bash
curl http://localhost:5100/ping
```

Expected output:

```json
{
  "datetime": "YYYY-MM-DD HH:MM:SS.ssssss",
  "status": "ok"
}
```

#### Prediction Endpoint

To get predictions from the deployed model, send a POST request to the `/predict` endpoint with your input data in JSON format. The input data should be a list of dictionaries, where each dictionary represents a loan application and contains the features required by the model.

**Example Request:**

```bash
curl -X POST \\
  http://localhost:5100/predict \\
  -H 'Content-Type: application/json' \\
  -d '[{
    "loan_id": "loan123",
    "current_loan_amount": 15000,
    "term": "short term",
    "credit_score": 720,
    "years_in_current_job": "5 years",
    "home_ownership": "rent",
    "annual_income": 60000,
    "purpose": "debt consolidation",
    "monthly_debt": 800,
    "years_of_credit_history": 10,
    "months_since_last_delinquent": 20,
    "no_of_open_accounts": 15,
    "no_of_credit_problems": 0,
    "current_credit_balance": 10000,
    "max_open_credit": 25000,
    "bankruptcies": 0,
    "tax_liens": 0,
    "no_of_properties": 1,
    "no_of_cars": 2,
    "no_of_children": 1
  }]'
```

**Example Response:**

```json
[
  {
    "loan_id": "loan123",
    "prediction": "loan given"
  }
]
```

**Note**: The input features must match the `PREDICTORS` defined in `config.py` and should be in a format that can be converted into a pandas DataFrame. The `loan_id` column is mandatory in the input data.

## Monitoring

This project places a strong emphasis on continuous monitoring of the machine learning pipeline and the deployed model to ensure its reliability and performance over time. The `deepchecks` library is integrated to provide comprehensive checks for data quality, data drift, and model drift.

### Data Quality Monitoring

Data quality checks are performed at the beginning of the pipeline to ensure that the incoming data is clean and suitable for processing. The `drifts.py` module includes the `check_data_quality` function, which identifies issues such as duplicate samples, conflicting labels, and outliers. A detailed HTML report is generated for each run, providing insights into any detected anomalies. If critical data quality issues are found, the Airflow DAG can be configured to halt further processing and alert the team via Slack.

### Data Drift Monitoring

Data drift refers to changes in the distribution of input data over time, which can significantly impact model performance. The `check_data_drift` function in `drifts.py` compares the characteristics of the current data with a reference dataset (typically the data used to train the last deployed model). It monitors various aspects, including:

* **New Labels**: Detection of new categorical values not seen during training.
* **Overall Dataset Drift**: A comprehensive score indicating the magnitude of change across the entire dataset.
* **Feature Drift**: Changes in the distribution of individual features.
* **Feature-Label Correlation Change**: Shifts in the relationship between features and the target variable.

An HTML report (`[job_id]_data_drift_report.html`) is generated, visualizing the detected drifts. If significant data drift is identified, it signals a potential need for model retraining to adapt to the new data patterns.

### Model Drift Monitoring

Model drift occurs when the performance of a deployed model degrades over time due to changes in the underlying data patterns or relationships. The `check_model_drift` function in `drifts.py` assesses the model's performance on new, unseen data and compares it against its performance on the reference data. Key metrics monitored include:

* **AUC Score**: Tracks the Area Under the Receiver Operating Characteristic curve, a common metric for classification model performance.
* **Prediction Drift**: Measures changes in the model's predictions over time.

Similar to data drift, an HTML report (`[job_id]_model_drift_report.html`) is generated, providing a visual summary of the model's performance and any detected degradation. If model drift is significant, the Airflow pipeline can automatically trigger a retraining process, ensuring that the model remains accurate and effective.

### Alerting and Reporting

All monitoring results are captured and can be used to trigger alerts. The Airflow DAG is integrated with Slack, allowing for real-time notifications when data quality issues, data drift, or model drift are detected. This proactive alerting mechanism ensures that the MLOps team is immediately aware of potential problems and can take timely action to maintain the integrity and performance of the ML system. Additionally, the generated HTML reports provide detailed insights for further investigation and analysis.
