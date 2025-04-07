# Real-Time-Taxi-Trip-Duration-Prediction-Pipeline-on-GCP
This project implements an end-to-end, real-time machine learning pipeline on Google Cloud Platform (GCP) for predicting taxi trip durations. The solution spans data ingestion, preprocessing, model training, deployment, and real-time inference. It leverages several GCP services including Pub/Sub, Dataflow (Apache Beam), BigQuery, and Vertex AI—all orchestrated to work together in a robust MLOps workflow.

> **Note:** All project IDs, regions, bucket names, etc., are masked in the examples below.

---

## Architecture Overview

1. **Data Ingestion & Preprocessing**  
   - **Pub/Sub:** Serves as the messaging backbone, receiving streaming taxi trip data (each row as a message).  
   - **Dataflow (Apache Beam):** A managed service that executes Apache Beam pipelines to process (clean, validate, and transform) the incoming data.  
     - *DoFn:* A core concept in Beam that represents a function applied to every element in a distributed collection (called a PCollection). Each DoFn runs in parallel on different workers managed by Dataflow.
   - **BigQuery:** Acts as the central data warehouse to store cleaned and enriched taxi trip records for analysis and model training.

2. **Model Training & Deployment**  
   - **Vertex AI Workbench (Notebook):** Used for Exploratory Data Analysis (EDA), feature engineering, model training, and evaluation. Multiple models (e.g., XGBoost, Random Forest, Linear Regression) are trained and the best performing model is selected.
   - **Vertex AI Model & Endpoint:** The best model artifact is uploaded and deployed to a managed endpoint for real-time predictions.

3. **(Optional) MLOps Automation**  
   - **Cloud Scheduler & Cloud Functions / Vertex AI Pipelines:** Automate periodic retraining and redeployment, ensuring that the model evolves with new data.

---

### 1. Data Publishing to Pub/Sub

**Purpose:**  
Stream raw taxi trip data from a local CSV file to a Pub/Sub topic, simulating real-time data ingestion.

**Code: `publish_to_pubsub.py`**

```python
import time
import pandas as pd
import json
from google.cloud import pubsub_v1

# Replace with your actual project ID and topic name.
project_id = "my-project-id"
topic_id = "my-taxi-trips-topic"

# Create a PublisherClient to send messages.
publisher = pubsub_v1.PublisherClient()

# Construct the topic path.
topic_path = publisher.topic_path(project_id, topic_id)

# Load local taxi trip data from CSV.
df = pd.read_csv("nyc_taxi_sample.csv")

# Iterate over each row and publish as JSON.
for i, row in df.iterrows():
    message = row.to_dict()                       # Convert row to dictionary.
    data = json.dumps(message).encode("utf-8")      # Encode as JSON bytes.
    publisher.publish(topic_path, data=data)        # Publish to Pub/Sub.
    print(f"Published row {i+1}")
    time.sleep(1)  # Simulate streaming at 1 row per second.
```

### Usage Instructions:

Update the project_id and topic_id with your actual values.

Ensure nyc_taxi_sample.csv is present in the working directory.

###  Run the script using:
python publish_to_pubsub.py


### 2. Dataflow Pipeline for Data Cleaning & Enrichment
**Purpose:**

Consume streaming data from Pub/Sub, clean and transform it (performing feature engineering such as timestamp parsing and extracting pickup hour/day), and write the processed records to BigQuery. Optionally, the pipeline can be extended to call a Vertex AI endpoint for real-time predictions.

**Key Concepts:**

Apache Beam: A unified programming model for both batch and streaming data processing. In Beam, data is represented as a PCollection and transformed using a series of steps (or PTransforms).

DoFn: A “Do Function” is a user-defined function that processes each element of a PCollection. It allows you to perform operations such as filtering, mapping, and feature engineering on each record. Beam runners (like Dataflow) execute these functions in parallel on distributed worker nodes.

**Code: `dataflow_beam.py`**
```python
import json
import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions

# Set global logging level.
logging.getLogger().setLevel(logging.INFO)

# Define a DoFn to clean and transform each row.
class CleanTransformRow(beam.DoFn):
    def process(self, element):
        from datetime import datetime
        import pandas as pd

        try:
            # Decode the Pub/Sub message from bytes to a dictionary.
            row = json.loads(element.decode("utf-8"))
            logging.info("Got row: %s", row)

            # List of required fields to validate.
            required_fields = [
                "pickup_datetime", "dropoff_datetime",
                "trip_duration", "trip_distance",
                "pickup_latitude", "pickup_longitude",
                "dropoff_latitude", "dropoff_longitude",
                "passenger_count"
            ]

            # Validate required fields.
            for field in required_fields:
                if field not in row or row[field] in [None, "", "null"]:
                    logging.warning("Missing/null field '%s' in row: %s", field, row)
                    return

            # Parse pickup and dropoff timestamps.
            try:
                pickup = datetime.strptime(row["pickup_datetime"], "%Y-%m-%d %H:%M:%S %Z")
                dropoff = datetime.strptime(row["dropoff_datetime"], "%Y-%m-%d %H:%M:%S %Z")
            except Exception as e:
                logging.warning("Invalid datetime format: %s", e)
                return

            # Validate numerical fields.
            duration = int(row["trip_duration"])
            if duration < 30 or duration > 7200:
                logging.warning("Invalid trip_duration: %s", duration)
                return

            distance = float(row["trip_distance"])
            if distance <= 0 or distance > 100:
                logging.warning("Invalid trip_distance: %s", distance)
                return

            passengers = int(row["passenger_count"])
            if passengers <= 0 or passengers > 6:
                logging.warning("Invalid passenger_count: %s", passengers)
                return

            for coord in ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]:
                try:
                    if pd.isnull(row[coord]) or float(row[coord]) == 0.0:
                        logging.warning("Invalid coord '%s': %s", coord, row[coord])
                        return
                except Exception as e:
                    logging.warning(" Coord check error on '%s': %s", coord, e)
                    return

            # Feature engineering: extract additional features.
            hour_of_day = pickup.hour
            day_of_week = pickup.strftime("%A")

            # Construct the transformed row.
            transformed_row = {
                "pickup_datetime": pickup.isoformat(),
                "dropoff_datetime": dropoff.isoformat(),
                "trip_duration": duration,
                "passenger_count": passengers,
                "trip_distance": distance,
                "pickup_latitude": float(row["pickup_latitude"]),
                "pickup_longitude": float(row["pickup_longitude"]),
                "dropoff_latitude": float(row["dropoff_latitude"]),
                "dropoff_longitude": float(row["dropoff_longitude"]),
                "store_and_fwd_flag": row.get("store_and_fwd_flag", "N"),
                "total_amount": float(row.get("total_amount", 0.0)),
                "hour_of_day": hour_of_day,
                "day_of_week": day_of_week
            }

            logging.info("Yielding cleaned row: %s", transformed_row)
            yield transformed_row

        except Exception as e:
            logging.error("JSON decode or transformation failed: %s", e)
            return

def run():
    pipeline_options = PipelineOptions(
        streaming=True,
        project='my-project-id',               # Masked project ID.
        region='my-region',                    # Masked region (e.g., 'us-central1').
        temp_location='gs://my-bucket/temp',    # Masked GCS bucket.
        job_name='taxi-stream-cleaned-pipeline'
    )
    pipeline_options.view_as(StandardOptions).streaming = True

    # Define BigQuery schema for processed taxi data.
    schema = {
        "fields": [
            {"name": "pickup_datetime", "type": "TIMESTAMP"},
            {"name": "dropoff_datetime", "type": "TIMESTAMP"},
            {"name": "trip_duration", "type": "INTEGER"},
            {"name": "passenger_count", "type": "INTEGER"},
            {"name": "trip_distance", "type": "FLOAT"},
            {"name": "pickup_latitude", "type": "FLOAT"},
            {"name": "pickup_longitude", "type": "FLOAT"},
            {"name": "dropoff_latitude", "type": "FLOAT"},
            {"name": "dropoff_longitude", "type": "FLOAT"},
            {"name": "store_and_fwd_flag", "type": "STRING"},
            {"name": "total_amount", "type": "FLOAT"},
            {"name": "hour_of_day", "type": "INTEGER"},
            {"name": "day_of_week", "type": "STRING"}
        ]
    }

    with beam.Pipeline(options=pipeline_options) as pipeline:
        (
            pipeline
            | "ReadFromPubSub" >> beam.io.ReadFromPubSub(
                subscription="projects/my-project-id/subscriptions/my-taxi-trips-sub-demo"  # Masked.
            )
            | "CleanAndTransform" >> beam.ParDo(CleanTransformRow())
            # Optional: Add another DoFn to call Vertex AI for real-time predictions.
            | "WriteToBigQuery" >> beam.io.WriteToBigQuery(
                table="my-project-id:taxi_data.processed_trips",  # Masked.
                schema=schema,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED
            )
        )

if __name__ == "__main__":
    run()

```
###  Usage Instructions:

Update all masked values (project ID, region, GCS bucket paths, subscription name, BigQuery table).

# Run the pipeline with:

```bash
python dataflow_beam.py \
  --runner DataflowRunner \
  --project=my-project-id \
  --region=my-region \
  --temp_location=gs://my-bucket/temp \
  --streaming
```

Dataflow will continuously process incoming messages, perform cleaning and feature engineering, and write the transformed data to BigQuery.

### 3. Model Training and Saving in Vertex AI Workbench
**Notebook: nyc_taxi_eda.ipynb**
**Purpose:**

Load historical taxi trip data from BigQuery.

Perform exploratory data analysis (EDA) and feature engineering.

Train multiple models and evaluate their performance.

Save the best model artifact (e.g., using XGBoost’s native Booster API).

Key Code Snippet:

### 4. Model Deployment on Vertex AI
**File/Script: `deploy_model.ipynb`**
**Purpose:**
Uploads the saved model artifact to Vertex AI, creates an endpoint, and deploys the model for real-time predictions.

```python
# !pip install google-cloud-bigquery db-dtypes pandas scikit-learn xgboost --upgrade

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import xgboost as xgb

# Load data from BigQuery.
client = bigquery.Client(project="my-project-id")  # Masked.
query = """
SELECT *
FROM `my-project-id.taxi_data.training_sample_raw`
"""
df = client.query(query).to_dataframe()
print("Data loaded:", df.shape)

# Feature engineering: Extract hour and day-of-week from pickup_datetime.
df["pickup_hour"] = pd.to_datetime(df["pickup_datetime"]).dt.hour
df["day_of_week"] = pd.to_datetime(df["pickup_datetime"]).dt.day_name()

# Prepare features (X) and target (y).
X = df.drop("trip_duration", axis=1)
y = df["trip_duration"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple models.
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    results[name] = (rmse, r2)
    print(f"\n{name} Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

# Select the best model based on RMSE.
best_model_name = min(results, key=lambda x: results[x][0])
best_model = models[best_model_name]
print("Best model:", best_model_name)

# Save the best model's underlying Booster (if using XGBoost).
if best_model_name == "XGBoost":
    booster = best_model.get_booster()
    # Attach feature names for future reference.
    booster.feature_names = X_train.columns.tolist()
    booster.save_model("model.bst")
else:
    import joblib
    joblib.dump(best_model, "model.pkl")
```
### Usage Instructions:

Update the BigQuery query with your dataset details.

Execute the notebook cells to perform EDA, model training, and save the best model.

``` python
from google.cloud import aiplatform

# Initialize Vertex AI.
aiplatform.init(
    project="my-project-id",                # Masked project ID.
    location="my-region",                   # e.g., 'us-central1'.
    staging_bucket="gs://my-bucket/staging"   # Masked GCS bucket.
)

# Upload the model artifact.
model = aiplatform.Model.upload(
    display_name="my-taxi-xgboost-model",
    artifact_uri="gs://my-bucket/model_artifacts",  # Folder containing model.bst.
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest",
    serving_container_deployment_timeout=600  # Optional: Fail fast if deployment stalls.
)
print("Model resource:", model.resource_name)

# Create an endpoint.
endpoint = aiplatform.Endpoint.create(display_name="my-taxi-endpoint")
print("Endpoint resource:", endpoint.resource_name)

# Deploy the model to the endpoint.
model.deploy(
    endpoint=endpoint,
    machine_type="n1-standard-4",
    traffic_split={"0": 100}
)
print("Deployment complete. Endpoint is ready for predictions.")
```

### 5. Testing the Deployed Model
**File/Script: `test_endpoint.py`**

**Purpose:**

Tests the deployed Vertex AI endpoint using a sample instance from your test data.
```
Test Instance Example (numeric array):

For example, consider a test point with the following features:

pickup_hour: 19.0

pickup_dayofweek: 2.0

trip_distance: 1.27

passenger_count: 3.0

total_amount: 7.8

pickup_location_id: 249.0

dropoff_location_id: 107.0

pickup_month: 12.0

speed: 8.843327

pickup_weekday_Monday: 1.0

pickup_weekday_Saturday: 0.0

pickup_weekday_Sunday: 0.0

pickup_weekday_Thursday: 0.0

pickup_weekday_Tuesday: 0.0

pickup_weekday_Wednesday: 0.0

store_and_fwd_flag_Y: 0.0
```
Note: Although the model was trained with these feature names, the pre-built serving container expects a 2D numeric array (list-of-lists) in the exact order of features.

```python
from google.cloud import aiplatform

# Replace with your actual endpoint resource name.
endpoint = aiplatform.Endpoint("projects/my-project-id/locations/my-region/endpoints/ENDPOINT_ID")

# Define a test instance as a list of values in the order of the training features.
instance = [19.0, 2.0, 1.27, 3.0, 7.8, 249.0, 107.0, 12.0, 8.843327, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Send the test instance as a 2D array.
prediction = endpoint.predict(instances=[instance])
print("Prediction:", prediction)

```
### Usage Instructions:

Replace ENDPOINT_ID with your actual endpoint’s ID.

Run the script to retrieve the prediction.

Compare the predicted value with the ground truth (if available) for evaluation

### How GCP Services Interact
Pub/Sub ingests raw data from external sources.

Dataflow (running an Apache Beam pipeline) continuously reads from Pub/Sub, cleans and transforms the data (using DoFns like CleanTransformRow), and writes enriched records to BigQuery.

BigQuery serves as a data warehouse that stores historical, processed data which is used for training and analysis.

Vertex AI Workbench leverages BigQuery data for model training and evaluation.

The trained model is then uploaded and deployed via Vertex AI Model & Endpoint for real-time inference.

Optionally, Dataflow can also invoke the Vertex AI endpoint to enrich streaming data with predictions, creating a feedback loop.
