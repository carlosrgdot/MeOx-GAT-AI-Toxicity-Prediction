import sys
import os
import pandas as pd
import numpy as np
import asyncio
from MeOx.exception.exception import MeOxException
from MeOx.logging.logger import logging
from MeOx.pipeline.prediction_pipeline import PredictionPipeline


class BatchPrediction:
    def __init__(self):
        try:
            logging.info("Initializing Batch Prediction output configuration...")
            self.prediction_output_dirname = "prediction_output"
            self.prediction_dir = os.path.join(os.getcwd(), self.prediction_output_dirname)

            os.makedirs(self.prediction_dir, exist_ok=True)
        except Exception as e:
            raise MeOxException(e, sys) from e

    async def start_batch_prediction(self, input_file_path: str, request=None) -> str:
        try:
            logging.info(f"Starting batch prediction for file: {input_file_path}")

            df = pd.read_csv(input_file_path,dtype=str)

            cols_check = ["Exposure dose (ug/mL)", "Exposure time"]
            needs_comma_fix = False
            for col in cols_check:
                df[col] = df[col].str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')

            required_cols = ["Exposure dose (ug/mL)", "Exposure time", "Material type"]
            df = df[required_cols].copy()
            if not all(col in df.columns for col in required_cols):
                missing = [c for c in required_cols if c not in df.columns]
                raise Exception(f"The CSV file must contain the following columns: {required_cols}. Mising: {missing}")

            df = df[required_cols].copy()
            initial_count = len(df)
            df.dropna(inplace=True)

            final_count = len(df)
            dropped = initial_count - final_count

            if dropped > 0:
                logging.warning(f"Ignoring {dropped} rows because they have empty values in required columns")

            if df.empty:
                raise Exception("The file has no valid rows to process.")

            logging.info("Loading Prediction Model...")
            prediction_pipeline = PredictionPipeline()

            predictions = []
            confidences = []

            logging.info(f"Processing {len(df)} rows...")
            for index, row in df.iterrows():

                await asyncio.sleep(0)

                if request and await request.is_disconnected():
                    raise Exception("BATCH_CANCELLED_BY_USER")

                try:
                    single_input = pd.DataFrame([row])

                    result = prediction_pipeline.predict(single_input)

                    predictions.append(result["prediction"])
                    confidences.append(result["confidence"])
                except Exception as e:
                    logging.error(f"Error predicting row {index}: {e}")
                    predictions.append("Error")
                    confidences.append(0.0)

            df["MeOx_Prediction"] = predictions
            df["Confidence_Score"] = confidences



            output_file_name = f"prediction_{os.path.basename(input_file_path)}"
            output_file_path = os.path.join(self.prediction_dir, output_file_name)

            df.to_csv(output_file_path, index=False)
            logging.info(f"Batch prediction saved successfully at: {output_file_path}")

            return output_file_path

        except Exception as e:
            raise MeOxException(e, sys) from e