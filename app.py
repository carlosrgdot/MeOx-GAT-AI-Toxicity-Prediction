import sys
import os
import shutil
import json
import pandas as pd
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from fastapi import HTTPException

from MeOx.logging.logger import logging
from MeOx.pipeline.training_pipeline import TrainingPipeline
from MeOx.pipeline.prediction_pipeline import PredictionPipeline
from MeOx.pipeline.batch_prediction import BatchPrediction

training_process = {"active": False, "stop_signal": False}

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists("metrics.json"):
        default_metrics = {
            "accuracy": 0.972,
            "materials": 15,
            "status": "System Ready"
        }
        with open("metrics.json", "w") as f:
            json.dump(default_metrics, f)
    yield


app = FastAPI(
    title="MeOx Toxicity Prediction AI",
    version="1.0.0",
    lifespan=lifespan
)

templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionInput(BaseModel):
    material: str
    dose: float = Field(..., gt=0, description="Dose must be positive")
    time: float = Field(..., gt=0, description="Time must be positive")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/train")
async def training_route():
    def iterfile():
        try:
            yield "Initializing Training Pipeline...\n"

            train_pipeline = TrainingPipeline()
            yield "Running Data Ingestion & Transformation...\n"

            train_pipeline.run_pipeline()

            yield "Training Completed Successfully!\n"
            yield "New Artifacts Saved.\n"
            yield "The model is now updated."

        except Exception as e:
            yield f"Error Occurred: {str(e)}\n"

    return StreamingResponse(iterfile(), media_type="text/plain")


@app.post("/predict/single")
async def predict_single(data: PredictionInput):
    try:
        input_data = {
            "Exposure dose (ug/mL)": [data.dose],
            "Exposure time": [data.time],
            "Material type": [data.material]
        }
        df = pd.DataFrame(input_data)

        pipeline = PredictionPipeline()
        result = pipeline.predict(df)

        return result

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


# --- RUTA BATCH CORREGIDA (CON AWAIT Y SEGURIDAD) ---
@app.post("/predict/batch")
async def predict_batch(request: Request, file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        return JSONResponse(status_code=400, content={"status": "error", "message": "Only .csv files are allowed"})

    MAX_FILE_SIZE = 10 * 1024 * 1024
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > MAX_FILE_SIZE:
        return JSONResponse(status_code=413,
                            content={"status": "error", "message": "File too large. Max size is 10MB."})

    file_location = f"temp_{file.filename}"

    try:
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())

        batch_pipeline = BatchPrediction()

        output_file_path = await batch_pipeline.start_batch_prediction(
            file_location,
            request=request
        )

        df = pd.read_csv(output_file_path)
        output_filename = os.path.basename(output_file_path)

        if os.path.exists(file_location):
            os.remove(file_location)

        return JSONResponse(content={
            "status": "success",
            "data": df.to_dict(orient="records"),
            "filename": output_filename
        })

    except Exception as e:
        if os.path.exists(file_location):
            os.remove(file_location)

        if "BATCH_CANCELLED_BY_USER" in str(e):
            print("Batch process killed by user reload.")
            return JSONResponse(status_code=499, content={"status": "cancelled"})

        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(os.getcwd(), "prediction_output", filename)
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename=filename, media_type='text/csv')
    return {"error": "File not found"}


@app.get("/api/metrics")
async def get_metrics():
    try:
        if os.path.exists("metrics.json"):
            with open("metrics.json", "r") as f:
                return json.load(f)
        else:
            return {"accuracy": 0.0, "materials": 0, "status": "Pending"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/cancel/train")
async def cancel_training():
    if training_process["active"]:
        training_process["stop_signal"] = True
        return {"status": "Cancelling..."}
    return {"status": "No training active"}


@app.get("/train")
async def training_route():
    def iterfile():
        new_artifact_dir = None

        try:
            training_process["active"] = True
            training_process["stop_signal"] = False

            yield "Initializing Training Pipeline...\n"

            if training_process["stop_signal"]: raise Exception("User Cancelled")

            train_pipeline = TrainingPipeline()
            yield "Ingesting Data...\n"

            model_trainer_artifact = train_pipeline.run_pipeline()
            if model_trainer_artifact:
                path_parts = model_trainer_artifact.trained_model_file_path.split(os.sep)
                if "artifacts" in path_parts:
                    idx = path_parts.index("artifacts")
                    new_artifact_dir = os.path.join(*path_parts[:idx + 2])

            if training_process["stop_signal"]:
                raise Exception("User Cancelled during training loop")

            yield "Training Completed Successfully!\n"
            yield "Dashboard metrics updating..."

        except Exception as e:
            error_msg = str(e)
            yield f"STOPPED: {error_msg}\n"

            if "User Cancelled" in error_msg and new_artifact_dir and os.path.exists(new_artifact_dir):
                yield "🧹 Cleaning up corrupted artifacts...\n"
                try:
                    shutil.rmtree(new_artifact_dir)  # BORRA LA CARPETA MALA
                    yield "System Safe. Reverted to previous model.\n"
                except:
                    yield "Warning: Could not delete temp files.\n"
            # --------------------------------

        finally:
            training_process["active"] = False
            training_process["stop_signal"] = False

    return StreamingResponse(iterfile(), media_type="text/plain")
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)