import sys
import os
import pandas as pd
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import json
from MeOx.exception.exception import MeOxException
from MeOx.logging.logger import logging
from MeOx.utils.main_utils.utils import load_object
from MeOx.components.model_trainer import GAT
from MeOx.constant.training_pipeline import (
    DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
    DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
    TRANSFORMED_DATA_FILE_NAME,
    PREPROCESSING_OBJECT_FILE_NAME,
    MODEL_TRAINER_TRAINED_MODEL_DIR,
    MODEL_FILE_NAME,
    ARTIFACT_DIR,
    GRAPH_DBSCAN_METRIC,
    GRAPH_K_INTRA_CLUSTER
)


class PredictionPipeline:
    def __init__(self):
        try:
            logging.info("Initializing Prediction Pipeline...")

            artifacts_root = os.path.join(os.getcwd(), ARTIFACT_DIR)
            if not os.path.exists(artifacts_root):
                raise Exception("No artifacts found. Please run Training Pipeline first.")

            all_folders = []
            for d in os.listdir(artifacts_root):
                full_path = os.path.join(artifacts_root, d)
                if os.path.isdir(full_path) and d.isdigit():
                    all_folders.append(full_path)

            if not all_folders:
                for d in os.listdir(artifacts_root):
                    full_path = os.path.join(artifacts_root, d)
                    if os.path.isdir(full_path) and "batch" not in d:
                        all_folders.append(full_path)

            if not all_folders:
                raise Exception(
                    "CRITICAL: No valid training artifacts found. Delete 'batch_predictions' folder manually.")


            latest_run_dir = max(all_folders, key=os.path.getmtime)


            logging.info(f"Loading artifacts from latest run: {latest_run_dir}")

            self.model_path = os.path.join(latest_run_dir, "model_trainer", MODEL_TRAINER_TRAINED_MODEL_DIR,
                                           MODEL_FILE_NAME)
            self.params_path = os.path.join(latest_run_dir, "model_trainer", MODEL_TRAINER_TRAINED_MODEL_DIR,
                                            "model_params.json")
            self.preprocessor_path = os.path.join(latest_run_dir, "data_transformation",
                                                  DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                  PREPROCESSING_OBJECT_FILE_NAME)
            self.graph_path = os.path.join(latest_run_dir, "data_transformation",
                                           DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, TRANSFORMED_DATA_FILE_NAME)

            self.preprocessor = load_object(file_path=self.preprocessor_path)
            self.base_graph = torch.load(self.graph_path, weights_only=False)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model_architecture = {
                "hidden_channels": 132,
                "dropout": 0.366,
                "num_layers": 3,
                "heads": 6,
                "activation": "relu"
            }
            if os.path.exists(self.params_path):
                try:
                    with open(self.params_path, 'r') as f:
                        loaded_params = json.load(f)
                        logging.info(f"Loaded dynamic model architecture: {loaded_params}")

                        for k in model_architecture.keys():
                            if k in loaded_params:
                                model_architecture[k] = loaded_params[k]
                except Exception as e:
                    logging.warning(f"Could not load model_params.json, using defaults. Error: {e}")
            else:
                logging.warning("No model_params.json found. Using default legacy architecture.")

            self.model = GAT(
                in_channels=self.base_graph.num_node_features,
                hidden_channels=model_architecture["hidden_channels"],
                out_channels=2,
                dropout=model_architecture["dropout"],
                num_layers=model_architecture["num_layers"],
                activation=model_architecture["activation"],
                heads=model_architecture["heads"]
            ).to(self.device)
            # -----------------------------------------

            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
            self.model.eval()

            self.reference_features = self.base_graph.x.cpu().numpy()

            self.knn_engine = NearestNeighbors(
                n_neighbors=GRAPH_K_INTRA_CLUSTER,
                metric=GRAPH_DBSCAN_METRIC
            )
            self.knn_engine.fit(self.reference_features)

            logging.info("Prediction Pipeline initialized successfully.")
        except Exception as e:
            raise MeOxException(e, sys) from e

    def predict(self, features: pd.DataFrame):
        try:

            processed_input = self.preprocessor.transform(features)

            distances, indices = self.knn_engine.kneighbors(processed_input)
            neighbor_indices = indices[0]
            neighbor_labels = self.base_graph.y[neighbor_indices].cpu().numpy()
            logging.info(f" DEBUG: Input Material: {features['Material type'].values[0]}")
            logging.info(f" DEBUG: Nearest Neighbor Indices (Graph Nodes): {neighbor_indices}")
            logging.info(f" DEBUG: Neighbor Labels (0=Safe, 1=Toxic): {neighbor_labels}")

            if all(label == 0 for label in neighbor_labels):
                logging.info(" INSIGHT: Unanimous Non-Toxic neighborhood. High confidence expected.")
            elif all(label == 1 for label in neighbor_labels):
                logging.info(" INSIGHT: Unanimous Toxic neighborhood. High confidence expected.")
            else:
                logging.info(" INSIGHT: Mixed neighborhood context. GAT attention mechanism resolving conflict.")

            new_x = torch.tensor(processed_input, dtype=torch.float)


            x_combined = torch.cat([self.base_graph.x, new_x], dim=0).to(self.device)


            new_node_idx = self.base_graph.num_nodes

            new_edges_source = []
            new_edges_target = []

            for neighbor_idx in indices[0]:

                new_edges_source.append(new_node_idx)
                new_edges_target.append(neighbor_idx)

                new_edges_source.append(neighbor_idx)
                new_edges_target.append(new_node_idx)

            new_edge_index = torch.tensor([new_edges_source, new_edges_target], dtype=torch.long)


            edge_index_combined = torch.cat([self.base_graph.edge_index, new_edge_index], dim=1).to(self.device)


            with torch.no_grad():
                logits = self.model(x_combined, edge_index_combined)


                target_logit = logits[-1].unsqueeze(0)
                probs = target_logit.softmax(dim=1)


                prediction = target_logit.argmax(dim=1).item()
                confidence = probs[0][prediction].item()


                result_text = "Toxic" if prediction == 1 else "Non-Toxic"

                logging.info(f"Prediction: {result_text} ({confidence:.4f})")

                return {
                    "prediction": result_text,
                    "confidence": float(confidence),
                    "is_toxic": bool(prediction)
                }

        except Exception as e:
            raise MeOxException(e, sys) from e