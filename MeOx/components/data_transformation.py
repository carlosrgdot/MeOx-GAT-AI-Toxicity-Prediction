import sys
import os
import pandas as pd
import numpy as np
import networkx as nx
import torch
import unicodedata
import re
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

from MeOx.exception.exception import MeOxException
from MeOx.logging.logger import logging
from MeOx.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from MeOx.entity.config_entity import DataTransformationConfig
from MeOx.utils.main_utils.utils import save_object
from MeOx.constant.training_pipeline import (
    TARGET_COLUMN,
    DATA_TRANSFORMATION_NUMERICAL_COLS,
    DATA_TRANSFORMATION_CATEGORICAL_COLS,
    GRAPH_DBSCAN_METRIC,
    GRAPH_MIN_SAMPLES,
    GRAPH_EPS_QUANTILE,
    GRAPH_K_INTRA_CLUSTER
)

DESCRIPTIVE_COLS_TO_DROP = [
    "Method core size", "Method hydro size", "Method surface charge",
    "Method surface area", "Assay", "Cell name", "Cell species",
    "Cell origin", "Cell type"
]
PROXY_COLS_TO_DROP = ["Viability (%)"]


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise MeOxException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path, sep=None, engine="python", encoding="utf-8-sig")
        except Exception as e:
            raise MeOxException(e, sys) from e

    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            logging.info('Creating preprocessor object (MinMax + OneHot)...')
            num_pipeline = Pipeline(steps=[('scaler', MinMaxScaler())])
            cat_pipeline = Pipeline(
                steps=[('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, DATA_TRANSFORMATION_NUMERICAL_COLS),
                ('cat_pipeline', cat_pipeline, DATA_TRANSFORMATION_CATEGORICAL_COLS)
            ])
            return preprocessor
        except Exception as e:
            raise MeOxException(e, sys) from e

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Starting EXACT Data Cleaning...")

            cols = pd.Index(df.columns.astype(str))
            cols = cols.str.replace(r"^\ufeff", "", regex=True).str.strip()
            cols = cols.map(lambda s: unicodedata.normalize("NFKC", s))
            df.columns = cols

            df = df.dropna(axis=1, how="all")
            df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False, case=False)]

            cols_to_drop = DESCRIPTIVE_COLS_TO_DROP + PROXY_COLS_TO_DROP
            existing_cols = [c for c in cols_to_drop if c in df.columns]
            df = df.drop(columns=existing_cols)
            logging.info(f"Dropped descriptive columns: {existing_cols}")

            if "Exposure time" in df.columns:
                df["Exposure time"] = (
                    df["Exposure time"]
                    .astype(str)
                    .str.extract(r"(\d+\.?\d*)", expand=False)
                    .astype(float)
                ).fillna(0.0)

            dose_cols = [c for c in df.columns if "Exposure dose" in c]
            for col in dose_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

            toxicity_mapping = {
                "nontoxic": 0, "toxic": 1,
                "non-toxic": 0, "non toxic": 0,
                "1": 1, "0": 0, 1: 1, 0: 0
            }
            if TARGET_COLUMN in df.columns:
                df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip().str.lower().map(toxicity_mapping)
                df = df.dropna(subset=[TARGET_COLUMN])
                df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

            initial_len = len(df)
            df = df.drop_duplicates(keep="first")
            logging.info(f"Dropped {initial_len - len(df)} duplicates. Expecting ~643 rows.")

            return df

        except Exception as e:
            raise MeOxException(e, sys) from e

    def _find_dbscan_eps(self, X: np.ndarray) -> float:
        try:
            nbrs = NearestNeighbors(n_neighbors=GRAPH_MIN_SAMPLES, metric=GRAPH_DBSCAN_METRIC).fit(X)
            distances, _ = nbrs.kneighbors(X)
            k_distances = np.sort(distances[:, GRAPH_MIN_SAMPLES - 1])
            eps_value = float(np.quantile(k_distances, GRAPH_EPS_QUANTILE))
            logging.info(f"Suggested Eps (quantile {GRAPH_EPS_QUANTILE}): {eps_value:.4f}")
            return eps_value
        except Exception as e:
            raise MeOxException(e, sys) from e

    def _build_graph_logic(self, x_array: np.ndarray, y_array: np.ndarray,
                           train_mask_full: np.ndarray, test_mask_full: np.ndarray) -> Data:
        try:
            eps = self._find_dbscan_eps(x_array)
            logging.info("Running DBSCAN...")
            dbscan = DBSCAN(eps=eps, min_samples=GRAPH_MIN_SAMPLES, metric=GRAPH_DBSCAN_METRIC, n_jobs=-1)
            labels = dbscan.fit_predict(x_array)

            n_clusters = len(set(labels) - {-1})
            n_outliers = np.sum(labels == -1)
            logging.info(f"DBSCAN Result: {n_clusters} clusters, {n_outliers} outliers.")

            G = nx.Graph()
            valid_node_indices = np.where(labels != -1)[0]
            G.add_nodes_from(valid_node_indices)

            all_edges = set()
            for cluster_id in sorted(set(labels) - {-1}):
                indices_in_cluster = np.where(labels == cluster_id)[0]
                if len(indices_in_cluster) <= 1:
                    continue

                X_cluster = x_array[indices_in_cluster]
                n_neighbors = min(GRAPH_K_INTRA_CLUSTER, len(indices_in_cluster) - 1)
                knn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=GRAPH_DBSCAN_METRIC).fit(X_cluster)
                _, indices = knn.kneighbors(X_cluster)

                for i, neighbors in enumerate(indices):
                    source_original_idx = indices_in_cluster[i]
                    for neighbor_local_idx in neighbors[1:]:
                        target_original_idx = indices_in_cluster[neighbor_local_idx]
                        edge = tuple(sorted((source_original_idx, target_original_idx)))
                        all_edges.add(edge)

            G.add_edges_from(all_edges)

            if G.number_of_nodes() == 0:
                raise Exception("Graph is empty after DBSCAN filtering.")

            logging.info(f"NX Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

            node_indices_list = list(G.nodes())
            mapping = {old_id: new_id for new_id, old_id in enumerate(node_indices_list)}
            G_relabeled = nx.relabel_nodes(G, mapping, copy=True)

            x_filtered = x_array[node_indices_list]
            y_filtered = y_array[node_indices_list]

            train_mask_filtered = train_mask_full[node_indices_list]
            test_mask_filtered = test_mask_full[node_indices_list]

            x_tensor = torch.tensor(x_filtered, dtype=torch.float)
            y_tensor = torch.tensor(y_filtered, dtype=torch.long)
            train_mask_tensor = torch.tensor(train_mask_filtered, dtype=torch.bool)
            test_mask_tensor = torch.tensor(test_mask_filtered, dtype=torch.bool)

            logging.info(f"FINAL TENSOR SHAPE X: {x_tensor.shape}")

            pyg_data = from_networkx(G_relabeled)
            pyg_data.x = x_tensor
            pyg_data.y = y_tensor
            pyg_data.train_mask = train_mask_tensor
            pyg_data.test_mask = test_mask_tensor

            return pyg_data

        except Exception as e:
            raise MeOxException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Initiating Data Transformation (Single Artifact Strategy)")

            data_file_path = self.data_validation_artifact.valid_data_file_path
            full_df = DataTransformation.read_data(data_file_path)

            full_df = self._clean_dataframe(full_df)

            input_feature_df = full_df.drop(columns=[TARGET_COLUMN], errors='ignore')
            target_feature_df = full_df[TARGET_COLUMN]

            new_len = len(full_df)
            split_idx = int(new_len * 0.8)
            full_train_mask = np.zeros(new_len, dtype=bool)
            full_train_mask[:split_idx] = True
            full_test_mask = np.zeros(new_len, dtype=bool)
            full_test_mask[split_idx:] = True

            preprocessor = self.get_data_transformer_object()
            transformed_arr = preprocessor.fit_transform(input_feature_df)

            logging.info("Building Single Solid Graph...")
            graph_data = self._build_graph_logic(
                transformed_arr,
                target_feature_df.values,
                full_train_mask,
                full_test_mask
            )

            if graph_data is not None:
                dir_path = os.path.dirname(self.data_transformation_config.transformed_data_file_path)
                os.makedirs(dir_path, exist_ok=True)

                torch.save(graph_data, self.data_transformation_config.transformed_data_file_path)
                save_object(self.data_transformation_config.transformed_object_file_path, obj=preprocessor)

                logging.info(
                    f"Single Graph Artifact saved at {self.data_transformation_config.transformed_data_file_path}")
            else:
                raise Exception("Graph creation failed")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_data_file_path=self.data_transformation_config.transformed_data_file_path
            )
            return data_transformation_artifact

        except Exception as e:
            raise MeOxException(e, sys) from e