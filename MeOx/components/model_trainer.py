import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import mlflow
from urllib.parse import urlparse
import json
from dotenv import load_dotenv
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, f1_score, roc_auc_score
)
from torch_geometric.nn import GATConv

from MeOx.exception.exception import MeOxException
from MeOx.logging.logger import logging
from MeOx.entity.config_entity import ModelTrainerConfig
from MeOx.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from MeOx.constant.training_pipeline import (
    MODEL_TRAINER_RANDOM_SEED,
    MODEL_TRAINER_OPTUNA_TRIALS,
    MODEL_TRAINER_CV_FOLDS,
    MODEL_TRAINER_TEST_SIZE,
    MODEL_TRAINER_MAX_EPOCHS_OPTUNA,
    MODEL_TRAINER_PATIENCE_OPTUNA,
    MODEL_TRAINER_MAX_EPOCHS_FINAL,
    MODEL_TRAINER_PATIENCE_FINAL
)

load_dotenv()


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, activation, heads):
        super().__init__()
        if isinstance(activation, str):
            self.activation = {'relu': F.relu, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid}[activation]
        else:
            self.activation = activation
        self.dropout = dropout
        self.convs = nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout))

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=dropout))

        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.set_seed(MODEL_TRAINER_RANDOM_SEED)
        except Exception as e:
            raise MeOxException(e, sys) from e

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def get_class_weights(self, data, mask):
        counts = torch.bincount(data.y[mask]).clamp(min=1)
        num_classes = int(data.y.max().item()) + 1
        return (counts.sum() / (counts * num_classes)).to(data.x.device)

    def objective(self, trial, data, train_indices_for_cv):
        self.set_seed(MODEL_TRAINER_RANDOM_SEED)

        params = {
            'hidden_channels': trial.suggest_int('hidden_channels', 32, 256),
            'dropout': trial.suggest_float('dropout', 0.2, 0.6),
            'lr': trial.suggest_float('lr', 5e-4, 5e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'num_layers': trial.suggest_int('num_layers', 2, 5),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'heads': trial.suggest_int('heads', 2, 8)
        }

        model_params = {k: v for k, v in params.items() if k not in ['lr', 'weight_decay']}
        y_full = data.y.cpu().numpy()
        y_train = y_full[train_indices_for_cv]

        skf = StratifiedKFold(n_splits=MODEL_TRAINER_CV_FOLDS, shuffle=True, random_state=MODEL_TRAINER_RANDOM_SEED)
        fold_scores = []

        for fold, (train_idx_rel, val_idx_rel) in enumerate(skf.split(train_indices_for_cv, y_train)):
            fold_train_idx = torch.tensor(train_indices_for_cv[train_idx_rel], device=self.device)
            fold_val_idx = torch.tensor(train_indices_for_cv[val_idx_rel], device=self.device)

            model = GAT(
                in_channels=data.num_node_features,
                out_channels=int(data.y.max().item()) + 1,
                **model_params
            ).to(self.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
            class_weights = self.get_class_weights(data, fold_train_idx)

            best_fold_bal_acc = 0.0
            wait = 0

            for epoch in range(MODEL_TRAINER_MAX_EPOCHS_OPTUNA):
                model.train()
                optimizer.zero_grad()
                logits = model(data.x, data.edge_index)
                loss = F.cross_entropy(logits[fold_train_idx], data.y[fold_train_idx], weight=class_weights)
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    probs = model(data.x, data.edge_index).softmax(dim=1)[:, 1]
                    y_v = data.y[fold_val_idx].cpu().numpy()
                    p_v = probs[fold_val_idx].cpu().numpy()
                    pred_v = (p_v >= 0.5).astype(int)
                    current_bal_acc = balanced_accuracy_score(y_v, pred_v)

                if current_bal_acc > best_fold_bal_acc:
                    best_fold_bal_acc = current_bal_acc
                    wait = 0
                else:
                    wait += 1
                    if wait >= MODEL_TRAINER_PATIENCE_OPTUNA:
                        break

            fold_scores.append(best_fold_bal_acc)
            trial.report(np.mean(fold_scores), fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(fold_scores)

    def train_final_model(self, data, best_params, train_mask, val_mask):
        model_params = {k: v for k, v in best_params.items() if k not in ['lr', 'weight_decay']}

        model = GAT(
            in_channels=data.num_node_features,
            out_channels=int(data.y.max().item()) + 1,
            **model_params
        ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay']
        )
        class_weights = self.get_class_weights(data, train_mask)

        best_f1_val, wait, best_state = -1.0, 0, None

        for epoch in range(MODEL_TRAINER_MAX_EPOCHS_FINAL):
            model.train()
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index)
            loss = F.cross_entropy(logits[train_mask], data.y[train_mask], weight=class_weights)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                preds = model(data.x, data.edge_index).argmax(dim=1)
                y_v = data.y[val_mask].cpu().numpy()
                p_v = preds[val_mask].cpu().numpy()
                val_f1 = f1_score(y_v, p_v, pos_label=1, zero_division=0)

            if val_f1 > best_f1_val:
                best_f1_val, wait = val_f1, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= MODEL_TRAINER_PATIENCE_FINAL:
                    break

        if best_state:
            model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        return model

    def find_optimal_threshold(self, probs, y_true):
        best_thr = 0.5
        best_metric = -1.0

        for t in np.linspace(0.01, 0.99, 200):
            preds = (probs >= t).astype(int)
            metric = balanced_accuracy_score(y_true, preds)
            if metric > best_metric:
                best_thr, best_metric = t, metric

        logging.info(f"Optimal Threshold found: {best_thr:.4f}")
        return best_thr

    def track_mlflow(self, model, metrics, params):
        try:
            if not os.getenv("MLFLOW_TRACKING_URI"):
                logging.warning("MLFLOW_TRACKING_URI not found in environment variables. MLflow tracking might fail.")

            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

            logging.info(f"Logging metrics to MLflow at {os.getenv('MLFLOW_TRACKING_URI')}")

            with mlflow.start_run():
                mlflow.log_params(params)

                mlflow.log_metric("f1_score", metrics.f1_score)
                mlflow.log_metric("accuracy", metrics.accuracy)
                mlflow.log_metric("balanced_accuracy", metrics.balanced_accuracy)

                mlflow.pytorch.log_model(model, "model")
                logging.info("MLflow logging completed successfully.")

        except Exception as e:
            logging.error(f"Error occurring during MLflow tracking: {str(e)}")

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Initiating Model Training with Optuna...")
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            file_path = self.data_transformation_artifact.transformed_data_file_path
            data = torch.load(file_path, map_location=self.device, weights_only=False)
            if hasattr(data, 'to'): data = data.to(self.device)

            indices_all = np.arange(data.num_nodes)
            labels_all = data.y.cpu().numpy()

            train_indices, test_indices = train_test_split(
                indices_all, test_size=MODEL_TRAINER_TEST_SIZE,
                stratify=labels_all, random_state=MODEL_TRAINER_RANDOM_SEED
            )

            test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=self.device)
            test_mask[test_indices] = True

            logging.info(f"Starting Hyperparameter Optimization ({MODEL_TRAINER_OPTUNA_TRIALS} trials)...")
            sampler = TPESampler(seed=MODEL_TRAINER_RANDOM_SEED)
            pruner = HyperbandPruner(min_resource=1, max_resource=MODEL_TRAINER_CV_FOLDS, reduction_factor=3)

            study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
            study.optimize(lambda trial: self.objective(trial, data, train_indices),
                           n_trials=MODEL_TRAINER_OPTUNA_TRIALS)

            best_params = study.best_trial.params
            logging.info(f"Best Optuna Params: {best_params}")
            logging.info(f"Best CV Score (Bal Acc): {study.best_value:.4f}")

            logging.info("Training Final Model with Best Params...")
            tr_idx_final, val_idx_final = train_test_split(
                train_indices, test_size=0.2, stratify=labels_all[train_indices], random_state=MODEL_TRAINER_RANDOM_SEED
            )

            tr_mask_final = torch.zeros(data.num_nodes, dtype=torch.bool, device=self.device)
            tr_mask_final[tr_idx_final] = True
            val_mask_final = torch.zeros(data.num_nodes, dtype=torch.bool, device=self.device)
            val_mask_final[val_idx_final] = True

            final_model = self.train_final_model(data, best_params, tr_mask_final, val_mask_final)

            final_model.eval()
            with torch.no_grad():
                probs = final_model(data.x, data.edge_index).softmax(dim=1)[:, 1]

            val_probs = probs[val_mask_final].cpu().numpy()
            y_val = data.y[val_mask_final].cpu().numpy()

            best_thr = self.find_optimal_threshold(val_probs, y_val)

            y_test = data.y[test_mask].cpu().numpy()
            y_probs = probs[test_mask].cpu().numpy()
            y_preds = (y_probs >= best_thr).astype(int)

            test_acc = accuracy_score(y_test, y_preds)
            test_bal_acc = balanced_accuracy_score(y_test, y_preds)
            test_f1 = f1_score(y_test, y_preds, pos_label=1, zero_division=0)

            logging.info(f"Test Balanced Accuracy: {test_bal_acc:.4f}")
            logging.info(f"Test F1 Score: {test_f1:.4f}")

            test_metric_artifact = ClassificationMetricArtifact(
                f1_score=test_f1,
                accuracy=test_acc,
                balanced_accuracy=test_bal_acc
            )

            train_metric_artifact = ClassificationMetricArtifact(f1_score=0.0, accuracy=0.0, balanced_accuracy=0.0)

            self.track_mlflow(final_model, test_metric_artifact, best_params)

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            torch.save(final_model.state_dict(), self.model_trainer_config.trained_model_file_path)
            logging.info(f"Final Model saved at {self.model_trainer_config.trained_model_file_path}")
            params_file_path = os.path.join(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                "model_params.json"
            )
            with open(params_file_path, "w") as f:
                json.dump(best_params, f)
            logging.info(f"Model architecture parameters saved at: {params_file_path}")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric_artifact,
                test_metric_artifact=test_metric_artifact
            )
            return model_trainer_artifact

        except Exception as e:
            raise MeOxException(e, sys) from e