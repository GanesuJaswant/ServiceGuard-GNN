from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch

from src.model import ServiceGuardGAT
from src.features import build_features


class ServiceGuardInference:

    def __init__(
        self,
        models_dir,
        metadata_path,
        baselines_path,
        device=None
    ):

        self.models_dir = Path(models_dir)
        self.metadata_path = Path(metadata_path)
        self.baselines_path = Path(baselines_path)

        self.device = (
            device
            if device is not None
            else torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        # -----------------------------
        # Load metadata
        # -----------------------------

        with open(
            self.metadata_path,
            "r"
        ) as f:

            self.metadata = json.load(f)

        # -----------------------------
        # Load baselines
        # -----------------------------

        with open(
            self.baselines_path,
            "r"
        ) as f:

            self.baselines = json.load(f)

        # -----------------------------
        # Containers
        # -----------------------------

        self.models = {}
        self.scalers = {}

        # -----------------------------
        # Load everything
        # -----------------------------

        self._load_artifacts()

    # ==================================================
    # Load models + scalers
    # ==================================================

    def _load_artifacts(self):

        model_files = {
            "hotel-reservation":
                "hotel_gat.pth",

            "media-service":
                "media_gat_8head_64hidden.pth",

            "social-network":
                "social_gat.pth",

            "ticket-booking":
                "ticket_gat.pth"
        }

        scaler_files = {
            "hotel-reservation":
                "hotel_reservation_scaler.pkl",

            "media-service":
                "media_service_scaler.pkl",

            "social-network":
                "social_network_scaler.pkl",

            "ticket-booking":
                "ticket_booking_scaler.pkl"
        }

        for app, info in self.metadata.items():

            # -----------------------------
            # Model architecture
            # -----------------------------

            model = ServiceGuardGAT(
                num_classes=info[
                    "num_classes"
                ],
                hidden_channels=info[
                    "hidden_channels"
                ],
                heads=info[
                    "attention_heads"
                ]
            ).to(self.device)

            # -----------------------------
            # Load checkpoint
            # -----------------------------

            checkpoint_path = (
                self.models_dir
                / model_files[app]
            )

            state = torch.load(
                checkpoint_path,
                map_location=self.device
            )

            model.load_state_dict(
                state
            )

            model.eval()

            self.models[app] = model

            # -----------------------------
            # Load scaler
            # -----------------------------

            scaler_path = (
                self.models_dir
                / scaler_files[app]
            )

            self.scalers[app] = joblib.load(
                scaler_path
            )

    # ==================================================
    # Build fully-connected graph
    # ==================================================

    def _build_graph(
        self,
        node_features
    ):

        num_nodes = len(
            node_features
        )

        x = torch.tensor(
            node_features,
            dtype=torch.float32,
            device=self.device
        )

        source = []
        target = []

        for i in range(num_nodes):

            for j in range(num_nodes):

                source.append(i)
                target.append(j)

        edge_index = torch.tensor(
            [source, target],
            dtype=torch.long,
            device=self.device
        )

        batch = torch.zeros(
            num_nodes,
            dtype=torch.long,
            device=self.device
        )

        return (
            x,
            edge_index,
            batch
        )

    # ==================================================
    # Prediction
    # ==================================================

    def predict(
        self,
        trace,
        app
    ):

        if app not in self.metadata:

            raise ValueError(
                f"Unknown application: {app}"
            )

        info = self.metadata[app]

        service_names = info[
            "service_names"
        ]

        num_nodes = info[
            "num_nodes"
        ]

        # -----------------------------
        # Validate input
        # -----------------------------

        missing = [
            service
            for service in service_names
            if service not in trace.columns
        ]

        if missing:

            raise ValueError(
                f"Missing service columns: "
                f"{missing}"
            )

        # -----------------------------
        # Baseline
        # -----------------------------

        baseline = pd.Series(
            self.baselines[app],
            index=service_names,
            dtype=float
        )

        # -----------------------------
        # Build 4 features
        # -----------------------------

        features = build_features(
            trace,
            service_names,
            baseline
        )

        # -----------------------------
        # Scale
        # -----------------------------

        scaler = self.scalers[app]

        scaled = scaler.transform(
            features
        )

        # -----------------------------
        # Convert to node features
        # -----------------------------

        n = num_nodes

        node_features = np.column_stack(
            [
                scaled[0, :n],
                scaled[0, n:2*n],
                scaled[0, 2*n:3*n],
                scaled[0, 3*n:4*n]
            ]
        )

        # -----------------------------
        # Build graph
        # -----------------------------

        (
            x,
            edge_index,
            batch
        ) = self._build_graph(
            node_features
        )

        # -----------------------------
        # Prediction
        # -----------------------------

        model = self.models[app]

        with torch.no_grad():

            output = model(
                x,
                edge_index,
                batch
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )[0]

        predicted_class = int(
            probabilities.argmax().item()
        )

        confidence = float(
            probabilities[
                predicted_class
            ].item()
        )

        # -----------------------------
        # Fault name
        # -----------------------------

        if predicted_class == 0:

            fault_name = (
                "No interference"
            )

        else:

            fault_name = service_names[
                predicted_class - 1
            ]

        # -----------------------------
        # Top predictions
        # -----------------------------

        k = min(
            5,
            len(probabilities)
        )

        top_probs, top_classes = (
            torch.topk(
                probabilities,
                k=k
            )
        )

        top_predictions = []

        for cls, prob in zip(
            top_classes.cpu().tolist(),
            top_probs.cpu().tolist()
        ):

            if cls == 0:

                name = (
                    "No interference"
                )

            else:

                name = service_names[
                    cls - 1
                ]

            top_predictions.append(
                {
                    "class": int(cls),
                    "fault_name": name,
                    "probability": (
                        float(prob) * 100
                    )
                }
            )

        return {
            "application": app,
            "predicted_class": predicted_class,
            "predicted_fault": fault_name,
            "confidence": confidence * 100,
            "top_predictions": top_predictions,
            "features": features,
            "scaled_features": scaled,
            "node_features": node_features,
            "edge_index": edge_index.cpu()
        }

                    # ==================================================
    # Service diagnosis
    # ==================================================

    def diagnose(
        self,
        trace,
        app
    ):

        if app not in self.metadata:

            raise ValueError(
                f"Unknown application: {app}"
            )

        service_names = self.metadata[
            app
        ]["service_names"]

        baseline = pd.Series(
            self.baselines[app],
            index=service_names,
            dtype=float
        )

        rows = []

        total_contribution = 0.0

        # ------------------------------------------
        # Calculate slowdown
        # ------------------------------------------

        for service in service_names:

            observed = float(
                trace.iloc[0][service]
            )

            normal = float(
                baseline[service]
            )

            slowdown = (
                observed / normal
                if normal != 0
                else 0.0
            )

            contribution = max(
                slowdown - 1.0,
                0.0
            )

            rows.append(
                {
                    "service": service,
                    "observed_latency": observed,
                    "normal_latency": normal,
                    "slowdown_x": slowdown,
                    "_raw_contribution": contribution
                }
            )

            total_contribution += contribution

        # ------------------------------------------
        # Contribution percentage + severity
        # ------------------------------------------

        for row in rows:

            if total_contribution > 0:

                contribution_pct = (
                    row["_raw_contribution"]
                    / total_contribution
                    * 100
                )

            else:

                contribution_pct = 0.0

            slowdown = row["slowdown_x"]

            if slowdown >= 5:

                severity = "CRITICAL"

            elif slowdown >= 2:

                severity = "HIGH"

            elif slowdown >= 1.2:

                severity = "LOW"

            else:

                severity = "NORMAL"

            row["contribution_pct"] = (
                contribution_pct
            )

            row["severity"] = severity

            del row[
                "_raw_contribution"
            ]

        diagnosis = pd.DataFrame(
            rows
        )

        diagnosis = diagnosis.sort_values(
            "contribution_pct",
            ascending=False
        ).reset_index(
            drop=True
        )

        return diagnosis
        # ==================================================
    # GAT attention relationships
    # ==================================================

    def get_attention(
        self,
        trace,
        app,
        top_k=10
    ):

        if app not in self.metadata:
            raise ValueError(
                f"Unknown application: {app}"
            )

        info = self.metadata[app]
        service_names = info["service_names"]

        # ------------------------------------------
        # Build features
        # ------------------------------------------

        baseline = pd.Series(
            self.baselines[app],
            index=service_names,
            dtype=float
        )

        features = build_features(
            trace,
            service_names,
            baseline
        )

        # ------------------------------------------
        # Scale
        # ------------------------------------------

        scaler = self.scalers[app]

        scaled = scaler.transform(
            features
        )

        n = len(service_names)

        node_features = np.column_stack(
            [
                scaled[0, :n],
                scaled[0, n:2*n],
                scaled[0, 2*n:3*n],
                scaled[0, 3*n:4*n]
            ]
        )

        # ------------------------------------------
        # Build graph
        # ------------------------------------------

        x, edge_index, batch = (
            self._build_graph(
                node_features
            )
        )

        model = self.models[app]

        model.eval()

        # ------------------------------------------
        # Extract attention from GAT layers
        # ------------------------------------------

        with torch.no_grad():

            _, attention_1 = model.gat1(
                x,
                edge_index,
                return_attention_weights=True
            )

            x1 = torch.relu(
                model.gat1(
                    x,
                    edge_index
                )
            )

            _, attention_2 = model.gat2(
                x1,
                edge_index,
                return_attention_weights=True
            )

        # ------------------------------------------
        # Use second GAT layer attention
        # ------------------------------------------

        edge_index_attn, alpha = (
            attention_2
        )

        # Average attention across heads
        mean_attention = (
            alpha.mean(dim=1)
            .cpu()
            .numpy()
        )

        edges = (
            edge_index_attn
            .cpu()
            .numpy()
        )

        rows = []

        for i in range(
            edges.shape[1]
        ):

            source_idx = int(
                edges[0, i]
            )

            target_idx = int(
                edges[1, i]
            )

            rows.append(
                {
                    "source": service_names[
                        source_idx
                    ],
                    "target": service_names[
                        target_idx
                    ],
                    "attention": float(
                        mean_attention[i]
                    )
                }
            )

        attention_df = pd.DataFrame(
            rows
        )

        attention_df = (
            attention_df
            .sort_values(
                "attention",
                ascending=False
            )
            .head(top_k)
            .reset_index(drop=True)
        )

        return attention_df
        # ==================================================
    # Complete trace analysis
    # ==================================================

    def analyze(
        self,
        trace,
        app,
        top_k=10
    ):

        prediction = self.predict(
            trace,
            app
        )

        diagnosis = self.diagnose(
            trace,
            app
        )

        attention = self.get_attention(
            trace,
            app,
            top_k=top_k
        )

        return {
            "application": app,
            "prediction": prediction,
            "service_diagnosis": diagnosis,
            "attention_relationships": attention
        }