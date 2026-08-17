# ServiceGuard-GNN

> Graph Attention Network based microservice fault localization

**🚀 Live Demo:** [Try ServiceGuard-GNN](https://serviceguard-gnn-jpzavgjqhwnhojdtmdhfmy.streamlit.app/)
## How It Works

1. Service latency data is provided for a distributed application.
2. A normal-latency baseline is used to derive service-level features.
3. Each service is represented as a graph node with four features:
   - Raw latency
   - Latency ratio
   - Rank
   - Contribution
4. Services are connected as a fully connected graph.
5. A Graph Attention Network learns relationships between services and predicts the most likely fault class.
6. The system also provides:
   - Fault confidence
   - Top predictions
   - Latency-based service diagnosis
   - GAT attention relationships

The project currently supports four applications:

- Hotel Reservation
- Media Service
- Social Network
- Ticket Booking

## Dataset

The dataset contains service-level latency observations from distributed applications.

The dataset was used to:
- Establish normal latency baselines for each service
- Generate latency-based features
- Construct service graphs
- Train and evaluate the GAT models
- Compare fault-localization performance across applications

Each application has a different number of services, while every service is represented using the same four engineered features.

## User Interaction

The current Streamlit application allows the user to:

1. Select an application.
2. Upload a CSV containing the observed latency of its services.
3. The system converts the service latencies into the four required features.
4. The features are transformed into a service graph.
5. The trained GAT model performs inference.
6. The application displays:
   - Predicted faulty service
   - Prediction confidence
   - Top fault predictions
   - Service-level latency diagnosis
   - GAT attention relationships

For example, a Hotel Reservation trace contains one row with the latency of its six services. The system converts those observations into a `6 × 4` node-feature matrix before passing the graph to the GAT.

## Current Output

Example:

**Predicted Fault:** `2_geo`

**Confidence:** `80.84%`

The system also identifies severe latency anomalies such as:

`4_profile → 20.59× slower than its normal baseline`

and reports the severity and contribution of each service.

## Future Work

The next stage is to move from manual trace analysis toward continuous production monitoring.

The planned architecture is:

Production Services
→ Distributed Traces
→ Trace Processing
→ Service-Level Features
→ ServiceGuard-GNN
→ Fault Localization
→ Alert

The system will eventually analyze traces continuously and maintain a history of detected faults. If the same service repeatedly becomes a risk across multiple traces, the system can identify recurring problems and notify the service owner that the service or its underlying infrastructure may require investigation or optimization.

This would transform ServiceGuard-GNN from a trace-analysis prototype into a continuous microservice fault-monitoring and early-warning system.
