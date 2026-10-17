# StockSentryAI
AI-driven supply chain risk and financial forecasting platform for the IBM Datathon, hosted on IBM LinuxONE (s390x).
# StockSentry AI: Intelligent Medical Supply Risk & Allocation Engine

> **IBM Z Datathon 2026 Submission** | **Event Code:** `SSF2026`  
> **Theme:** AI for Critical Decisions

StockSentry AI is a real-time risk assessment engine engineered to mitigate critical medical supply shortages, optimize inventory allocation, and forecast supply chain disruption before it impacts patient care. Deployed on an enterprise-grade IBM LinuxONE virtual instance, the system pairs high-throughput REST APIs with predictive machine learning algorithms to serve actionable intelligence directly to healthcare operators.

---

## Architecture Overview

StockSentry AI implements a decoupled client-server architecture designed for low latency, secure enterprise workload isolation, and real-time inference:
```┌─────────────────────────────────┐
│     Client / Presentation       │
│  • Modern HTML5 / CSS3 / ES6    │
│  • Dynamic Fetch & DOM Updates  │
└────────────────┬────────────────┘
│ HTTP POST (JSON) / CORS
▼
┌─────────────────────────────────┐
│   IBM LinuxONE Cloud Instance   │
│  ┌───────────────────────────┐  │
│  │ FastAPI REST Service       │  │
│  │ • Pydantic Schema Mapping │  │
│  │ • Uvicorn ASGI Server     │  │
│  └─────────────┬─────────────┘  │
│                │ Feature Alignment
│                ▼                │
│  ┌───────────────────────────┐  │
│  │ Machine Learning Pipeline │  │
│  │ • Scikit-Learn Model      │  │
│  │ • Dynamic One-Hot Enc.    │  │
│  │ • Real-time Risk Scoring  │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘```

---

## Technologies Used

### Backend & Cloud Infrastructure
* **IBM Z / LinuxONE:** Enterprise-grade cloud environment hosting backend execution, offering resilient, high-security compute for mission-critical healthcare operations.
* **FastAPI:** High-performance asynchronous Python web framework used to expose clean, documented REST API endpoints (`/predict/stock-risk`).
* **Uvicorn:** Lightning-fast ASGI web server implementation powering asynchronous request handling.
* **Pydantic:** Robust data validation and settings management enforcing strict typing for supply chain and financial payload schemas.

### Machine Learning & Data Processing
* **Scikit-Learn:** Machine learning framework utilized for training and inference of the risk-assessment ensemble model.
* **Pandas & NumPy:** In-memory data manipulation, real-time feature alignment, and dynamic one-hot encoding across categorical supply parameters.
* **Joblib:** Optimized serialization and loading of trained model weights (`stocksentry_model.pkl`).

### Frontend & Client Interface
* **HTML5 / CSS3:** Modern, responsive dashboard interface designed for real-time risk reporting and healthcare inventory tracking.
* **JavaScript (Fetch API):** Client-side asynchronous networking and real-time DOM mutation enabling dynamic UI updates without page reloads.

---

## Repository Structure

```text
├── backend_api/
│   ├── main.py              # FastAPI application & ML feature alignment
│   ├── requirements.txt     # Backend dependencies
│   └── ml_training/         # Local model storage directory
├── ml_training/
│   ├── StockSentry_Training.ipynb  # ML model exploration and training
│   └── stocksentry_model.pkl       # Serialized Scikit-Learn model
├── UI/
│   └── index.html           # Real-time risk assessment dashboard
└── .gitignore               # System, cache, and artifact exclusions

## Getting Started

### Prerequisites
* Python 3.10+
* Git[cite: 11]

### Local / Server Setup

1. **Clone the repository:**[cite: 11]
   ```bash
   git clone [https://github.com/]https://github.com/kumar-pavan-kb/StockSentryAI.git
   cd StockSentryAI
2. **Install backend dependencies:**
   ```bash
   pip install -r backend_api/requirements.txt
3. **Launch the FastAPI Server:**
   ```bash
   cd backend_api
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
Launch the Dashboard:
Open UI/index.html directly in any modern browser. Enter supply parameters and run live evaluations against the predictive model.
