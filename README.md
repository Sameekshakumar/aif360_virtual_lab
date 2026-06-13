# AIF360 Virtual Lab

Interactive web app for exploring algorithmic bias and fairness mitigation techniques using IBM's AIF360 toolkit. Run real bias detection and mitigation algorithms (Reweighing, Disparate Impact Remover, Adversarial Debiasing, Exponentiated Gradient Reduction) on real-world datasets (Adult Census, German Credit, COMPAS) and visualize before/after fairness metrics — with plain-English explanations designed for students with no machine learning background.

## Tech Stack
- **Frontend:** React + Vite
- **Backend:** FastAPI + AIF360 + scikit-learn + TensorFlow

## Running Locally

### Prerequisites
- Python 3.10 (required for AIF360 + TensorFlow compatibility — newer versions like 3.12/3.13 will cause errors)
- Node.js + npm
- conda (recommended) or venv

### Backend Setup

```bash
# create and activate a Python 3.10 environment
conda create -n aif360 python=3.10
conda activate aif360

# from the project root, install dependencies
pip install -r requirements.txt

# start the backend server
uvicorn main:app --reload
```

The backend will run at `http://localhost:8000`. The first run may take a minute as TensorFlow loads.

### Frontend Setup

Open a **new terminal** (keep the backend running in the first one):

```bash
cd frontend
npm install
npm run dev
```

The frontend will run at `http://localhost:5173` (Vite will print the exact URL).

### Usage

1. Start the backend first (`uvicorn main:app --reload`) — wait until you see "Application startup complete".
2. Start the frontend (`npm run dev`).
3. Open the frontend URL shown in your terminal in a browser.
4. The "Backend Connected" indicator (if shown) confirms the two are talking to each other.

## Project Structure
aif360_virtual_lab/
├── main.py              # FastAPI backend — all algorithm endpoints
├── requirements.txt      # Python dependencies
├── backend/data/         # Dataset files (Adult, German, COMPAS)
├── frontend/              # React + Vite frontend
│   ├── src/App.jsx       # Main application
│   └── public/           # Icons, Jupyter notebooks for download
└── Security_analysis.ipynb

## Troubleshooting

- **`ModuleNotFoundError`**: make sure your conda environment is activated and `pip install -r requirements.txt` completed without errors.
- **TensorFlow / numpy errors**: confirm you're using Python 3.10 — `python --version`. AIF360's adversarial debiasing algorithm is incompatible with Python 3.11+.
- **Frontend shows "Could not reach backend"**: make sure uvicorn is running on port 8000 and there are no errors in its terminal.
