# ShopSense: E-Commerce Purchase Prediction

ShopSense is a Streamlit dashboard for exploring online shopping session data and estimating whether a session will result in a purchase. It includes an overview, customer insights, a data explorer, a project guide, and a live prediction page.

## Project files

- `app.py` — Streamlit application.
- `requirements.txt` — Python dependencies.
- `online_shoppers_intention.csv` — optional dataset used for data-driven views. If it is not present, the app uses illustrative demo data.
- `balanced_purchase_prediction_model.pkl` — optional trained model required for live predictions.
- `Ecommerce_Purchase_Prediction.ipynb` — notebook used for project analysis and model training.

## Run locally on Windows

Open PowerShell in this project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually <http://localhost:8501>. Keep the terminal open while using the app. Press **Ctrl+C** in the terminal to stop it.

If PowerShell blocks virtual-environment activation, run the app with the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Model requirements

The app looks for `balanced_purchase_prediction_model.pkl` in the same folder as `app.py`. The file must be a serialized, trained Python model, not a renamed notebook. The prediction page requires a model compatible with the app's 17 input features and supporting `predict_proba()`.

Models serialized with scikit-learn may not load across different scikit-learn versions. Use the same scikit-learn version for training and running the app. To check the local version:

```powershell
.\.venv\Scripts\python.exe -c "import sklearn; print(sklearn.__version__)"
```

If the model file is absent or incompatible, the rest of the app can still be explored, but real predictions will be unavailable.

## Deploy to Streamlit Community Cloud

1. Upload the app, `requirements.txt`, the compatible trained model, and (optionally) the CSV dataset to a GitHub repository.
2. In `requirements.txt`, pin `scikit-learn` to the version used to train the model if the model relies on scikit-learn.
3. Create an app at <https://share.streamlit.io/> and select the repository, branch, and `app.py` as the app file.
4. Review the deployment logs if startup fails.

Do not upload the `.venv` folder. Files committed to a public GitHub repository are publicly accessible; consider this before uploading a model or dataset.

## Dataset

The project is based on the [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).
