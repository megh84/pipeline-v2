# Tree Model Pipelines Dashboard

This version fixes the deployment issues by using:
- one single Python app file
- one requirements file
- one README file

## Included pipeline families

The dashboard compares these 5 tree-model families in both regression and classification form:

1. Decision Tree
2. Random Forest
3. Extra Trees
4. Gradient Boosting
5. HistGradient Boosting

## Files

- `app.py` → complete Streamlit app with dataset generation and model evaluation
- `requirements.txt` → required packages
- `README.md` → setup instructions

## Dataset expected

Your CSV should include:
- at least 50 predictor columns
- `y_continuous`
- `y_binary`

If no CSV is uploaded, the app can generate a simulated dataset automatically.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Why this update was made

The earlier deployment logs showed two main issues:
- missing `scikit-learn`
- app importing from a separate Python module that failed in deployment

This updated version keeps everything inside one app file and includes scikit-learn in requirements.
