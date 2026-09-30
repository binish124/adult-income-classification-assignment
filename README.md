# Adult Income Classification

This project predicts whether a record belongs to the `>50K` annual income class using the supplied `adult_raw.csv`. It completes the data preparation, two-model comparison, evaluation, explanation, and API portions of the assignment. The CSV contains 32,561 labeled rows with numeric and categorical inputs. Literal `?` values represent missing categories.

## Files

| File | Purpose |
| --- | --- |
| `data/adult_raw.csv` | User-supplied source data |
| `train.py` | Cleaning, preprocessing, model comparison, evaluation, and saved model |
| `app.py` | FastAPI prediction service |
| `REPORT.md` | Results and explanation for Tasks 1 through 4 |
| `models/metrics.json` | Machine-readable results created by training |
| `models/model.joblib` | Trained pipeline created by training |

The `models` directory is generated locally and excluded from Git. Run training before starting the API. Load a model file only from a source you trust because joblib uses Python pickle.

## Setup

Use Python 3.11 or newer. From this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
python -m uvicorn app:api --host 127.0.0.1 --port 8000
```

On macOS or Linux, activate with `source .venv/bin/activate`. `python train.py --data path/to/another_adult.csv` accepts a compatible CSV. Training prints the results and writes `models/metrics.json` and `models/model.joblib`.

## API

Open `http://127.0.0.1:8000/docs` for the interactive schema. `GET /health` shows whether the model exists. `POST /predict` accepts one JSON record with the eight selected features. Missing values can be sent as `null` or omitted; unknown categories are accepted.

```powershell
$body = @{
  age = 39
  education_num = 13
  capital_gain = 2174
  capital_loss = 0
  hours_per_week = 40
  workclass = 'State-gov'
  marital_status = 'Never-married'
  occupation = 'Adm-clerical'
} | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/predict' -Method Post -ContentType 'application/json' -Body $body
```

The response has `prediction` (`>50K` or `<=50K`) and `probability_over_50k`. The probability is the model estimate; it has not been calibrated for decisions involving people.

## Modeling choices

The positive class is `>50K`. The code uses stratified 60/20/20 training, validation, and test splits with random seed 42. Preprocessing is inside each scikit-learn pipeline, so medians, category modes, encodings, and scaling are learned from training data only. Logistic regression scales numeric features. Random forest uses the same imputation and encoding without scaling. Both use class weighting because the classes are imbalanced. The validation F1 chooses the final model, which is refitted on training plus validation data and evaluated once on the held-out test set. See `REPORT.md` for the results and limitations.

## GitHub submission

This project is published at [binish124/adult-income-classification-assignment](https://github.com/binish124/adult-income-classification-assignment). The supplied CSV is included so training can be reproduced. The generated model artifacts are excluded from Git; run `python train.py` after cloning.
