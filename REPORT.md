# Adult Income Classification Assignment Report

## Problem and data

The task is to classify whether a record is in the `>50K` income group. The supplied `adult_raw.csv` has 32,561 labeled rows, 14 source features, and one income label. The classes are uneven: 24,720 rows are `<=50K` and 7,841 are `>50K`. This is a demonstration model for an assignment, not a basis for decisions about individuals.

## Task 1 Data understanding and preprocessing

The CSV mixes numeric values, such as age and hours worked, with categories, such as workclass and occupation. It has no native CSV nulls, but `?` marks missing values in `workclass` (1,836 rows), `occupation` (1,843), and `native_country` (583). The model uses eight inputs: `age`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`, `workclass`, `marital_status`, and `occupation`.

The training script trims category whitespace, converts `?` and empty strings to missing values, parses numeric columns, and checks labels. A median imputer handles missing numeric values and a most-frequent imputer handles missing categories. One-hot encoding turns categories into model inputs and ignores new categories at prediction time. Logistic regression receives standardized numeric inputs; random forest does not require numeric scaling. All preprocessing is fitted inside each training pipeline to prevent information from the validation or test sets leaking into training.

`education` is omitted because `education_num` carries the same ordering. `fnlwgt` is a survey weight rather than a simple personal attribute. `race`, `sex`, `native_country`, and `relationship` are omitted to avoid using those direct fields. The remaining fields can still act as proxies, so omission alone does not establish fairness.

## Task 2 Model building and evaluation

I trained logistic regression as a simple interpretable baseline and a random forest to capture nonlinear relationships and interactions. Both use class weighting to address the smaller `>50K` class. A stratified 60/20/20 split produced 19,536 training, 6,512 validation, and 6,513 test rows. The random seed is 42. The positive class for precision, recall, and F1 is `>50K`.

| Model | Validation accuracy | Validation precision | Validation recall | Validation F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.8054 | 0.5633 | 0.8540 | 0.6788 |
| Random forest | 0.8345 | 0.6134 | 0.8450 | 0.7108 |

Random forest had the higher validation F1. I refitted it on training plus validation data, then evaluated it on the held-out test set.

| Test accuracy | Test precision | Test recall | Test F1 |
| ---: | ---: | ---: | ---: |
| 0.8339 | 0.6175 | 0.8158 | 0.7029 |

Always predicting `<=50K` would yield 0.7591 test accuracy. Accuracy alone would therefore hide poor detection of the `>50K` group. The final model captures about 81.6% of the positive test records, while about 38.3% of its positive predictions are incorrect. The exact machine-readable results are in `models/metrics.json` after training.

## Task 3 AI logic and explanation

Random forest was selected because it performed best on the predeclared validation F1 comparison. To estimate input impact, I shuffled one feature at a time on the test set three times and measured the mean drop in F1. The largest drops were `marital_status` (0.1632), `capital_gain` (0.0540), `education_num` (0.0522), `age` (0.0393), and `occupation` (0.0305). These are associations in this dataset and model; they do not show that changing a person's feature causes income to change. Correlated inputs can also share importance, so the numbers should not be read as causal rankings.

Possible improvements include tuning hyperparameters with cross-validation inside the training partition, calibrating probabilities, adjusting the classification threshold for a stated use case, testing on newer or different populations, and auditing errors across demographic groups. The dataset is historical and the model has not been validated for real deployment.

## Task 4 Deployment and API

`app.py` exposes `GET /health` and `POST /predict` with FastAPI. A POST request accepts the eight selected inputs and returns JSON containing the predicted label and the probability of `>50K`. Input types and numeric ranges are checked. The full preprocessing pipeline is saved with the classifier, so the API applies the same transformations used during training. Start it with `python -m uvicorn app:api --host 127.0.0.1 --port 8000`; see `README.md` for a request example.

## Task 5 Git and documentation

`README.md` contains setup, training, API usage, model explanation, and GitHub publishing commands. The code, report, requirements, and supplied CSV are ready for version control. Publishing requires a GitHub repository URL and access to that account.
