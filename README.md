# Internship Ready

The app now starts with a **pretrained educational ML model** using AI Explorer's public Internship Selection Prediction Dataset, Kaggle version 1, MIT license. The source is explicitly **synthetic**. No observed real hiring outcomes are claimed.

Source: https://www.kaggle.com/datasets/aiexplorer77/internship-selection-prediction-dataset

## Bundled training and limitations

- 10,000 source records, 8,000 train and 2,000 untouched test records.
- Three-fold training-only selection compares Logistic Regression, Decision Tree, Random Forest and SVM by F1.
- The selected Random Forest achieved 53.85% accuracy and 0.6370 F1 on the test split. The majority-class baseline achieved 73.75% accuracy and 0.8489 F1. The app discloses that the model does not beat this baseline. These are limited educational results, not real-world hiring accuracy.
- Only CGPA, projects, certifications, aptitude, communication and interview scores are used. Coding-language names and internship months are absent from this source and are explicitly excluded rather than fabricated. They remain available for readiness guidance and user-uploaded datasets.
- Source aptitude, communication and interview values use 1–10. The prepared CSV multiplies them by ten for the app's 0–100 fields.
- Inputs outside the dataset's observed feature ranges return readiness guidance without a model prediction.
- Full source attribution, hashes, mapping, license and limitations: `data/provenance.json`, `data/README.md`, `data/LICENSE.txt`. Actual results: `data/public_report.json`.

Rebuild with `python train_public.py` after installing requirements. The script generates the bundled `data/public_model.joblib` artifact and report. Only this repository-owned artifact is loaded at startup. Model uploads are never accepted. The downloadable report includes the majority baseline and dataset provenance.

On restart or reset, the public model returns automatically. User-trained models stay in shared process memory only.

## Run

Install Python 3.11 or newer from python.org. Open a terminal in this project folder:

```sh
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python app.py
```

macOS / Linux:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

Open **http://127.0.0.1:8000**. Use Ctrl+C to stop. Set the `PORT` environment variable to change the port. The dashboard and demo guidance also run with `python app.py` without dependencies; real model training requires scikit-learn.

## First use

1. In Readiness studio, select **Try a sample**, then **Analyze my readiness**. These are illustrative profile values, not a labeled dataset.
2. Demo mode provides a preparation index and suggestions. It makes no ML selection prediction and displays no accuracy or selection probability.
3. In Model & dataset, download the empty CSV header template. Add authorized, de-identified records of actual historical outcomes.
4. Supply dataset provenance, confirm permission and observed labels, then train. The app compares four models and reports actual held-out results for the winner.
5. Return to the studio and analyze a profile again to receive a model classification, estimated probability, strengths, gaps, and model sensitivity checks.

The bundled synthetic public dataset and trained model power the educational mode. Separate tiny synthetic fixtures remain confined to software tests. Real hiring validation is still pending.

## CSV contract

Exact headers and order:

```csv
cgpa,coding_languages,projects,certifications,aptitude,communication,internship_experience,interview,selected
```

| Column | Meaning | Allowed range |
|---|---|---|
| cgpa | CGPA on a 10 point scale | 0–10 |
| coding_languages | Languages used, separated by `|` (e.g. `Python` or `Python&#124;Java`) | Supported names below, or blank for none |
| projects | Completed project count | integer 0–20 |
| certifications | Relevant completed certification count | integer 0–20 |
| aptitude | Aptitude assessment score | 0–100 |
| communication | Communication assessment score | 0–100 |
| internship_experience | Completed internship months | integer 0–24 |
| interview | Interview assessment score | 0–100 |
| selected | Observed outcome | exactly 0 or 1 |

Minimum 50 unique profiles and 15 records in each class. Maximum 10,000 rows and 2 MB request size (UI accepts files up to 1.9 MB to allow JSON overhead). Extra columns, invalid ranges, nonfinite numbers, missing labels and conflicting labels for identical profiles are rejected. Identical profiles with the same label are deduplicated. Blank numeric feature values are accepted in datasets. Every numeric feature must contain an observed value. A blank coding_languages cell explicitly means no languages selected. Training uses median imputation within each fold. Profile form fields cannot be blank.

Collect consistent assessments from a documented rubric. Do not upload names, emails, IDs, protected attributes, or confidential information. CSV is parsed as data; spreadsheets, scripts, pickle files and user-supplied models are not accepted. Formula-like or arbitrary text values fail numeric validation. No uploaded file is executed or written to disk.

## Model and evaluation

Stratified 80:20 train/test split, seed 42. Three-fold stratified cross-validation on the training portion selects the highest mean F1 from logistic regression, decision tree, random forest, and SVM. Median imputation and standardization are inside each pipeline, preventing preprocessing leakage. Class weighting addresses imbalance. The seven numeric inputs and a binary indicator for each supported coding language form the model inputs. Each indicator records whether that language is selected, without assigning proficiency scores or ranking languages. The chosen pipeline is fitted on the training split only; the test set is used once for final accuracy, precision, recall, F1, confusion matrix, and post-selection permutation analysis. Positive label is selected (1). Zero-denominator precision/recall/F1 are reported as zero.

The final model is intentionally not refitted on the held-out test rows, so the evaluated model and prediction model match. CV standard deviations describe fold variation, not confidence intervals. Probabilities are uncalibrated model estimates. The confusion matrix explicitly labels actual rows and predicted columns. Downloadable JSON records provenance, sample sizes, class balance, deduplication/missing counts, model comparison, metrics, and permutation importance.

Global permutation importance reports mean held-out F1 decrease over five shuffles. Local sensitivity replaces one input with its documented readiness benchmark. Neither is causal. Readiness strengths and recommendations are independent benchmark rules, not claims of learned feature attribution.

## Benchmark assumptions

The deck does not prescribe scales, cutoffs or a readiness formula. This implementation defines the scales above and benchmarks: CGPA 7.5; projects 3; certifications 2; aptitude 70; communication 70; experience 3 months; interview 70. The readiness index is the average of `min(value / benchmark, 1)` across the seven numeric fields, expressed as a percentage. A field meeting its benchmark is a strength; otherwise it receives a concrete practice recommendation. These are editable preparation assumptions, not validated hiring criteria. Readiness headings use index thresholds of 55 and 80.

Interview scores can be unavailable before hiring, and may leak later-stage information into an earlier-stage prediction. Only use this model at a point where all inputs would actually be known. For pre-interview deployment, develop and separately validate a model that excludes interview scores.

## Deck coverage

| Deck specification | Status |
|---|---|
| Slides 3–5: eight factors, binary prediction, strengths and recommendations | Implemented; prediction requires an uploaded real dataset |
| Slide 8: cleanup, missing values, duplicates, scaling, train/test split and class balance | Implemented with explicit numeric input schema, median imputation, deduplication, scaling, stratification and class weights |
| Slides 4 and 9: compare classification algorithms | Logistic regression, decision tree, random forest and SVM implemented |
| Slides 4 and 9: accuracy, precision, recall, F1 and confusion matrix | Implemented and shown only after training |
| Slides 5 and 9: influential features | Permutation importance and one-feature sensitivity implemented |
| XGBoost, categorical project types, automated feature selection, SMOTE | Not implemented; alternative techniques proposed by the deck |
| Real dataset collection, external validation and company-specific deployment | Future work |
| Accounts, database persistence and longitudinal tracking | Outside the deck's essential workflow; not implemented |

Literature review accuracies on slides 6–7 describe other studies and are not this application's results. The deck's paper claims have not been independently verified. This project does not reuse those values.

## Architecture and privacy

Vanilla HTML/CSS/JavaScript frontend; Python standard-library HTTP backend; scikit-learn ML. No Node build, CDN, API key or database required. Assets are local. User-uploaded model/data remain in process memory and disappear on restart or reset. The bundled public model loads again from the repository. Evaluation reports persist only if downloaded. `/api/status` provides the current model report; `/api/predict` accepts a profile; `/api/train` accepts JSON containing CSV text, provenance and consent; `/api/reset` clears the model. Prediction and training state swaps are synchronized. Failed training leaves the existing model intact.

The server binds to loopback, rejects cross-origin browser POSTs, limits request size and accepts only fixed static paths. It is intended for a trusted single-user local demonstration. No authentication, TLS, multi-user isolation, rate limiting, durable storage or production server is included. Do not expose it on a network. Production use needs authenticated job handling, limits, monitoring, bias review, retention controls and independent validation.

Historical hiring labels may reflect bias. A random split is not temporal or institutional validation. Models may not generalize across cohorts or companies. Students should use guidance to prepare; institutions must not use this unvalidated tool as an automated hiring gate.

## Tests

From the project directory after installing requirements:

```sh
python -m unittest discover -s tests -v
```

Tests cover demo truthfulness, input validation, malformed datasets, conflicting duplicates, and the full four-model training/evaluation/prediction path using a clearly marked synthetic software fixture. Synthetic test scores are not reported as real-world accuracy.

## Files

- `app.py`: backend, validation, model training and prediction.
- `static/`: responsive frontend and empty CSV header template.
- `tests/`: synthetic software-only verification.
- `requirements.txt`: training dependency.

Attribution: B. Likhitha, K. Sai Charan Reddy and K. Abhishek, Team 06, KG Reddy College of Engineering and Technology.

## Render demo deployment

Create a Python web service from this repository using:

- Build command: `pip install -r requirements.txt`
- Start command: `HOST=0.0.0.0 python app.py`
- Health check: `/api/status`
- Python version: set `PYTHON_VERSION=3.12.12`

Render supplies `PORT` and HTTPS. The application accepts same-host HTTPS requests.
This is a public demonstration with shared in-memory model state: visitors can
train or clear the same model, and its report is visible to all visitors. Use only
non-confidential demonstration data. Restarting the service clears its model.
Authenticated users, isolated model state, training job limits and a production
server are required before use with private datasets or institutional users.

## Coding-language input

Technical skills now uses language checkboxes rather than a numeric score. Supported choices are Python, Java, C, C++, JavaScript, TypeScript, C#, Go, Rust, Kotlin, Swift, PHP, Ruby, R and SQL. Select every language you have used, or leave them unchecked if you are starting out. Names are case-sensitive in CSV/API requests.

The profile API accepts `coding_languages` as a JSON array, for example `["Python", "Java"]`. CSV records use `Python|Java` in the `coding_languages` column. The downloadable template reflects this new schema. Old CSV files with a numeric `technical_skills` column must be updated with actual language names before training. Numeric technical scores cannot be converted to languages.

The readiness index now covers only seven numeric factors. Language count does not increase the index or imply proficiency. Model training encodes each language separately as present/absent and reports importance per language. Numeric what-if checks leave language selections unchanged. Models must be retrained using the updated dataset schema.
