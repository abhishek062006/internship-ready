# Public training dataset

Source: AI Explorer, Internship Selection Prediction Dataset, version 1.
https://www.kaggle.com/datasets/aiexplorer77/internship-selection-prediction-dataset

The provider labels the source **synthetic**, for educational and research use, under the MIT license. It is not a collection of observed student hiring outcomes. See provenance.json for source hashes, transformations and omitted fields.

The bundled CSV keeps six matching inputs and the original selected label. Aptitude, communication and interview scores are multiplied by ten. Coding languages and internship duration are absent and are excluded from this bundled model. User datasets still support all app inputs.

Retrain reproducibly with `python train_public.py`. The model uses an 80:20 stratified split (seed 42), three-fold training-only model selection by F1, and a test split used once after selection. An always-selected baseline is reported.
