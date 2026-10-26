"""Train the bundled educational model from the attributed public synthetic CSV."""
import csv, io, json, hashlib
from pathlib import Path
import joblib
import app

ROOT = Path(__file__).parent

def build():
    path = ROOT / 'data' / 'internship_public.csv'
    rows = list(csv.DictReader(path.open()))
    out = io.StringIO(); writer = csv.DictWriter(out, fieldnames=app.CSV_FIELDS + ['selected'])
    writer.writeheader()
    for row in rows:
        # Missing source fields remain missing and are explicitly excluded from training.
        writer.writerow({**row, 'coding_languages': '', 'internship_experience': ''})
    indices = [0, 1, 2, 3, 4, 6]
    state = app.train(out.getvalue(), 'AI Explorer / Kaggle internship dataset v1 (synthetic)', active_indices=indices)
    meta = json.loads((ROOT / 'data' / 'provenance.json').read_text())
    state['input_ranges'] = {i: [min(float(r[app.FEATURES[i]]) for r in rows), max(float(r[app.FEATURES[i]]) for r in rows)] for i in indices}
    report = state['report']
    report.update(data_kind='synthetic', dataset_url=meta['url'], dataset_author=meta['author'], license=meta['license'], dataset_version=1, training_csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), excluded_inputs=meta['excluded_inputs'], limitations=meta['limitations'], schema='public-six-inputs-v1', input_ranges={app.FEATURES[i]: bounds for i, bounds in state['input_ranges'].items()})
    report['baseline_comparison'] = 'Model F1 exceeds the majority baseline.' if report['metrics']['f1'] > report['baseline']['f1'] else 'Model F1 does not exceed the majority baseline. Treat predictions as a limited educational experiment.'
    joblib.dump(state, ROOT / 'data' / 'public_model.joblib', compress=3)
    (ROOT / 'data' / 'public_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:report[k] for k in ['rows','train_rows','test_rows','class_counts','model','metrics','baseline','baseline_comparison']},indent=2))

if __name__ == '__main__': build()
