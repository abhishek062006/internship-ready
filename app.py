"""Local internship readiness application. Run: python app.py"""
import csv
import io
import json
import math
import os
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = Path(__file__).parent
FEATURES = ['cgpa', 'projects', 'certifications', 'aptitude', 'communication', 'internship_experience', 'interview']
MAXIMA = [10, 20, 20, 100, 100, 24, 100]
TARGETS = [7.5, 3, 2, 70, 70, 3, 70]
LABELS = ['CGPA', 'Projects', 'Certifications', 'Aptitude', 'Communication', 'Internship experience', 'Interview']
ACTIONS = ['Review weak subjects and build a weekly study plan.', 'Build and deploy a project with a clear README and tests.', 'Complete a relevant certification and apply it in a project.', 'Practice timed numerical and logical reasoning sets.', 'Record a two-minute project explanation and ask for feedback.', 'Seek a short supervised project, volunteering role, or internship.', 'Practice a mock interview and prepare concrete project examples.']
LANGUAGES = ['Python', 'Java', 'C', 'C++', 'JavaScript', 'TypeScript', 'C#', 'Go', 'Rust', 'Kotlin', 'Swift', 'PHP', 'Ruby', 'R', 'SQL']
CSV_FIELDS = ['cgpa', 'coding_languages'] + FEATURES[1:]
MODEL_LABELS = LABELS + ['Language: ' + language for language in LANGUAGES]
STATE = None
LOCK = threading.Lock()

def coding_languages(data):
    if 'technical_skills' in data:
        raise ValueError('Choose coding_languages by name. The old technical_skills numeric score is no longer supported.')
    raw = data.get('coding_languages', [])
    if isinstance(raw, str):
        raw = [] if not raw.strip() else [v.strip() for v in raw.split('|')]
    if not isinstance(raw, list) or any(not isinstance(v, str) or v not in LANGUAGES for v in raw):
        raise ValueError('coding_languages: choose supported language names, not a numeric score.')
    return [language for language in LANGUAGES if language in raw]


def profile(data, missing=False):
    values = []
    for key, limit in zip(FEATURES, MAXIMA):
        raw = data.get(key)
        if missing and (raw is None or str(raw).strip() == ''):
            values.append(float('nan'))
            continue
        try:
            value = float(raw)
        except (ValueError, TypeError):
            raise ValueError(f'{key}: enter a number.')
        if not math.isfinite(value) or not 0 <= value <= limit:
            raise ValueError(f'{key}: expected 0 to {limit}.')
        if key in ('projects', 'certifications', 'internship_experience') and not value.is_integer():
            raise ValueError(f'{key}: expected a whole number.')
        values.append(value)
    languages = coding_languages(data)
    return values + [int(language in languages) for language in LANGUAGES]

def train(text, source, active_indices=None):
    # Dependencies are optional until real training is requested.
    try:
        import numpy as np
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        from sklearn.inspection import permutation_importance
    except ImportError:
        raise ValueError('Training needs scikit-learn. Run pip install -r requirements.txt and restart.')
    reader = csv.DictReader(io.StringIO(text.lstrip('\ufeff')))
    if reader.fieldnames != CSV_FIELDS + ['selected']:
        raise ValueError('CSV headers must exactly match the downloadable template, in order. No personal identifiers.')
    records, seen, removed = [], set(), 0
    for number, row in enumerate(reader, 2):
        if number > 10001:
            raise ValueError('Maximum 10,000 rows.')
        if None in row:
            raise ValueError(f'Row {number}: extra columns.')
        x = profile(row, missing=True)
        if row.get('selected') not in ('0', '1'):
            raise ValueError(f'Row {number}: selected must be 0 or 1.')
        key = tuple(None if math.isnan(v) else v for v in x)
        y = int(row['selected'])
        if key in seen:
            prior = next(r[1] for r in records if r[2] == key)
            if prior != y:
                raise ValueError(f'Row {number}: conflicting labels for identical features.')
            removed += 1
            continue
        seen.add(key)
        records.append((x, y, key))
    if len(records) < 50:
        raise ValueError('Provide at least 50 unique labeled profiles, with at least 15 per class.')
    X = np.array([r[0] for r in records]); y = np.array([r[1] for r in records])
    indices = list(range(len(MODEL_LABELS))) if active_indices is None else active_indices
    X = X[:, indices]
    model_labels = [MODEL_LABELS[i] for i in indices]
    if min(sum(y == 0), sum(y == 1)) < 15 or np.isnan(X).all(axis=0).any():
        raise ValueError('Need at least 15 profiles per class and observed values in every feature.')
    # Keep the test split untouched until selection by training-only cross validation.
    a, b, c, d = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    models = {'Logistic regression': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42), 'Decision tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, class_weight='balanced', random_state=42), 'Random forest': RandomForestClassifier(n_estimators=120, max_depth=7, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=1), 'SVM': SVC(probability=True, class_weight='balanced', random_state=42)}
    comparison = []; fitted = {}
    for name, model in models.items():
        pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), model)
        scores = cross_val_score(pipe, a, c, cv=StratifiedKFold(3, shuffle=True, random_state=42), scoring='f1', n_jobs=1)
        comparison.append({'name': name, 'cv_f1': float(scores.mean()), 'cv_std': float(scores.std())})
        fitted[name] = pipe
    comparison.sort(key=lambda r: r['cv_f1'], reverse=True)
    name = comparison[0]['name']; model = fitted[name].fit(a, c); pred = model.predict(b)
    imp = permutation_importance(model, b, d, scoring='f1', n_repeats=5, random_state=42)
    report = {'mode': 'trained', 'schema': 'coding-languages-v1', 'language_options': LANGUAGES, 'source': source[:120], 'model': name, 'rows': len(y), 'duplicates_removed': removed, 'missing_cells': int(np.isnan(X).sum()), 'train_rows': len(c), 'test_rows': len(d), 'class_counts': {'not_selected': int(sum(y == 0)), 'selected': int(sum(y == 1))}, 'metrics': {'accuracy': accuracy_score(d, pred), 'precision': precision_score(d, pred, zero_division=0), 'recall': recall_score(d, pred, zero_division=0), 'f1': f1_score(d, pred, zero_division=0)}, 'confusion_matrix': confusion_matrix(d, pred, labels=[0, 1]).tolist(), 'comparison': comparison, 'importance': [{'feature': label, 'value': float(value)} for label, value in zip(model_labels, imp.importances_mean)]}
    from sklearn.dummy import DummyClassifier
    baseline = DummyClassifier(strategy='most_frequent').fit(a, c).predict(b)
    report['baseline'] = {'name': 'Always predict the training majority class', 'accuracy': float(accuracy_score(d, baseline)), 'f1': float(f1_score(d, baseline, zero_division=0))}
    report['active_inputs'] = model_labels
    return {'model': model, 'report': report, 'active_indices': indices}

def predict(data):
    x = profile(data)
    with LOCK:
        state = STATE
    readiness = round(sum(min(v/t, 1) for v, t in zip(x, TARGETS)) / len(TARGETS) * 100)
    insights = [{'feature': label, 'value': v, 'benchmark': target, 'strength': v >= target, 'action': action} for label, v, target, action in zip(LABELS, x, TARGETS, ACTIONS)]
    result = {'mode': 'demo', 'readiness': readiness, 'insights': insights, 'prediction': None, 'probability': None, 'effects': [], 'coding_languages': coding_languages(data)}
    if state:
        indices = state.get('active_indices', list(range(len(x))))
        model_x = [x[i] for i in indices]
        meta = state['report']
        result.update(data_kind=meta.get('data_kind', 'user-supplied'), source=meta['source'], excluded_inputs=meta.get('excluded_inputs', []), model_warning=meta.get('baseline_comparison', ''))
        outside = [LABELS[i] for i, bounds in state.get('input_ranges', {}).items() if not bounds[0] <= x[i] <= bounds[1]]
        if outside:
            result.update(mode='unsupported', message='Outside the bundled dataset range: ' + ', '.join(outside) + '. Readiness guidance remains available.')
            return result
        m = state['model']; probability = float(m.predict_proba([model_x])[0][1])
        effects = []
        for i, target in enumerate(TARGETS):
            if i not in indices: continue
            changed = model_x.copy(); changed[indices.index(i)] = target
            effects.append({'feature': LABELS[i], 'change': float(m.predict_proba([changed])[0][1]) - probability})
        result.update(mode='trained', prediction='Likely to be Selected' if int(m.predict([model_x])[0]) else 'Not Selected', probability=probability, model=state['report']['model'], effects=effects)
    return result

class Handler(BaseHTTPRequestHandler):
    def reply(self, payload, status=200):
        body = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status); self.send_header('Content-Type', 'application/json'); self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/api/status':
            with LOCK:
                report = STATE['report'] if STATE else {'mode': 'demo'}
            return self.reply(report)
        if path == '/public-dataset.csv':
            body = (ROOT / 'data' / 'internship_public.csv').read_bytes()
            self.send_response(200); self.send_header('Content-Type', 'text/csv; charset=utf-8'); self.send_header('Content-Disposition', 'attachment; filename=internship_public.csv'); self.end_headers(); self.wfile.write(body)
            return
        public_files = {'/api/dataset': 'provenance.json', '/api/public-evaluation': 'public_report.json'}
        if path in public_files:
            return self.reply(json.loads((ROOT / 'data' / public_files[path]).read_text()))
        files = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'), '/style.css': ('style.css', 'text/css'), '/template.csv': ('template.csv', 'text/csv')}
        if path not in files:
            return self.reply({'error': 'Not found'}, 404)
        file, mime = files[path]; body = (ROOT / 'static' / file).read_bytes()
        self.send_response(200); self.send_header('Content-Type', mime+'; charset=utf-8'); self.send_header('X-Content-Type-Options', 'nosniff'); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        global STATE
        try:
            # Accept the same host over HTTP locally or HTTPS behind Render TLS.
            origin = self.headers.get('Origin')
            if origin and origin not in ('http://' + self.headers.get('Host', ''), 'https://' + self.headers.get('Host', '')):
                return self.reply({'error': 'Cross-origin request rejected'}, 403)
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 2_000_000:
                raise ValueError('Request must be between 1 byte and 2 MB.')
            data = json.loads(self.rfile.read(size))
            if self.path == '/api/predict':
                return self.reply(predict(data))
            if self.path == '/api/train':
                if data.get('consent') is not True:
                    raise ValueError('Confirm data permission and that labels are real observed outcomes.')
                if not LOCK.acquire(blocking=False):
                    return self.reply({'error': 'Training already in progress'}, 409)
                try:
                    new_state = train(data.get('csv', ''), str(data.get('source', 'Uploaded CSV')))
                    STATE = new_state
                finally:
                    LOCK.release()
                return self.reply(new_state['report'])
            if self.path == '/api/reset':
                with LOCK:
                    STATE = load_bundled_model()
                return self.reply(STATE['report'] if STATE else {'mode': 'demo'})
            return self.reply({'error': 'Not found'}, 404)
        except (ValueError, TypeError, KeyError, csv.Error) as exc:
            self.reply({'error': str(exc)}, 400)
        except Exception:
            self.reply({'error': 'Training failed. Check the dataset and server logs.'}, 500)

def load_bundled_model():
    # Only deserialize the model built by this repository. User model uploads are never accepted.
    import joblib
    path = ROOT / 'data' / 'public_model.joblib'
    return joblib.load(path) if path.exists() else None


if __name__ == '__main__':
    STATE = load_bundled_model()
    port = int(os.environ.get('PORT', 8000))
    host = os.environ.get('HOST', '127.0.0.1')
    print(f'Internship Ready: http://{host}:{port}', flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
