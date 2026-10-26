"""Synthetic fixtures test software only. They are not an internship dataset."""
import csv
import io
import random
import unittest
import app

def fixture():
    rng = random.Random(71)
    out = io.StringIO(); w = csv.writer(out)
    w.writerow(app.CSV_FIELDS + ['selected'])
    for i in range(100):
        w.writerow([round(rng.uniform(4,10),2),rng.choice(['Python|Java', 'C++', '', 'JavaScript|SQL']),rng.randint(0,6),rng.randint(0,4),rng.randint(0,100),rng.randint(0,100),rng.randint(0,12),rng.randint(0,100),i % 2])
    return out.getvalue()

class AppTests(unittest.TestCase):
    def setUp(self):
        app.STATE = None

    def test_demo_has_no_fabricated_prediction(self):
        result = app.predict(dict(zip(app.FEATURES, app.TARGETS)))
        self.assertIsNone(result['probability'])
        self.assertIsNone(result['prediction'])
        self.assertEqual(result['readiness'],100)

    def test_invalid_ranges_and_nonfinite(self):
        for value in ['NaN','inf',-1,11]:
            data = dict(zip(app.FEATURES,app.TARGETS)); data['cgpa']=value
            with self.assertRaises(ValueError): app.predict(data)

    def test_real_pipeline_with_synthetic_software_fixture(self):
        app.STATE = app.train(fixture(),'SYNTHETIC TEST FIXTURE, NOT REAL EVIDENCE')
        report=app.STATE['report']
        self.assertEqual(report['test_rows'],20)
        self.assertEqual(sum(map(sum,report['confusion_matrix'])),20)
        self.assertEqual(len(report['comparison']),4)
        prediction=app.predict(dict(zip(app.FEATURES,app.TARGETS)))
        self.assertEqual(prediction['mode'],'trained')
        self.assertTrue(0 <= prediction['probability'] <= 1)
        self.assertEqual(len(prediction['effects']),7)

    def test_languages_are_separate_inputs_not_scores(self):
        data = dict(zip(app.FEATURES, app.TARGETS))
        base = app.predict(data)
        data['coding_languages'] = ['Java', 'Python', 'Python']
        result = app.predict(data)
        self.assertEqual(result['coding_languages'], ['Python', 'Java'])
        self.assertEqual(result['readiness'], base['readiness'])
        values = app.profile(data)
        self.assertEqual(len(values), len(app.FEATURES) + len(app.LANGUAGES))
        self.assertEqual(sum(values[len(app.FEATURES):]), 2)
        data['coding_languages'] = ['Python', 'Klingon']
        with self.assertRaises(ValueError): app.predict(data)
        data['coding_languages'] = 75
        with self.assertRaises(ValueError): app.predict(data)

    def test_legacy_score_is_rejected(self):
        data = dict(zip(app.FEATURES, app.TARGETS))
        data['technical_skills'] = 75
        with self.assertRaisesRegex(ValueError, 'coding_languages'): app.predict(data)

    def test_bundled_model_uses_supported_inputs_only(self):
        app.STATE = app.load_bundled_model()
        report = app.STATE['report']
        self.assertEqual(report['data_kind'], 'synthetic')
        self.assertEqual(report['rows'], 10000)
        self.assertEqual(report['train_rows'] + report['test_rows'], 10000)
        self.assertEqual(len(report['importance']), 6)
        self.assertLess(report['metrics']['f1'], report['baseline']['f1'])
        data = dict(zip(app.FEATURES, app.TARGETS))
        first = app.predict(data)
        data['coding_languages'] = ['Python', 'Java']
        data['internship_experience'] = 24
        second = app.predict(data)
        self.assertEqual(first['probability'], second['probability'])
        self.assertEqual(second['data_kind'], 'synthetic')
        self.assertEqual(len(second['effects']), 6)
        data['cgpa'] = 4
        unsupported = app.predict(data)
        self.assertEqual(unsupported['mode'], 'unsupported')
        self.assertIsNone(unsupported['probability'])

    def test_bundled_dataset_provenance_hash(self):
        import hashlib, json
        meta = json.loads((app.ROOT / 'data/provenance.json').read_text())
        digest = hashlib.sha256((app.ROOT / 'data/internship_public.csv').read_bytes()).hexdigest()
        self.assertEqual(digest, meta['training_csv_sha256'])

    def test_reject_bad_dataset(self):
        with self.assertRaises(ValueError): app.train('name,selected\nSomeone,1','bad')
        with self.assertRaises(ValueError): app.train(','.join(app.CSV_FIELDS+['selected'])+'\n','empty')

    def test_conflicting_duplicates_rejected(self):
        data=fixture().splitlines()
        same=data[1].rsplit(',',1)[0]+',1'
        with self.assertRaisesRegex(ValueError,'conflicting'): app.train('\n'.join(data+[same]),'bad')

if __name__ == '__main__': unittest.main()
