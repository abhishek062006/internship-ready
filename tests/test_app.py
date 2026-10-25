"""Synthetic fixtures test software only. They are not an internship dataset."""
import csv
import io
import random
import unittest
import app

def fixture():
    rng = random.Random(71)
    out = io.StringIO(); w = csv.writer(out)
    w.writerow(app.FEATURES + ['selected'])
    for i in range(100):
        w.writerow([round(rng.uniform(4,10),2),rng.randint(0,100),rng.randint(0,6),rng.randint(0,4),rng.randint(0,100),rng.randint(0,100),rng.randint(0,12),rng.randint(0,100),i % 2])
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
        self.assertEqual(len(prediction['effects']),8)

    def test_reject_bad_dataset(self):
        with self.assertRaises(ValueError): app.train('name,selected\nSomeone,1','bad')
        with self.assertRaises(ValueError): app.train(','.join(app.FEATURES+['selected'])+'\n','empty')

    def test_conflicting_duplicates_rejected(self):
        data=fixture().splitlines()
        same=data[1].rsplit(',',1)[0]+',1'
        with self.assertRaisesRegex(ValueError,'conflicting'): app.train('\n'.join(data+[same]),'bad')

if __name__ == '__main__': unittest.main()
