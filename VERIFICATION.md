# Verification

Verified on the build host with Python 3.12 and scikit-learn 1.9.1:

- Five backend tests passed, including four-classifier training, cross-validation, held-out metrics, confusion matrix counts, probability output, local sensitivity, demo behavior, numeric validation and conflicting duplicate rejection.
- JavaScript syntax check passed (`node --check static/app.js`).
- Local server started successfully on loopback port 8000.

The test training data is synthetic and is used only to verify software. No scores from these fixtures are presented as internship selection accuracy. No labeled internship dataset or trained model ships with the project.

Browser preview access was declined by the browser approval gate. Visual layout, interactive browser flows, and responsive behavior have not been verified in a browser. Frontend code includes responsive styles and accessible form labels, but these are implementation details rather than evidence of a completed visual inspection.

scikit-learn 1.9.1 emits a forward-looking deprecation warning for SVC's probability option. It remains supported in the pinned version. Upgrade to a future scikit-learn version only after replacing this option with an explicit calibration wrapper and rerunning tests.
