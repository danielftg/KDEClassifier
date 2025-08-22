# KDEClassifier

A pip-installable distribution of KDEClassifier introduced by:  
Yang Liu, Hongxia He, Zhi-Xiong Xiao, Anquan Ji, Jian Ye, Qifan Sun, Yang Cao.  
See: [https://pubmed.ncbi.nlm.nih.gov/33313714/](https://pubmed.ncbi.nlm.nih.gov/33313714/)

Code adapted from: [http://cao.labshare.cn/KDEClassifier/index.html](http://cao.labshare.cn/KDEClassifier/index.html)

This repository provides two implementations:

* **`kde_classifier.KDEClassifier`** — An enhanced version of the original implementation; vectorized and sklearn-compatible.
* **`tests/KDEClassifier_orig.py`** — the original reference implementation (kept for comparison and tests).

---


## Installation
```bash
pip install kde-classifier
```

## Usage example

```python
from kde_classifier import KDEClassifier

clf = KDEClassifier(mybandwidth=0.7)
clf.fit(X_train, y_train)
probs = clf.predict_proba(X_test)
preds = clf.predict(X_test)
```

`predict_proba` always returns a 2-D array shaped `(n_samples, n_classes)` (even when `n_samples==0`).

---


## Tests

A pytest test file is provided under `tests/` that compares the original implementation with the new one on synthetic datasets. Run tests with:

```bash
pytest -q
```

---

