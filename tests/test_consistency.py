from tests.KDEClassifier_orig import kde as KDEClassifier_orig
from kde_classifier.KDEClassifier import KDEClassifier as KDEClassifier_new
import numpy as np
import pytest

def make_synthetic(seed=0, n_train=120, n_test=30, n_features=2):
    rng = np.random.RandomState(seed)
    X0 = rng.normal(loc=-1.0, scale=0.5, size=(n_train // 2, n_features))
    X1 = rng.normal(loc=1.0, scale=0.5, size=(n_train // 2, n_features))
    X_train = np.vstack([X0, X1])
    y_train = np.array([0] * (n_train // 2) + [1] * (n_train // 2))

    Xt0 = rng.normal(loc=-1.0, scale=0.5, size=(n_test // 2, n_features))
    Xt1 = rng.normal(loc=1.0, scale=0.5, size=(n_test // 2, n_features))
    X_test = np.vstack([Xt0, Xt1])
    y_test = np.array([0] * (n_test // 2) + [1] * (n_test // 2))

    return X_train, y_train, X_test, y_test


@pytest.mark.parametrize("bandwidth", [0.7, 1.0])
def test_predict_proba_and_predict_equal(bandwidth):
    X_train, y_train, X_test, y_test = make_synthetic(seed=0)

    orig = KDEClassifier_orig(mybandwidth=bandwidth)
    r = orig.fit(X_train, y_train)

    new = KDEClassifier_new(mybandwidth=bandwidth)
    n = new.fit(X_train, y_train)

    p_orig = orig.predict_proba(X_test)
    p_new = new.predict_proba(X_test)

    # Align columns in case label ordering differs
    labels_orig = orig.trainlabels_set
    labels_new = new.trainlabels_set
    if labels_orig != labels_new:
        idx_map = [labels_new.index(lbl) for lbl in labels_orig]
        p_new = p_new[:, idx_map]

    assert p_orig.shape == p_new.shape
    np.testing.assert_allclose(p_orig, p_new, rtol=1e-6, atol=1e-8)

    preds_orig = orig.predict(X_test)
    preds_new = new.predict(X_test)
    assert np.array_equal(preds_orig, preds_new)


def test_single_sample_class():
    # one class has a single sample — both implementations should align
    rng = np.random.RandomState(1)
    X0 = rng.normal(-1, 0.5, size=(59, 2))
    X1 = rng.normal(1, 0.5, size=(60, 2))
    X2 = np.array([[0.0, 0.0]])
    X_train = np.vstack([X0, X1, X2])
    y_train = np.array([0]*59 + [1]*60 + [2])

    X_test = np.vstack([rng.normal(-1,0.5,size=(5,2)), rng.normal(1,0.5,size=(5,2)), np.array([[0.0,0.0]])])

    orig = KDEClassifier_orig(mybandwidth=0.7)
    orig.fit(X_train, y_train)
    new = KDEClassifier_new(mybandwidth=0.7)
    new.fit(X_train, y_train)

    p_orig = orig.predict_proba(X_test)
    p_new = new.predict_proba(X_test)

    labels_orig = orig.trainlabels_set
    labels_new = new.trainlabels_set
    if labels_orig != labels_new:
        idx_map = [labels_new.index(lbl) for lbl in labels_orig]
        p_new = p_new[:, idx_map]

    np.testing.assert_allclose(p_orig, p_new, rtol=1e-6, atol=1e-8)
