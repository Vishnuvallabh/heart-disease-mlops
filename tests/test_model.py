from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from features import CATEGORICAL, FEATURES, NUMERIC, build_preprocessor
from predict import predict
from train import evaluate, get_models


def fitted_pipeline(df):
    pipe = Pipeline([("preprocess", build_preprocessor()),
                     ("model", LogisticRegression(max_iter=1000))])
    return pipe.fit(df[FEATURES], df["target"])


def test_preprocessor_scales_numeric_columns(clean_df):
    out = build_preprocessor().fit_transform(clean_df[FEATURES])
    scaled = out[:, :len(NUMERIC)]
    assert abs(scaled.mean()) < 1e-6          # mean ~0 after StandardScaler


def test_preprocessor_one_hot_encodes(clean_df):
    out = build_preprocessor().fit_transform(clean_df[FEATURES])
    n_categories = sum(clean_df[c].nunique() for c in CATEGORICAL)
    assert out.shape == (len(clean_df), len(NUMERIC) + n_categories + 3)


def test_two_models_defined():
    assert set(get_models()) == {"logistic_regression", "random_forest"}


def test_probabilities_are_valid(clean_df):
    prob = fitted_pipeline(clean_df).predict_proba(clean_df[FEATURES])
    assert ((prob >= 0) & (prob <= 1)).all()


def test_evaluate_returns_all_metrics(clean_df):
    model = fitted_pipeline(clean_df)
    _, _, metrics = evaluate(model, clean_df[FEATURES], clean_df["target"])
    assert set(metrics) == {"test_accuracy", "test_precision",
                            "test_recall", "test_roc_auc"}
    assert all(0 <= v <= 1 for v in metrics.values())


def test_predict_output_format(clean_df):
    model = fitted_pipeline(clean_df)
    patient = clean_df[FEATURES].iloc[0].to_dict()
    result = predict(patient, model, FEATURES)
    assert result["prediction"] in (0, 1)
    assert result["label"] in ("disease", "no disease")
    assert 0.5 <= result["confidence"] <= 1.0


def test_unknown_category_does_not_crash(clean_df):
    model = fitted_pipeline(clean_df)
    patient = clean_df[FEATURES].iloc[0].to_dict()
    patient["cp"] = 99                         # category never seen in training
    assert predict(patient, model, FEATURES)["prediction"] in (0, 1)
