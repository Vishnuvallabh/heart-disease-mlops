from preprocess import COLUMNS, clean, load_raw


def test_load_raw_reads_question_marks_as_missing(tmp_path):
    f = tmp_path / "raw.data"
    f.write_text("63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,?,6.0,0\n")
    df = load_raw(f)
    assert list(df.columns) == COLUMNS
    assert df["ca"].isna().sum() == 1


def test_clean_leaves_no_missing_values(raw_df):
    assert clean(raw_df).isna().sum().sum() == 0


def test_clean_makes_binary_target(raw_df):
    out = clean(raw_df)
    assert "num" not in out.columns
    assert set(out["target"].unique()) <= {0, 1}
    # num values 0,2,1,0,4,0 -> target 0,1,1,0,1,0
    assert out["target"].tolist() == [0, 0, 1, 0, 1, 0]


def test_clean_keeps_all_rows(raw_df):
    assert len(clean(raw_df)) == len(raw_df)


def test_clean_does_not_change_input(raw_df):
    clean(raw_df)
    assert "num" in raw_df.columns
