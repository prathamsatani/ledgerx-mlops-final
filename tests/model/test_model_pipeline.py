import pytest
import pandas as pd
from src.model.train_model import load_data, extract_features, train_model

@pytest.fixture
def sample_data():
    """Creates a sample DataFrame for testing."""
    data = {'ocr_text': [
        'some text with numbers 10.5 and 20.2 and 30.7',
        'more text with numbers 5.1 and 15.3 and 25.8',
        'text with only two numbers 1.0 and 2.0',
        'text with no numbers'
    ]}
    return pd.DataFrame(data)

def test_load_data(tmp_path):
    """Tests the data loading function."""
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "hello.csv"
    p.write_text("a,b,c\n1,2,3")
    df = load_data(p)
    assert not df.empty

def test_extract_features(sample_data):
    """Tests the feature extraction function."""
    df = extract_features(sample_data)
    assert 'feature1' in df.columns
    assert 'feature2' in df.columns
    assert 'target' in df.columns
    assert df.loc[0, 'feature1'] == 10.5
    assert df.loc[1, 'feature2'] == 15.3
    assert df.loc[2, 'target'] == 0 # no third number
    assert df.loc[3, 'feature1'] == 0 # no numbers

def test_train_model(sample_data):
    """Tests the model training function."""
    df = extract_features(sample_data)
    features = ['feature1', 'feature2']
    target = 'target'

    model, X_test, y_test = train_model(df, features, target)

    assert model is not None
    assert not X_test.empty
    assert not y_test.empty
