import pytest
import utils
import glob
import os

@pytest.fixture
def get_data():
    files = glob.glob(os.path.join("data","sample","*"))
    headlines, texts, labels = utils.parse_xml(files)
    return headlines, texts, labels

def test_parse_xml(get_data):
    headlines, texts, labels = get_data

    assert len(headlines) == len(texts)
    assert len(texts) == len(labels)

    for i in range(len(labels)):
        assert isinstance(headlines[i], str)
        assert isinstance(texts[i], str)
        assert all([isinstance(label, str) for label in labels[i]])