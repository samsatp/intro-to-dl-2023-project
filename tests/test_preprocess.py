import pytest
import utils
import glob
import os


def test_parse_xml():
    files = glob.glob(os.path.join("data","sample","*"))
    headlines, texts, labels = utils.parse_xml(files)

    assert len(headlines) == len(texts)
    assert len(texts) == len(labels)

    for i in range(len(labels)):
        assert isinstance(headlines[i], str)
        assert isinstance(texts[i], str)
        assert all([isinstance(label, str) for label in labels[i]])