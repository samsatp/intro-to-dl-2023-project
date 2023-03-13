from bs4 import BeautifulSoup
import glob, os, re

from typing import List

def preprocess_text(x: str):
    x = x.strip()

    # Non-alphabet normalization
    x = re.sub(r"\\[tn]|\W", " ", x)

    # Whitespace normalization
    x = re.sub(r"\s+", " ", x)

    return x

def parse_xml(files: List[os.PathLike]):
    """
        Parameters
        ---
        `data_path`: a list of all files to use

        Returns
        ---
        A tuple of

        - `headlines`: a list of headlines
        - `texts`: a list of texts
        - `labels`: a list of list of labels

        in a corresponding order
    """

    headlines = []
    texts = []
    labels = []

    for f in files:
        with open(f, "r") as s:
            xml = s.read()
        soup = BeautifulSoup(xml, features="xml")

        # Extract headline
        headlines.append(soup.headline.text)

        # Extract text
        text = ' '.join([
            preprocess_text(t) 
            for t in soup.find('text').text.split('\n')
        ]).strip()
        texts.append(text)

        # Extract code
        codes = soup.find_all("code")
        label = [code.attrs.get("code") for code in codes]
        labels.append(label)