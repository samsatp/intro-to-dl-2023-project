from bs4 import BeautifulSoup
import glob, os, re, json
import pandas as pd
from typing import List

def preprocess_text(x: str):
    x = x.strip()

    # Non-alphabet normalization
    x = re.sub(r"\\[tn]|\W", " ", x)

    # Whitespace normalization
    x = re.sub(r"\s+", " ", x)

    return x

def preprocess_text_series(text: pd.Series):
    text = text.str.lower()
    text = text.str.strip()
    return text

def parse_xml(files: List[os.PathLike]):
    """
        Parameters
        ---
        `data_path`: a list of XML file paths

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

    for i, f in enumerate(files):
        if i%1000 == 0:
            print(f"\treading file: {i}")
        with open(f, "r", encoding="ISO-8859-1") as s:
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
        
    return headlines, texts, labels

def get_data(file, nrows=None):
    # Load headlines, texts and labels
    df = pd.read_csv(file, sep = '|', nrows=nrows)
    if "headline" in df.columns:
        df["headline"].fillna("", inplace=True)
        df["text"].fillna("", inplace=True)
        data = preprocess_text_series(df["headline"]) + " " + preprocess_text_series(df["text"])
    else:
        data = preprocess_text_series(df["text"])

    labels = None
    if 'label' in df.columns:
        labels = df['label'].values
        labels = [json.loads(item.replace("'", "\"")) for item in labels]

    return data, labels