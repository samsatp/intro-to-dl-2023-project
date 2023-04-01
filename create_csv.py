from bs4 import BeautifulSoup
import glob, os, lxml, sys
from utils import preprocess_text
import zipfile
import os
import pandas as pd

def data_extract(contents):
    headlines = []
    texts = []
    labels = []


    for xml in contents:
        soup = BeautifulSoup(xml, features="xml")

        headlines.append(soup.headline.text)

        text = ' '.join([
            preprocess_text(t) 
            for t in soup.find('text').text.split('\n')
        ]).strip()
        texts.append(text)

        codes = soup.find_all("code")
        label = [code.attrs.get("code") for code in codes]
        labels.append(label)
    return headlines, texts, labels




import zipfile
import os
import pandas as pd

def generate_csv(data_path, out_data = 'data.csv'):
    contents = []
    for file in os.listdir(data_path):

        # Ignore special files
        if file == 'codes.zip' or file == 'dtds.zip':
            continue

        file = os.path.join(data_path, file)

        # Ignore other than zipfiles
        if not zipfile.is_zipfile(file):
            continue

        # Open the zipfile
        with zipfile.ZipFile(file, 'r') as zip_file: 
            for xml_file in zip_file.namelist():
            # Read the contents of every file in the archive
                with zip_file.open(xml_file) as f:
                    contents.append(f.read())
    headlines, texts, labels = data_extract(contents)
    unique_labels = set([item for label in labels for item in label])
    df = pd.DataFrame({'headline': headlines, 'text': texts, 'label': labels})
    df.to_csv(out_data, sep = '|')


if __name__ == "__main__":
    DATA_PATH = sys.argv[1]
    generate_csv(DATA_PATH)