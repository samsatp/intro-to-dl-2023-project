from bs4 import BeautifulSoup
import glob, os, re, argparse

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


## TODO: Get headlines, texts, labels from the saved text files
def get_data_from_text_files(headlines_file, texts_file, labels_file):

    with open(headlines_file, "r") as f:
        headlines = f.readlines()

    with open(texts_file, "r") as f:
        texts = f.readlines()

    with open(labels_file, "r") as f:
        labels = f.readlines()
    
    assert len(headlines) == len(texts)

    X = [headline + " " + text for headline, text in zip(headlines, texts)]
    y = [label.split("\t") for label in labels]

    return X, y

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="utility")
    parser.add_argument("-c", "--command", choices = ["parseXml"], required=True)
    args = parser.parse_args()
    command = args.command
    print("command:", command)
    
    if command == "parseXml":
        data_path = os.path.join("data","*.xml")
        files = glob.glob(data_path)
        headlines, texts, labels = parse_xml(files)
    
        with open("headlines.txt", "w") as f:
            f.writelines([headline + "\n" for headline in headlines])

        with open("texts.txt", "w") as f:
            f.writelines([text+"\n" for text in texts])
    
        with open("labels.txt", "w") as f:
            f.writelines(['\t'.join(e)+"\n" for e in labels])

