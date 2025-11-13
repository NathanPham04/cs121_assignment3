# poop fart
import json

from bs4 import BeautifulSoup
from typing import DefaultDict


def main():
    inverted_index = DefaultDict(list)
    filepaths = get_json_files(".")

    for doc_id, filepath in enumerate(filepaths):
        file_contents = parse_file(filepath)

        tokens = tokenize(file_contents)
        stems = [stem(token) for token in tokens]

        update_index(inverted_index, doc_id, stems)

    # write to file
    with open("tmp.txt", "w") as file:
        for stem, docs in sorted(inverted_index.items()):
            docs.sort()

            file.write(f"{stem}:{docs}\n")
            print(f"{stem}: {docs}")

# Retrieves all .json filenames from the given directory
def get_json_files(dir: str) -> list[str]:
    pass

# Use beautifulsoup to parse files in a directory and return their text content
def parse_file(path: list[str]) -> str:
    with open(path, "r") as f:
        data = json.load(f)

        url = data["url"]
        encoding = data["encoding"]
        content = data["content"]

        soup = BeautifulSoup(content, "html.parser")

        return soup.get_text()

# Take text and return list of tokens
def tokenize(text:str):
    # simple whitespace tokenizer
    return text.split()

# Use porter stemming
def stem(token:str):  
    pass

# Take in list of stemmed tokens and update inverted index
def update_index(inverted_index:dict, doc_id:int, stems:list[str]):
    for stem in stems:
        inverted_index[stem].append(doc_id)


if __name__ == "__main__":
    main()
