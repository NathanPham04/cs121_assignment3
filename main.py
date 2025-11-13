# poop fart
from typing import DefaultDict


def main():
    inverted_index = DefaultDict(list)
    file_texts = parse_files(".")

    for doc_id, file in enumerate(file_texts):
        tokens = tokenize(file)
        stems = [stem(token) for token in tokens]

        update_index(inverted_index, doc_id, stems)

    # write to file
    with open("tmp.txt", "w") as file:
        for stem, docs in sorted(inverted_index.items()):
            docs.sort()

            file.write(f"{stem}:{docs}\n")
            print(f"{stem}: {docs}")

def parse_files(path:str) -> list[str]:
    pass

def tokenize(text:str):
    # simple whitespace tokenizer
    return text.split()

def stem(token:str):  # use porter stemming
    pass

def update_index(inverted_index:dict, doc_id:int, stems:list[str]):
    for stem in stems:
        inverted_index[stem].append(doc_id)


if __name__ == "__main__":
    main()
