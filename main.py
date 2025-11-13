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

# Use beautifulsoup to parse files in a directory and return their text content
def parse_files(path:str):
    pass

# Time: O(n) where n = number of characters in file (single pass).
# Space: O(t) to store all tokens in a list (t = number of tokens).
def tokenize(text: str) -> list:
    """
    Reads a text file and returns a list of lowercase alphanumeric tokens.  
    Non-alphanumeric characters act as delimiters.
    """
    tokens = []
    current_token = []
    
    for line in text:
        for ch in line:
            if ch.isalnum():
                current_token.append(ch.lower())
            else:
                if current_token:
                    tokens.append(''.join(current_token))
                    current_token.clear()
    if current_token:
        tokens.append(''.join(current_token))
    return tokens

# Use porter stemming
def stem(token:str):  
    pass

# Take in list of stemmed tokens and update inverted index
def update_index(inverted_index:dict, doc_id:int, stems:list[str]):
    for stem in stems:
        inverted_index[stem].append(doc_id)


if __name__ == "__main__":
    main()
