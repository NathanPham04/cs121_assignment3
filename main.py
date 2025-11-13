# poop fart
import os
import json

from bs4 import BeautifulSoup
from collections import defaultdict


def main():
    inverted_index = defaultdict(list)
    filepaths = get_json_files(".")

    for doc_id, filepath in enumerate(filepaths):
        file_contents = parse_file(filepath)

        tokens = tokenize(file_contents)
        stems = [porter_stem(token) for token in tokens]

        update_index(inverted_index, doc_id, stems)

    # write to file
    with open("tmp.txt", "w") as file:
        for stem, docs in sorted(inverted_index.items()):
            docs.sort()

            file.write(f"{stem}:{docs}\n")
            print(f"{stem}: {docs}")

# Retrieves all .json filenames from the given directory
def get_json_files(dir: str) -> list[str]:
    json_files = []
    for root, _, files in os.walk(dir):
        for f in files:
            if f.endswith('.json'):
                json_files.append(os.path.join(root, f))
    return json_files


# Use beautifulsoup to parse files in a directory and return their text content
def parse_file(path: str) -> str:
    with open(path, "r") as f:
        data = json.load(f)

        url = data["url"]
        encoding = data["encoding"]
        content = data["content"]

        soup = BeautifulSoup(content, "lxml")

        # TODO: handle titles and headings differently

        # TODO: handle near and exact duplicates

        return soup.get_text()

# Tokenize text into lowercase alphanumeric tokens
def tokenize(text: str) -> list[str]:
    tokens = []
    current_token = []
    
    for line in text:
        for ch in line:
            if (ch >= 'a' and ch <= 'z') or (ch >= 'A' and ch <= 'Z') or (ch >= '0' and ch <= '9'):
                current_token.append(ch.lower())
            else:
                if current_token:
                    tokens.append(''.join(current_token))
                    current_token.clear()
    if current_token:
        tokens.append(''.join(current_token))
    return tokens


# ---------------------------------------Helpers for Porter Stemming----------------------------------------
def is_vowel(ch):
    return ch in "aeiou"

def contains_vowel(word):
    return any(is_vowel(c) for c in word)

def ends_with_double_consonant(word):
    return len(word) > 1 and word[-1] == word[-2] and not is_vowel(word[-1])
# ----------------------------------------------------------------------------------------------------------

def porter_stem(word):
    w = word.lower()

    # --- Step 1a ---
    if w.endswith("sses"):
        w = w[:-2]                      # stresses → stress
    elif w.endswith("ied") or w.endswith("ies"):
        if len(w) > 4:
            w = w[:-3] + "i"            # cries → cri, ties → tie
        else:
            w = w[:-3] + "ie"           # cried → crie
    elif w.endswith("ss") or w.endswith("us"):
        pass                            # stress → stress
    elif w.endswith("s"):
        stem = w[:-1]
        if contains_vowel(stem):
            w = stem                    # gaps → gap

    # --- Step 1b ---
    if w.endswith("eed") or w.endswith("eedly"):
        stem = w[:-3] if w.endswith("eed") else w[:-5]
        # Replace if there's a vowel before the last consonant cluster
        # (simplified: just always replace)
        w = stem + "ee"
    else:
        suffixes = ["ed", "edly", "ing", "ingly"]
        for suf in suffixes:
            if w.endswith(suf):
                stem = w[:-len(suf)]
                if contains_vowel(stem):
                    w = stem
                    # post-processing rules
                    if w.endswith(("at", "bl", "iz")):
                        w += "e"
                    elif ends_with_double_consonant(w) and w[-1] not in ("l", "s", "z"):
                        w = w[:-1]
                    elif len(w) <= 3:    # short word rule (approx)
                        w += "e"
                break

    return w


# Take in list of stemmed tokens and update inverted index
def update_index(inverted_index:dict, doc_id:int, stems:list[str]):
    freq_map = defaultdict(int)
    for stem in stems:
        freq_map[stem] += 1

    for stem, freq in freq_map.items():
        inverted_index[stem].append((doc_id, freq))


if __name__ == "__main__":
    main()
