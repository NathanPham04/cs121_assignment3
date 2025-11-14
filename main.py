# poop fart
import os
import json

from bs4 import BeautifulSoup
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

seen_ngram_sets = []

hashed_seen_content_for_exact_duplicates = set()
num_documents_indexed = 0
exact_duplicates_skipped = 0
near_duplicates_skipped = 0

def main():
    inverted_index = defaultdict(list)
    document_id_map = {}
    filepaths = get_json_files("./DEV")
    stemmer = PorterStemmer()
    global num_documents_indexed
    global hashed_seen_content_for_exact_duplicates
    global exact_duplicates_skipped
    global near_duplicates_skipped

    for doc_id, filepath in enumerate(filepaths):
        file_contents = parse_file(filepath)

        # Check for exact duplicates here by hashing all content
        hashed_content = hash(file_contents)
        if hashed_content in hashed_seen_content_for_exact_duplicates:
            exact_duplicates_skipped += 1
            continue
        else:
            hashed_seen_content_for_exact_duplicates.add(hashed_content)

        tokens = tokenize(file_contents)

        # Porter stemming
        stems = [stemmer.stem(token) for token in tokens]

        # Check for near duplicates here
        if len(stems) > 50 and similar_to_seen(stems):
            near_duplicates_skipped += 1
            continue

        update_index(inverted_index, doc_id, stems)
        document_id_map[doc_id] = filepath
        num_documents_indexed += 1

    # write to file the index
    with open("index.txt", "w") as file:
        for stem, docs in sorted(inverted_index.items()):
            docs.sort()

            file.write(f"{stem}:{docs}\n")

    # write to file the document id map
    with open("doc_id_map.json", "w") as file:
        json.dump(document_id_map, file, indent=4)

    # Write out the report for assignment 1
    with open("assignment1_report.txt", "w") as report:
        report.write(f"Number of documents indexed: {num_documents_indexed}\n")
        report.write(f"Number of unique tokens: {len(inverted_index)}\n")
        report.write(f"Size of index on disk: {get_index_size_on_disk_in_kb('index.txt')} KB\n")

        report.write("\n\nAdditional Statistics:\n")
        report.write(f"Number of exact duplicate documents skipped: {exact_duplicates_skipped}\n")
        report.write(f"Number of near duplicate documents skipped: {near_duplicates_skipped}\n")



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

        return soup.get_text()

# Tokenize text using NLTK
def tokenize(text: str) -> list[str]:
    tokens = word_tokenize(text.lower())
    return [token for token in tokens if token.isalnum()]


# ---------------------------------------Jaccard Similarity----------------------------------------

def jaccard_similarity(set1, set2):
        """
        adapted from https://www.geeksforgeeks.org/data-science/how-to-calculate-jaccard-similarity-in-python/
        """
        union = set1.union(set2)
        if not union:
            return 0.0
        intersection = set1.intersection(set2)
        return len(intersection) / len(union)
        
# https://medium.com/data-science/text-analysis-basics-in-python-443282942ec5
def similar_to_seen(text: list[str], threshold:float=0.90):
    global seen_ngram_sets

    # helper for making n-grams
    def make_ngrams(word_list, n=7):
        return set(" ".join(word_list[i:i+n]) for i in range(len(word_list) - n + 1))
    
    current_ngrams = make_ngrams(text)

     # Compare against previously seen pages
    for prior_ngrams in seen_ngram_sets:
        similarity_score = jaccard_similarity(current_ngrams, prior_ngrams)
        if similarity_score > threshold:  # similarity threshold
            return True  

    # otherwise, store this page’s n-gram set for future comparisons
    seen_ngram_sets.append(current_ngrams)
    return False

# Take in list of stemmed tokens and update inverted index
def update_index(inverted_index:dict, doc_id:int, stems:list[str]):
    freq_map = defaultdict(int)
    for stem in stems:
        freq_map[stem] += 1

    for stem, freq in freq_map.items():
        inverted_index[stem].append((doc_id, freq))

def get_index_size_on_disk_in_kb(index_filepath: str) -> int:
    return os.path.getsize(index_filepath) // 1024

if __name__ == "__main__":
    main()
