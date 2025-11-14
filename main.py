import os
import json
from bs4 import BeautifulSoup
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

seen_documents = []
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

hashed_seen_content_for_exact_duplicates = set()
num_documents_indexed = 0
exact_duplicates_skipped = 0
near_duplicates_skipped = 0

def main():
    inverted_index = defaultdict(list)
    document_id_map = {}
    filepaths = get_json_files("./ANALYST")
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
    with open("assignment3_report.txt", "w") as report:
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


def similar_to_seen(text: list[str], threshold: float = 0.85):
    global seen_documents, tfidf_vectorizer
    
    current_text = " ".join(text)
    
    if not seen_documents:
        seen_documents.append(current_text)
        return False
    
    # Create corpus with current text and all seen documents
    corpus = seen_documents + [current_text]
    
    # Compute TF-IDF vectors
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    
    # Get similarity between current document and all previous documents
    current_vector = tfidf_matrix[-1]
    previous_vectors = tfidf_matrix[:-1]
    
    similarities = cosine_similarity(current_vector, previous_vectors).flatten()
    
    # Check if any similarity exceeds threshold
    if np.max(similarities) > threshold:
        return True
    
    seen_documents.append(current_text)
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
