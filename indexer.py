"""
This module contains the main indexing logic for processing documents,
building inverted indexes, handling duplicate detection, and outputting
partial indexes to disk.
"""

import os
import json
from bs4 import BeautifulSoup
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

seen_ngram_sets = []
inverted_index = defaultdict(list)
important_words_inverted_index = defaultdict(list)
link_graph = defaultdict(list)
hashed_seen_content_for_exact_duplicates = set()
num_documents_indexed = 0
exact_duplicates_skipped = 0
near_duplicates_skipped = 0

CORPUS = "DEV"
PARTIAL_INDEX_BODY_DIR = f"{CORPUS}_TEST/partial_indexes/"
PARTIAL_INDEX_IMPORTANT_DIR = f"{CORPUS}_TEST/partial_indexes_important_words/"
BATCH_SIZE = 10000  # Number of documents to process before checking index size and dumping to disk

def main():
    global inverted_index
    global important_words_inverted_index
    global link_graph

    document_id_map = {}
    url_to_doc_id = {}
    filepaths = get_json_files(f"./{CORPUS}")
    stemmer = PorterStemmer()
    global num_documents_indexed
    global hashed_seen_content_for_exact_duplicates
    global exact_duplicates_skipped
    global near_duplicates_skipped

    partial_num_index_body = 0
    partial_num_index_important = 0

    for doc_id, filepath in enumerate(filepaths):
        file_contents, important_words, url, links = parse_file(filepath)

        # Check for exact duplicates here by hashing all content
        hashed_content = hash(file_contents)
        if hashed_content in hashed_seen_content_for_exact_duplicates:
            exact_duplicates_skipped += 1
            continue
        else:
            hashed_seen_content_for_exact_duplicates.add(hashed_content)

        tokens = tokenize(file_contents)
        important_words_tokens = tokenize(important_words)

        # Porter stemming
        stems = [stemmer.stem(token) for token in tokens]
        important_stems = [stemmer.stem(token) for token in important_words_tokens]

        # Check for near duplicates here
        if len(stems) > 50 and similar_to_seen(stems):
            near_duplicates_skipped += 1
            continue

        # Global index update
        update_index(inverted_index, doc_id, stems)
        # Important words index update
        update_index(important_words_inverted_index, doc_id, important_stems)

        # Store links for PageRank
        link_graph[doc_id] = links

        # Check size of indexes and dump them if they have over 10,000 unique tokens
        if check_to_dump_index(inverted_index, BATCH_SIZE):
            partial_index_path = f"{PARTIAL_INDEX_BODY_DIR}full_index_part_{partial_num_index_body}.jsonl"
            output_full_index_to_file(partial_index_path)
            inverted_index = defaultdict(list)  # reset index
            partial_num_index_body += 1

        if check_to_dump_index(important_words_inverted_index, BATCH_SIZE):
            partial_important_index_path = f"{PARTIAL_INDEX_IMPORTANT_DIR}important_words_index_part_{partial_num_index_important}.jsonl"
            output_important_words_index_to_file(partial_important_index_path)
            important_words_inverted_index = defaultdict(list)  # reset index
            partial_num_index_important += 1

        document_id_map[doc_id] = url
        url_to_doc_id[url] = doc_id
        num_documents_indexed += 1

    if inverted_index:
        partial_index_path = f"{PARTIAL_INDEX_BODY_DIR}full_index_part_{partial_num_index_body}.jsonl"
        output_full_index_to_file(partial_index_path)
    
    if important_words_inverted_index:
        partial_important_index_path = f"{PARTIAL_INDEX_IMPORTANT_DIR}important_words_index_part_{partial_num_index_important}.jsonl"
        output_important_words_index_to_file(partial_important_index_path)

    # ------------------------------DO NOT UNCOMMENT (OLD CODE BEFORE PARTIAL INDEXING)-------------------------------

    # write to file the full inverted index with sorted lists
    # output_full_index_to_file("ANALYST_test/full_index.jsonl")

    # write to file the important words inverted index with sorted lists
    # output_important_words_index_to_file("ANALYST_test/important_words_index.jsonl")

    # ------------------------------DO NOT UNCOMMENT (OLD CODE BEFORE PARTIAL INDEXING)-------------------------------

    # write to file the document id map
    output_doc_id_map_to_file(f"{CORPUS}_TEST/doc_id_map.jsonl", document_id_map)

    # Convert URLs to doc_ids in link graph and save
    output_link_graph_to_file(f"{CORPUS}_TEST/link_graph.json", url_to_doc_id)




# Retrieves all .json filenames from the given directory
def get_json_files(dir: str) -> list[str]:
    json_files = []
    for root, _, files in os.walk(dir):
        for f in files:
            if f.endswith('.json'):
                json_files.append(os.path.join(root, f))
    return json_files


# Use beautifulsoup to parse files in a directory and return their text content
def parse_file(path: str) -> tuple[str, str, str, list[str]]:
    with open(path, "r") as f:
        data = json.load(f)

        url = data["url"]
        encoding = data["encoding"]
        content = data["content"]

        soup = BeautifulSoup(content, "lxml")

        # Important words are in their own index
        important_words = " ".join(
            tag.get_text()
            for tag in soup.find_all(['h1', 'h2', 'h3', 'strong', 'title'])
        )

        # Extract all links for PageRank
        links = [a.get('href') for a in soup.find_all('a', href=True)]

        return (soup.get_text(), important_words, url, links)

# Tokenize text using NLTK
def tokenize(text: str) -> list[str]:
    tokens = word_tokenize(text.lower())
    return [token for token in tokens if token.isalnum()]

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

def report_output():
    # Write out the report for assignment 1
    with open("assignment3_report.txt", "w") as report:
        report.write(f"Number of documents indexed: {num_documents_indexed}\n")
        report.write(f"Number of unique tokens: {len(inverted_index)}\n")
        report.write(f"Size of index on disk: {get_index_size_on_disk_in_kb('full_index.json')} KB\n")

        report.write("\n\nAdditional Statistics:\n")
        report.write(f"Number of exact duplicate documents skipped: {exact_duplicates_skipped}\n")
        report.write(f"Number of near duplicate documents skipped: {near_duplicates_skipped}\n")

def check_to_dump_index(inverted_index: dict, threshold: int = 10000) -> bool:
    return len(inverted_index) >= threshold

def output_full_index_to_file(path: str):
    # write to file the full inverted index with sorted lists
    ensure_dir(path)
    with open(path, "w") as file:
        for stem in inverted_index:
            inverted_index[stem].sort()
        
        for stem, postings in sorted(inverted_index.items()):
            line = json.dumps({stem: postings})
            file.write(line + "\n")

def output_important_words_index_to_file(path: str):
    # write to file the important words inverted index with sorted lists
    ensure_dir(path)
    with open(path, "w") as file:
        for stem in important_words_inverted_index:
            important_words_inverted_index[stem].sort()
        
        for stem, postings in sorted(important_words_inverted_index.items()):
            line = json.dumps({stem: postings})
            file.write(line + "\n")

def output_doc_id_map_to_file(path: str, document_id_map: dict):
    ensure_dir(path)
    with open(path, "w") as file:
        for doc_id, url in sorted(document_id_map.items()):
            line = json.dumps({doc_id: url})
            file.write(line + "\n")

def output_link_graph_to_file(path: str, url_to_doc_id: dict):
    global link_graph
    ensure_dir(path)
    # Convert URL links to doc_id links
    doc_id_graph = {}
    for doc_id, url_links in link_graph.items():
        doc_id_links = [url_to_doc_id[url] for url in url_links if url in url_to_doc_id]
        doc_id_graph[doc_id] = doc_id_links
    
    with open(path, "w") as file:
        json.dump(doc_id_graph, file)

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

if __name__ == "__main__":
    main()
