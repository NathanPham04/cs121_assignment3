"""
This module contains functions to perform search queries on the indexed documents
"""

from nltk.stem import PorterStemmer
import json
import math
from collections import defaultdict
from indexer import CORPUS, tokenize
import os
import pickle
from merged_index_splitter import SECONDARY_BODY_INDEX_PATH, SECONDARY_IMPORTANT_INDEX_PATH, SPLIT_OUTPUT_BODY_DIR as BODY_INDEX_DIR, SPLIT_OUTPUT_IMPORTANT_DIR as IMPORTANT_INDEX_DIR
import time


CORPUS_SIZE = 44845
SECONDARY_INDEX_BODY = list()
SECONDARY_INDEX_IMPORTANT = list()
DOC_MAP = dict()

def search(query_text, top_k=5):
    tokenized_query = tokenize(query_text)
    stemmer = PorterStemmer()
    stemmed_query = [stemmer.stem(token) for token in tokenized_query]
    sorted_postings, all_terms_found = get_postings(stemmed_query)
    if not all_terms_found:
        return []
    intersected_postings = boolean_AND_search(sorted_postings)
    sorted_doc_scores = sorted(score_query(intersected_postings).items(), key=lambda x: x[1], reverse=True)
    return [(DOC_MAP[str(doc_id)], score) for doc_id, score in sorted_doc_scores[:top_k]]

def search_query():
    # External data structure setup
    setup_search_environment()

    query = input("Enter your search query: ")
    results = search(query)
    if not results:
        print("No documents found matching the query for boolean retrieval.")
        return
    for url, score in results:
        print(f"Document: {url}, Score: {score}")

# Get a list of all the postings for each term and remove duplicates and sort them by length
def get_postings(stemmed_query: list[str]) -> tuple[list[tuple[str, list[tuple[int, int]]]], bool]:
    postings = []
    stemmed_set = set(stemmed_query)
    all_terms_found = True

    for term in stemmed_set:
        # term_postings = get_postings_from_full_index(term, inverted_index)
        term_postings = search_partial_index_for_term(SECONDARY_INDEX_BODY, BODY_INDEX_DIR, term)
        if not term_postings:
            all_terms_found = False
        postings.append((term, term_postings))

    sorted_postings = sorted(postings, key=lambda x: len(x[1]))
    return sorted_postings, all_terms_found

# Performs a boolean AND search on the sorted postings lists and returns the dict of term: list of (doc_id, tf-idf)
# Input: [(term, [(doct_id, tf-idf), ...]), ...]
# Output: {term: [(doc_id, tf-idf), ...]}
def boolean_AND_search(sorted_postings: list[tuple[str, list[tuple[int, int]]]]) -> list[tuple[str, list[tuple[int, int]]]]:
    if not sorted_postings:
        return []

    # Get intersect of all documents in postings
    documents = intersect_documents(sorted_postings)

    # Create new list with only intersected
    intersect_postings = filter_postings(sorted_postings, documents)

    return intersect_postings

def intersect_documents(indexes: list[tuple[str, list[tuple[int, int]]]]) -> set[int]:
    if not indexes:
        return set()
    
    docs = set([posting[0] for posting in indexes[0][1]])

    for _, postings in indexes[1:]:
        term_docs = set([posting[0] for posting in postings])
        docs &= term_docs
    
    return docs

def filter_postings(sorted_postings: list[tuple[str, list[tuple[int, int]]]], valid_docs: set[int]) -> list[tuple[str, list[tuple[int, int]]]]:
    filtered_postings = []
    for term, postings in sorted_postings:
        filtered_list = [posting for posting in postings if posting[0] in valid_docs]
        if filtered_list:
            filtered_postings.append((term, filtered_list))
    return filtered_postings
    

def score_query(sorted_postings):
    doc_scores = defaultdict(float)

    for term, posting_list in sorted_postings:
        for doc_id, tf_idf_score in posting_list:
            # accumulate doc score
            doc_scores[doc_id] += tf_idf_score

    return dict(doc_scores)

def search_partial_index_for_term(secondary_index: list[tuple[str, str, str]], split_index_dir: str, term: str) -> list[tuple[int, int]] | None:
    file = binary_search_secondary_index(secondary_index, term)
    if not file:
        return None
    file_path = os.path.join(split_index_dir, file)
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return binary_search_partial_index(lines, term)

def binary_search_secondary_index(secondary_index: list[tuple[str, str, str]], term: str) -> str | None:
    low = 0
    high = len(secondary_index) - 1

    while low <= high:
        mid = (low + high) // 2
        first_term, last_term, filename = secondary_index[mid]

        if first_term <= term <= last_term:
            return filename
        elif term < first_term:
            high = mid - 1
        else:
            low = mid + 1

    return None

def binary_search_partial_index(lines: list[str], term: str) -> list[tuple[int, int]] | None:
    low = 0
    high = len(lines) - 1

    while low <= high:
        mid = (low + high) // 2
        mid_term_data = json.loads(lines[mid])
        mid_term = list(mid_term_data.keys())[0]

        if mid_term == term:
            return mid_term_data[mid_term]
        elif term < mid_term:
            high = mid - 1
        else:
            low = mid + 1

    return None

def load_secondary_index(path: str) -> list[tuple[str, str, str]]:
    with open(path, 'rb') as f:
        secondary_index = pickle.load(f)
    return secondary_index

def setup_search_environment():
    global SECONDARY_INDEX_BODY
    global SECONDARY_INDEX_IMPORTANT
    global DOC_MAP
    SECONDARY_INDEX_BODY = load_secondary_index(SECONDARY_BODY_INDEX_PATH)
    SECONDARY_INDEX_IMPORTANT = load_secondary_index(SECONDARY_IMPORTANT_INDEX_PATH)
    with open("doc_id_map.json", "r") as file:
        DOC_MAP = json.load(file)

def benchmark_search():
    queries = [
        "machine learning",
        "computer science",
        "artificial intelligence",
        "course offerings",
        "professorships",
        "graduate students",
        "research labs",
        "data science",
        "undergraduate programs",
        "faculty publications",
        "data structures",
        "operating systems",
        "network security",
        "database management",
        "software engineering",
        "human-computer interaction",
        "cloud computing",
        "cybersecurity",
        "big data analytics",
        "natural language processing"
    ]
    
    setup_search_environment()
    search_times = []
    for query in queries:
        start = time.perf_counter()
        results = search(query, top_k=5)
        elapsed = time.perf_counter() - start

        search_times.append(elapsed)
        print(f"\nQuery: '{query}'")
        print(f"Time: {elapsed*1000:.2f}ms")
        print(f"Results: {len(results)}")
        for url, score in results[:3]:
            print(f"  {score:.4f} - {url}")

    print("=====================================================")
    print(f"Average search time for {len(queries)} queries: {sum(search_times)/len(queries)*1000 :.2f}ms")
    print("=====================================================")

if __name__ == '__main__':
    search_query()
    # benchmark_search()
