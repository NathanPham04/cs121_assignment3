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

CORPUS = "DEV_TEST_PROD"
SECONDARY_BODY_INDEX_PATH = f"{CORPUS}/secondary_index/body.pkl"
SECONDARY_IMPORTANT_INDEX_PATH = f"{CORPUS}/secondary_index/important_words.pkl"
BODY_INDEX_DIR = f"{CORPUS}/split_index_weighted/body/"
IMPORTANT_INDEX_DIR = f"{CORPUS}/split_index_weighted/important_words/"

CORPUS_SIZE = 44845
SECONDARY_INDEX_BODY = list()
SECONDARY_INDEX_IMPORTANT = list()
DOC_MAP = dict()

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd",
    "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
    "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
    "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
    "will", "can", "s", "d", "p", "b", "m", "j", "pp", "n", "e", "t", "o"
}

def search(query_text, top_k=5):
    removed_stop_words_query = remove_stop_words_from_query(query_text)
    tokenized_query = tokenize(removed_stop_words_query)
    stemmer = PorterStemmer()
    stemmed_query = []
    for token in tokenized_query:
        stem = stemmer.stem(token)
        stemmed_query.append(stem if stem else token)

    # Get postings for the content body (THIS STILL REQUIRES ALL TERMS TO BE PRESENT IN THE CONTENT)
    sorted_postings_body, all_terms_found = get_postings(stemmed_query, SECONDARY_INDEX_BODY, BODY_INDEX_DIR)
    if not all_terms_found:
        return []
    intersected_postings, documents = boolean_AND_search(sorted_postings_body)

    # Get postings for the important sections (title, headings, bold) - DOESN'T REQUIRE ALL TERMS TO BE PRESENT
    sorted_postings_important, all_terms_found = get_postings(stemmed_query, SECONDARY_INDEX_IMPORTANT, IMPORTANT_INDEX_DIR)
    found_important_postings = filter_postings(sorted_postings_important, documents)

    sorted_doc_scores = sorted(score_query(intersected_postings, found_important_postings).items(), key=lambda x: x[1], reverse=True)
    return [(DOC_MAP[str(doc_id)], score) for doc_id, score in sorted_doc_scores[:top_k]]

def remove_stop_words_from_query(query_string: str) -> str:
    return ' '.join([word for word in query_string.split() if word.lower() not in STOP_WORDS])

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
def get_postings(stemmed_query: list[str], secondary_index: list[tuple[str, str, str]], split_index_dir: str) -> tuple[list[tuple[str, list[tuple[int, int]]]], bool]:
    postings = []
    stemmed_set = set(stemmed_query)
    all_terms_found = True

    for term in stemmed_set:
        term_postings = search_partial_index_for_term(SECONDARY_INDEX_BODY, BODY_INDEX_DIR, term)
        if not term_postings:
            all_terms_found = False
        if term_postings:
            postings.append((term, term_postings))

    sorted_postings = sorted(postings, key=lambda x: len(x[1]))
    return sorted_postings, all_terms_found

# Performs a boolean AND search on the sorted postings lists and returns the dict of term: list of (doc_id, tf-idf)
# Input: [(term, [(doct_id, tf-idf), ...]), ...]
# Output: {term: [(doc_id, tf-idf), ...]}
def boolean_AND_search(sorted_postings: list[tuple[str, list[tuple[int, int]]]]) -> tuple[list[tuple[str, list[tuple[int, int]]]], set[int]]:
    if not sorted_postings:
        return [], set()

    # Get intersect of all documents in postings
    documents = intersect_documents(sorted_postings)

    # Create new list with only intersected
    intersect_postings = filter_postings(sorted_postings, documents)

    return intersect_postings, documents

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
    

def score_query(sorted_postings_body, sorted_postings_important) -> dict[int, float]:
    doc_scores = defaultdict(float)

    for term, posting_list in sorted_postings_body:
        for doc_id, tf_idf_score in posting_list:
            # accumulate doc score
            doc_scores[doc_id] += tf_idf_score

    for term, posting_list in sorted_postings_important:
        for doc_id, tf_idf_score in posting_list:
            # accumulate doc score with a higher weight for important sections
            doc_scores[doc_id] += tf_idf_score * 1.5  # Weight factor for important sections

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
    good_queries = [
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

    bad_queries = [
        "kendrick lamar",
        "this is a really long query that is supposed to take much longer than the good queries",
        "asdfghjkl qwertyuiop zxcvbnm",
        "lorem ipsum dolor sit amet consectetur adipiscing elit",
        "the quick brown fox jumps over the lazy dog",
        "to be or not to be that is the question",
        "macklemore",
        "random words that do not make sense together",
        "thom yorke",
        "janice joplin",
        "the foo fighters",
        "leo fender",
        "david goggins",
        "elon musk",
        "jeff bezos",
        "the meaning of life the universe and everything",
        "leo tolstory",
        "zack efron",
        "chris hemsworth",
        "scarlett johansson",
        "walt disney"
    ]
    
    setup_search_environment()
    search_times = []
    for query in good_queries:
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
    print(f"Average search time for {len(good_queries)} queries: {sum(search_times)/len(good_queries)*1000 :.2f}ms")
    print("=====================================================")


    search_times = []
    for query in bad_queries:
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
    print(f"Average search time for {len(bad_queries)} queries: {sum(search_times)/len(bad_queries)*1000 :.2f}ms")
    print("=====================================================")

if __name__ == '__main__':
    # search_query()
    benchmark_search()
