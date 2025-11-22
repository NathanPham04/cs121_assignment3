from nltk.stem import PorterStemmer
import json
import math
from collections import defaultdict
from main import tokenize

CORPUS_SIZE = 44845

def search_query():
    tokenized_query = tokenize(input("Enter your search query: "))
    stemmer = PorterStemmer()
    stemmed_query = [stemmer.stem(token) for token in tokenized_query]
    print("Stemmed Query:", stemmed_query)
    sorted_postings, all_terms_found = get_postings(stemmed_query, partial_indexes=False)
    if not all_terms_found:
        print("No documents found matching the query for boolean retrieval.")
        return
    intersected_postings, idf_dict = boolean_AND_search(sorted_postings)
    sorted_doc_scores = sorted(score_query(intersected_postings, idf_dict).items(), key=lambda x: x[1], reverse=True)
    for doc_id, score in sorted_doc_scores[:5]:
        print(f"Document ID: {doc_id}, Score: {score}")

# Get a list of all the postings for each term and remove duplicates and sort them by length
# partial_indexes = true if we are using partial indexes
def get_postings(stemmed_query: list[str], partial_indexes: bool) -> tuple[list[tuple[str, list[tuple[int, int]]]], bool]:
    postings = []
    stemmed_set = set(stemmed_query)
    all_terms_found = True
    inverted_index = None
    if not partial_indexes:
        with open("full_index.json", "r") as file:
            inverted_index = json.load(file)

    for term in stemmed_set:
        # ----------------Implement this later for full project and replace the term_postings---------------
        # if partial_indexes:
        #    
        #     term_postings = get_postings_from_partial_indexes(term)
        # else:
        #     term_postings = get_postings_from_full_index(term, inverted_index)
        # --------------------------------------------------------------------------------------------------

        term_postings = get_postings_from_full_index(term, inverted_index)
        if not term_postings:
            all_terms_found = False
        postings.append((term, term_postings))

    sorted_postings = sorted(postings, key=lambda x: len(x[1]))
    return sorted_postings, all_terms_found

# Takes in a term and returns the posting from the full index
def get_postings_from_full_index(term: str, inverted_index) -> list[tuple[int, int]]:
    if term in inverted_index:
        return inverted_index[term]
    return []

# Performs a boolean AND search on the sorted postings lists and returns the dict of term: list of (doc_id, frequency) tuples and idf weights
# Input: [(term, [(doct_id, frequency), ...]), ...]
# Output: {term: [(doc_id, frequency), ...]}, {term: idf weight}
def boolean_AND_search(sorted_postings: list[tuple[str, list[tuple[int, int]]]]) -> tuple[list[tuple[str, list[tuple[int, int]]]], dict[str, float]]:
    if not sorted_postings:
        return [], {}

    # Calculate IDF weights
    idf_dict = dict()
    for term, postings in sorted_postings:
        idf_dict[term] = math.log(CORPUS_SIZE / len(postings))

    # Get intersect of all documents in postings
    documents = intersect_documents(sorted_postings)

    # Create new list with only intersected
    intersect_postings = filter_postings(sorted_postings, documents)

    return intersect_postings, idf_dict

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
    

def score_query(sorted_postings, idf_dict):
    doc_scores = defaultdict(float)

    for term, posting_list in sorted_postings:
        if term not in idf_dict:
            continue
        
        idf_t = idf_dict[term]

        for doc_id, tf_td in posting_list:
            # tf weight: (1 + log(tf))
            tf_weight = 1 + math.log(tf_td)

            # w_{t,d} = (1 + log(tf)) * idf_t
            w_t_d = tf_weight * idf_t

            # accumulate doc score
            doc_scores[doc_id] += w_t_d

    return dict(doc_scores)

if __name__ == '__main__':
    search_query()
