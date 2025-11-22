from nltk.stem import PorterStemmer
import json

def search_query():
    query = input("Enter your search query: ")
    stemmed_query = stem_query(query)
    print("Stemmed Query:", stemmed_query)
    sorted_postings = get_postings(stemmed_query, partial_indexes=False)
    print(sorted_postings)

# Stems the query to work with our index
def stem_query(query: str) -> list[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in query.split()]

# Get a list of all the postings for each term and remove duplicates and sort them by length
# partial_indexes = true if we are using partial indexes
def get_postings(stemmed_query: list[str], partial_indexes: bool) -> list[tuple[str, list[int]]]:
    postings = []
    stemmed_set = set(stemmed_query)

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
        postings.append((term, term_postings))

    sorted_postings = sorted(postings, key=lambda x: len(x[1]))
    return sorted_postings

# Takes in a term and returns the posting from the full index
def get_postings_from_full_index(term: str, inverted_index) -> list[int]:
    if term in inverted_index:
        return inverted_index[term]
    return []

# Performs a boolean AND search on the sorted postings lists and returns the list of (doc_id, frequency) tuples
def boolean_AND_search(sorted_postings: list[tuple[str, list[int]]]) -> list[tuple[int, int]]:
    if not sorted_postings:
        return []

    pointers = [0] * len(sorted_postings)
    result = []


if __name__ == '__main__':
    search_query()