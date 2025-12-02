"""
This module contains functions to split merged index files into smaller partial index files
with TF-IDF weighted postings, and to create secondary indexes for efficient searching.
"""

import os
import json
import math
from index_processor import OUTPUT_DIR
from indexer import CORPUS, ensure_dir
import pickle

MERGED_INDEX_BODY_DIR = OUTPUT_DIR + "merged_index.jsonl"
MERGED_INDEX_IMPORTANT_DIR = OUTPUT_DIR + "merged_index_important_words.jsonl"
MERGED_INDEX_ANCHOR_DIR = OUTPUT_DIR + "merged_index_anchor_words.jsonl"
SPLIT_OUTPUT_DIR = f"{CORPUS}_TEST/split_index_weighted/"
SPLIT_OUTPUT_BODY_DIR = SPLIT_OUTPUT_DIR + "body/"
SPLIT_OUTPUT_IMPORTANT_DIR = SPLIT_OUTPUT_DIR + "important_words/"
SPLIT_OUTPUT_ANCHOR_DIR = SPLIT_OUTPUT_DIR + "anchor_words/"
NUM_TOKENS_PER_FILE = 1000
CORPUS_SIZE = 44845

SECONDARY_INDEX_DIR = f"{CORPUS}_TEST/secondary_index/"
SECONDARY_BODY_INDEX_PATH = SECONDARY_INDEX_DIR + "body.pkl"
SECONDARY_IMPORTANT_INDEX_PATH = SECONDARY_INDEX_DIR + "important_words.pkl"
SECONDARY_ANCHOR_INDEX_PATH = SECONDARY_INDEX_DIR + "anchor_words.pkl"

def split_to_partial_indexes_with_tf_idf_embedding(input_dir: str, output_dir: str):
    curr_file_index = 0
    curr_tokens = 0
    with open(input_dir, 'r') as f:
        while True:
            partial_index_path = os.path.join(output_dir, f"{curr_file_index}.jsonl")
            with open(partial_index_path, 'a') as partial_file:
                while curr_tokens < NUM_TOKENS_PER_FILE:
                    line = f.readline()

                    if not line:
                        return  # End of file
                    
                    token_data = json.loads(line)
                    term = list(token_data.keys())[0]
                    postings = token_data[term]
                    doc_freq = len(postings)
                    idf = math.log(CORPUS_SIZE / doc_freq)
                    weighted_postings = []

                    # Calculate TF-IDF for each posting: (doc_id, freq) -> (doc_id, tf_idf)
                    for doc_id, freq in postings:
                        tf = 1 + math.log(freq)
                        tf_idf = tf * idf
                        weighted_postings.append((doc_id, tf_idf))

                    # Write the term with weighted postings to the partial index file
                    partial_file.write(json.dumps({term: weighted_postings}) + '\n')
                    
                    curr_tokens += 1

            # Reset for next file
            curr_file_index += 1
            curr_tokens = 0

# Secondary index creation for a split index directory
# (first_term, last_term, filename)
def create_secondary_index_on_split_dir(input_dir: str, output_path: str):
    secondary_index = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                first_term = json.loads(lines[0])
                last_term = json.loads(lines[-1])
                first_term_key = list(first_term.keys())[0]
                last_term_key = list(last_term.keys())[0]
                secondary_index.append((first_term_key, last_term_key, filename))

    with open(output_path, 'wb') as f:
        pickle.dump(secondary_index, f)

def create_split_indexes():
    ensure_dir(SPLIT_OUTPUT_BODY_DIR)
    ensure_dir(SPLIT_OUTPUT_IMPORTANT_DIR)
    ensure_dir(SPLIT_OUTPUT_ANCHOR_DIR)
    split_to_partial_indexes_with_tf_idf_embedding(MERGED_INDEX_BODY_DIR, SPLIT_OUTPUT_BODY_DIR)
    split_to_partial_indexes_with_tf_idf_embedding(MERGED_INDEX_IMPORTANT_DIR, SPLIT_OUTPUT_IMPORTANT_DIR)
    split_to_partial_indexes_with_tf_idf_embedding(MERGED_INDEX_ANCHOR_DIR, SPLIT_OUTPUT_ANCHOR_DIR)

def create_secondary_index_on_split():
    ensure_dir(SECONDARY_INDEX_DIR)
    create_secondary_index_on_split_dir(SPLIT_OUTPUT_BODY_DIR, SECONDARY_BODY_INDEX_PATH)
    create_secondary_index_on_split_dir(SPLIT_OUTPUT_IMPORTANT_DIR, SECONDARY_IMPORTANT_INDEX_PATH)
    create_secondary_index_on_split_dir(SPLIT_OUTPUT_ANCHOR_DIR, SECONDARY_ANCHOR_INDEX_PATH)
    
if __name__ == '__main__':
    create_split_indexes()
    create_secondary_index_on_split()