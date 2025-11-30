import os
import json

from indexer import PARTIAL_INDEX_BODY_DIR, PARTIAL_INDEX_IMPORTANT_DIR, CORPUS


OUTPUT_DIR = f"{CORPUS}_test/merged_index/"
INPUT_DIR_BODY = PARTIAL_INDEX_BODY_DIR
INPUT_DIR_IMPORTANT = PARTIAL_INDEX_IMPORTANT_DIR

def merge_partial_indexes():
    body_files_list = get_files_in_directory_os(INPUT_DIR_BODY)
    important_files_list = get_files_in_directory_os(INPUT_DIR_IMPORTANT)
    create_merged_index_directory(OUTPUT_DIR)
    merge_partial_index_files(body_files_list, INPUT_DIR_BODY, OUTPUT_DIR + "merged_index.jsonl")
    merge_partial_index_files(important_files_list, INPUT_DIR_IMPORTANT, OUTPUT_DIR + "merged_index_important_words.jsonl")

# Retrieves a list of all files in the specified directory
def get_files_in_directory_os(directory_path):
    files_list = []
    for item in os.listdir(directory_path):
        full_path = os.path.join(directory_path, item)
        if os.path.isfile(full_path) and item.endswith('.jsonl'):
            files_list.append(item)
    return files_list

# Create the merged index directory
def create_merged_index_directory(directory_path: str):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# Open all partial index files and read them line by line to merge
def merge_partial_index_files(files_list: list[str], directory_path: str, output_path: str):
    current_term = None
    current_postings = []
    
    files = [open(os.path.join(directory_path, file), 'r') for file in files_list]
    curr_tokens = [None] * len(files)

    with open(output_path, 'w') as output_file:
        while not check_done_reading(files):
            for i, file in enumerate(files):
                if file is not None and curr_tokens[i] is None:
                    line = file.readline()
                    if line:
                        curr_tokens[i] = json.loads(line)
                    else:
                        files[i] = None  # type: ignore # Mark file as done
                        
            # Find the smallest term among the current tokens
            smallest_term = None
            for token in curr_tokens:
                if token is not None:
                    term = list(token.keys())[0]
                    if smallest_term is None or term < smallest_term:
                        smallest_term = term
            if smallest_term is None:
                break  # All files are done 

            # Merge postings for the smallest term
            merged_postings = []
            for i, token in enumerate(curr_tokens):
                if token is not None:
                    term = list(token.keys())[0]
                    if term == smallest_term:
                        merged_postings.extend(token[term])
                        curr_tokens[i] = None  # Mark this token as processed

            # Write merged term and postings to output file
            merged_postings.sort()
            merged_entry = {smallest_term: merged_postings}
            output_file.write(json.dumps(merged_entry) + '\n')
            

def check_done_reading(files):
    return all(f is None for f in files) 

if __name__ == '__main__':
    merge_partial_indexes()