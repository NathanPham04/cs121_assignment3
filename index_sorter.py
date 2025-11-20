import json

"""
Takes in a full inverted index JSON file, sorts the document ID lists for each stem,
and writes the sorted index back to a new JSON file.
"""

def sort_index(inverted_index):
    for stem in inverted_index:
        inverted_index[stem].sort()
    return inverted_index

def sort_and_write_index(inverted_index, filename="full_index.json"):
    sorted_index = sort_index(inverted_index)
    with open(filename, "w") as file:
        json.dump(sorted_index, file, indent=4)

if __name__ == "__main__":
    inverted_index = json.load(open("full_index.json"))
    sort_and_write_index(inverted_index, "full_index_sorted.json")
    print("Sorted index written to full_index_sorted.json")