# file for calculating the pagerank based on the graph created with links
import json
import numpy as np
from collections import defaultdict

CORPUS = "DEV"

def calculate_page_rank(iterations:int=50, d:float=0.85):
    """
    Calculate the page rank for the graph. d = damping factor
    """
    
    with open(f"{CORPUS}_TEST/link_graph.json", "r") as file:
        graph = json.load(file)

    # need dictionary where key = page and values = pages that link to key
    inverse_graph = defaultdict(list)
    for page, neighbors in graph.items():
        for neighbor in neighbors:
            inverse_graph[str(neighbor)].append(page)

    page_rank = {}
    num_pages = len(graph)

    # initialize page_rank
    for page in graph.keys():
        page_rank[page] = 1 / num_pages

    for i in range(iterations):
        new_rank = {}
        for page, neighbors in graph.items():
            # PR(page) = (1-d)/N + d * Σ(PR(linking_page) / outlinks(linking_page))
            rank_sum = 0
            for linking_page in inverse_graph[page]:  # Pages that link TO current page
                rank_sum += page_rank[linking_page] / len(graph[linking_page])

            new_rank[page] = (1 - d) / num_pages + d * rank_sum
        page_rank = new_rank

    # Save to file
    with open("page_rank.json", "w") as f:
        json.dump(page_rank, f)

if __name__ == "__main__":
    calculate_page_rank()
