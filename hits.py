import json
from collections import defaultdict

CORPUS = "DEV"

def compute_hits_score(k:int):
    """
    Compute the HITS score for the graph.
    k = number of iterations
    """
    
    with open(f"{CORPUS}_TEST/link_graph.json", "r") as file:
        graph = json.load(file)

    # need dictionary where key = page and values = pages that link to key
    inverse_graph = defaultdict(list)
    for page, neighbors in graph.items():
        for neighbor in neighbors:
            inverse_graph[str(neighbor)].append(str(page))

    authority = {}
    hub = {}

    # initialize all pages to have an authority and hub score of 1
    for page in graph.keys():
        authority[str(page)] = 1
        hub[str(page)] = 1

    # print(list(graph.keys())[:50])

    for i in range(k):
        # update authorities first
        for page in graph.keys():
            authority[page] = 0
            for incoming_page in inverse_graph[page]:
                authority[page] += hub[incoming_page]
        
        norm = sum(authority[page] ** 2 for page in graph.keys()) ** 0.5
        for page in graph.keys():
            if norm != 0:
                authority[page] /= norm
        
        # update hubs
        for page in graph.keys():
            hub[page] = 0
            for outgoing_page in graph[page]:
                hub[page] += authority[str(outgoing_page)]
        
        norm = sum(hub[page] ** 2 for page in graph.keys()) ** 0.5
        for page in graph.keys():
            if norm != 0:
                hub[page] /= norm
    
    with open(f"{CORPUS}_TEST/authority_scores.json", "w") as f:
        json.dump(authority, f)
    
    with open(f"{CORPUS}_TEST/hub_scores.json", "w") as f:
        json.dump(hub, f)

    print("scores saved")
    
    return authority, hub

if __name__ == "__main__":
    compute_hits_score(50)