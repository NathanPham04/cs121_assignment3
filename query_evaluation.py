from search import search
import time
import numpy as np

def evaluate_query(query, top_k=5):
    start = time.time()
    results = search(query, top_k)
    elapsed = time.time() - start
    return results, elapsed

if __name__ == "__main__":
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
    
    search_times = []
    for query in queries:
        results, time_ms = evaluate_query(query)

        search_times.append(time_ms)
        print(f"\nQuery: '{query}'")
        print(f"Time: {time_ms*1000:.2f}ms")
        print(f"Results: {len(results)}")
        for url, score in results[:3]:
            print(f"  {score:.4f} - {url}")

    print("=====================================================")
    print(f"Average search time for {len(queries)} queries: {np.mean(search_times)*1000 :.2f}ms")
    print("=====================================================")
