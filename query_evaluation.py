from old_search import search
import time
import numpy as np

# start: 1704.28

def evaluate_query(query, top_k=5):
    start = time.time()
    results = search(query, top_k)
    elapsed = time.time() - start
    return results, elapsed

if __name__ == "__main__":
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
    
    
    with open("query_evaluation_report.txt", "w") as f:
        search_times = []
        for query in good_queries:
            results, time_ms = evaluate_query(query)
            search_times.append(time_ms)
            
            output = f"\nQuery: '{query}'\nTime: {time_ms*1000:.2f}ms\nResults: {len(results)}\n"
            for url, score in results[:3]:
                output += f"  {score:.4f} - {url}\n"
            
            print(output, end="")
            f.write(output)
        
        summary = f"\n=====================================================\nAverage search time for {len(good_queries)} good queries: {np.mean(search_times)*1000 :.2f}ms\n=====================================================\n"
        print(summary)
        f.write(summary)

        search_times = []
        for query in bad_queries:
            results, time_ms = evaluate_query(query)
            search_times.append(time_ms)
            
            output = f"\nQuery: '{query}'\nTime: {time_ms*1000:.2f}ms\nResults: {len(results)}\n"
            for url, score in results[:3]:
                output += f"  {score:.4f} - {url}\n"
            
            print(output, end="")
            f.write(output)
        
        summary = f"\n=====================================================\nAverage search time for {len(bad_queries)} bad queries: {np.mean(search_times)*1000 :.2f}ms\n=====================================================\n"
        print(summary)
        f.write(summary)
