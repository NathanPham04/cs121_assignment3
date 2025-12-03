from flask import Flask, render_template, request, jsonify
from search import search, setup_search_environment
import time

app = Flask(__name__)

setup_search_environment()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_query():
    query = request.json.get('query', '')
    if not query:
        return jsonify({'results': [], 'time': 0})
    
    start = time.perf_counter()
    results = search(query, top_k=10)
    elapsed = time.perf_counter() - start
    
    return jsonify({
        'results': [{'url': url, 'score': score} for url, score in results],
        'time': f'{elapsed*1000:.2f}'
    })

if __name__ == '__main__':
    app.run(debug=True)
