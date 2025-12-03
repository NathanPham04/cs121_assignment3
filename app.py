from flask import Flask, render_template, request, jsonify
from search import search, setup_search_environment

app = Flask(__name__)

setup_search_environment()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_query():
    query = request.json.get('query', '')
    if not query:
        return jsonify({'results': []})
    
    results = search(query, top_k=10)
    return jsonify({'results': [{'url': url, 'score': score} for url, score in results]})

if __name__ == '__main__':
    app.run(debug=True)
