import os
from flask import Flask, request, render_template_string
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import requests  # HTTP client for external API calls

app = Flask(__name__)

# Initialize ChromaDB persistent client and collection
client = chromadb.PersistentClient(path="Chroma_db")
collection = client.get_or_create_collection(name="products")

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# HTML template for the interface
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>E-commerce Semantic Search</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; }
    input[type=text], input[type=file], select, button {
      padding: 8px; margin: 6px 0; width: 300px;
    }
    button { width: 150px; }
    .result { margin-top: 15px; }
  </style>
</head>
<body>
  <h1>🛒 E-commerce Semantic Search Demo</h1>

  <h2>Upload CSV</h2>
  <form method="POST" action="/upload" enctype="multipart/form-data">
    <input type="file" name="csv_file" accept=".csv" required>
    <br>
    <button type="submit">Upload and Index</button>
  </form>
  {% if upload_message %}
    <p style="color:green;">{{ upload_message }}</p>
  {% elif upload_error %}
    <p style="color:red;">{{ upload_error }}</p>
  {% endif %}

  <h2>Search Products</h2>
  <form method="GET" action="/">
    <input type="text" name="query" placeholder="Search for products" value="{{ query|default('') }}" required>
    <br>
    <label for="top_k">Number of results:</label>
    <select name="top_k" id="top_k">
      {% for i in range(1,11) %}
        <option value="{{ i }}" {% if top_k==i %}selected{% endif %}>{{ i }}</option>
      {% endfor %}
    </select>
    <br>
    <button type="submit">Search</button>
  </form>

  {% if results %}
    <div class="result">
      <h3>Top Results:</h3>
      <ul>
        {% for item in results %}
          <li><strong>{{ item.name }}</strong>: {{ item.description }}</li>
        {% endfor %}
      </ul>
    </div>
  {% elif query %}
    <p>No results found.</p>
  {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    query = request.args.get("query", "")
    top_k = request.args.get("top_k", 3, type=int)
    results = []

    if query:
        query_emb = model.encode(query).tolist()
        res = collection.query(query_embeddings=[query_emb], n_results=top_k)
        metadatas = res.get("metadatas", [[]])[0]
        for md in metadatas:
            results.append({"name": md.get("name", ""), "description": md.get("description", "")})

    return render_template_string(HTML_TEMPLATE, query=query, top_k=top_k, results=results)

@app.route("/upload", methods=["POST"])
def upload():
    csv_file = request.files.get("csv_file")
    if not csv_file:
        return render_template_string(HTML_TEMPLATE, upload_error="No file selected.")

    try:
        df = pd.read_csv(csv_file)
    except Exception:
        return render_template_string(HTML_TEMPLATE, upload_error="Error reading CSV file.")

    if "name" not in df.columns or "description" not in df.columns:
        return render_template_string(HTML_TEMPLATE, upload_error="CSV must have 'name' and 'description' columns.")

    # Replace this URL with your actual external API endpoint
    external_api_url = "https://external.api/endpoint"

    # Process each row: send to external API, then index embedding
    for idx, row in df.iterrows():
        try:
            # Send product data to external service
            external_response = requests.post(
                external_api_url,
                json={
                    "id": row.get("id", idx),
                    "name": row["name"],
                    "description": row["description"]
                },
                timeout=5 # seconds
            )
            external_response.raise_for_status()  # Raise error if bad response
        except Exception as e:
            # Log or handle external API errors (optional)
            print(f"Warning: External API call failed for row {idx}: {e}")

        # Generate embedding and add to ChromaDB
        emb = model.encode(row['description']).tolist()
        collection.add(
            ids=[str(row.get("id", idx))],
            metadatas=[{"name": row['name'], "description": row['description']}],
            embeddings=[emb]
        )

    return render_template_string(HTML_TEMPLATE, upload_message=f"Successfully processed and indexed {len(df)} products!")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8501)
