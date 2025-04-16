# app.py
import gradio as gr
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data = pd.read_json('tops_fashion.json')

# Load the SBERT model
sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Precompute embeddings for all product titles
titles = data['title'].tolist()
sbert_embeddings = sbert_model.encode(titles, show_progress_bar=True)

def sbert_recommender_gradio(input_title, num_results=5):
    query_vec = sbert_model.encode([input_title])
    sim_scores = cosine_similarity(query_vec, sbert_embeddings)[0]

    data['score'] = sim_scores
    top_matches = data.sort_values(by='score', ascending=False).head(num_results + 1)
    
    results = []
    count = 0
    
    for _, row in top_matches.iterrows():
        if row['title'].lower() == input_title.lower():
            continue  # skip the input item itself

        try:
            price_val = int(row['formatted_price'])
            price_str = f"₹{price_val:,}"
        except:
            price_str = "N/A"

        card_html = f"""
        <div style="display:flex;align-items:center;border:1px solid #ccc;border-radius:10px;padding:10px;margin-bottom:10px;gap:20px;">
            <img src="{row['medium_image_url']}" alt="image" style="height:150px;border-radius:8px;transition:transform 0.3s ease-in-out;" onmouseover="this.style.transform='scale(1.1)'" onmouseout="this.style.transform='scale(1)'">
            <div>
                <div><b>{row['title']}</b></div>
                <div>Brand: {row['brand']}</div>
                <div>Price: {price_str}</div>
                <div>Similarity: {row['score']:.2f}</div>
            </div>
        </div>
        """
        results.append(card_html)
        count += 1
        if count >= num_results:
            break

    return "\n".join(results) if results else "No similar products found."

# Gradio Interface
demo = gr.Interface(
    fn=sbert_recommender_gradio,
    inputs=[
        gr.Textbox(label="Enter Product Title"),
        gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Recommendations")
    ],
    outputs=gr.HTML(label="Recommended Products"),
    title="Amazon Apparel Recommender (SBERT)",
    description="Enter a product title and get similar apparel recommendations using SBERT embeddings."
)

if __name__ == "__main__":
    demo.launch()