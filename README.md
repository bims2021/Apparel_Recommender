 Amazon Apparel Recommender System

A Gradio-based product recommender system built using SBERT and other NLP techniques on apparel product data.
  Overview

This project explores various approaches for text-based product recommendation using product titles. The goal is to recommend visually and semantically similar fashion products given a query product title.
    Final Gradio App (SBERT-based)

    Takes in a product title

    Returns similar product titles with images

    Built using SBERT (all-MiniLM-L6-v2) – the best performing model

     Models Experimented
Model	Notes
Bag of Words	          Baseline model
TF-IDF	                Improved word weighting
Idf Only	              Tested variation
Avg Word2Vec	          Underperformed
Weighted Word2Vec	      Underperformed
SBERT	                  Best performance
Brand + Color Matching	Performed okay
Autoencoder	🥈          Second best


 Extras

    Implemented MinHash + LSH for duplicate product detection

    Used cosine similarity as the main similarity metric

    Focused on textual similarity, not visual features
