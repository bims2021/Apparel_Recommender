# Apparel_Recommender
Gradio based apparel recommder system using sbert.
Started with bag of words model, then went to tf-idf model followed by idf model.
Experimented with average word2vec and weighted word2vec.
Then went for transformers (sbert ie.all_MiniLMweighted -L6-v2) which is the best performing model out of all.
Have also tried  brand and colour model and autoencoder(second best performing model).
All of these models are in AmazonApparelRecommendation.ipynb file
The gradio_apparel_recommender has only the sbert model which takes title of the product and recommends products .
Have used MinHash-LSH to find near duplicates.
