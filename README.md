# CuisineRecommender
Smart Cuisine Recommendation App


:

🍽️ Cuisine Recommendation System using Streamlit & KNN
Developed a smart Cuisine Recommendation App that suggests the top 3 dishes based on user-input ingredients using a K-Nearest Neighbors (KNN) model trained on a custom cleaned dataset of over 3,800+ dishes and their ingredients.

Performed data wrangling and ingredient normalization on a large food dataset by applying regex cleaning, text preprocessing, and custom NLP tokenization to create a reliable training base for ingredient vectorization.

Engineered a feature extraction pipeline using CountVectorizer from Scikit-learn to convert dish ingredients into sparse matrices, followed by cosine similarity for dish matching through Nearest Neighbors algorithm.

Designed an intuitive and visually appealing Streamlit web interface, enriched with emojis and dynamic markdown, allowing users to input basic ingredients (like garlic, onion, tomato) and instantly get matching dishes with cuisine type and full ingredient list.

Deployed the app on Hugging Face Spaces, making the tool publicly accessible for real-time experimentation, and added detailed usage instructions, helping non-tech users get seamless recommendations without needing domain knowledge.

