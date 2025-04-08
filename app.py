import streamlit as st
import json
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors


# -------------------- TITLE & EXPANDER --------------------
st.title("🍽️ Cuisine Recommender App")

st.markdown("""
## Smart Cuisine Recommender

Welcome to the **AI-Powered Dish Predictor**! This tool uses **Machine Learning (K-Nearest Neighbors)** to suggest dishes 
based on a few ingredients you have.


### 🔍 What This App Does:
- 📝 You enter a few ingredients (e.g., tomato, eggs, garlic, cheese)
- 🧾 It suggests the **Top 3 dishes** that best match your ingredients
- 🌍 Shows **Dish Name**, **Cuisine Type**, and full list of ingredients

""")

with st.expander("ℹ️ How to use this tool"):
    st.markdown("""
    👋 **Welcome!**

    ➤ Enter a few ingredients (e.g., _tomato, salt, sugar, garlic, onion, cheese_) in the box below.  
    ➤ The app will suggest a dish name based on your input using a trained **KNN (K-Nearest Neighbors)** model.  
    ➤ Make sure ingredients are comma-separated and basic (e.g., **no quantities/units/brands**).  

    **Example Ingredients**:
    - garlic, lemon juice, romaine lettuce, black olives, olive oil, feta cheese  
    - rice, cumin, coriander, turmeric, onion, salt.
    """)

# -------------------- LOAD & CLEAN DATA --------------------
@st.cache_data
def load_and_prepare_data():
    with open("cleaned_dishes.json", "r") as file:
        data = json.load(file)
    
    all_ingredients = [' '.join(d["cleaned_ingredients"]) for d in data]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(all_ingredients)

    knn_model = NearestNeighbors(n_neighbors=3, metric='cosine')
    knn_model.fit(X)

    return data, vectorizer, knn_model

dishes_data, vectorizer, knn_model = load_and_prepare_data()

# -------------------- CUISINE EMOJIS --------------------
cuisine_emojis = {
    "italian": "🍝", "mexican": "🌮", "indian": "🍛",
    "chinese": "🥡", "japanese": "🍣", "greek": "🥗",
    "thai": "🍜", "french": "🥖", "american": "🍔"
}

# -------------------- INPUT --------------------
user_input = st.text_input("🔍 Enter ingredients (comma-separated):")

def preprocess_input(ingredients_str):
    tokens = [ing.strip().lower() for ing in ingredients_str.split(',')]
    tokens = [re.sub(r'[^a-zA-Z\s]', '', ing) for ing in tokens if ing]
    return ' '.join(tokens)

# -------------------- PREDICT --------------------
if user_input:
    cleaned_input = preprocess_input(user_input)
    user_vector = vectorizer.transform([cleaned_input])
    distances, indices = knn_model.kneighbors(user_vector)

    st.subheader("🔝 Top 3 Dish Recommendations")

    for rank, idx in enumerate(indices[0], start=1):
        dish = dishes_data[idx]
        cuisine = dish["cuisine"]
        emoji = cuisine_emojis.get(cuisine.lower(), "🍽️")

        st.markdown(f"""
        ### 🥇 Dish #{rank}
        - 🍽️ **Dish Name**: `{dish['dish_name']}`
        - 🌍 **Cuisine**: {emoji} `{cuisine.title()}`
        - 🧂 **Ingredients**:
        """)

        for ing in dish['ingredients']:
            st.markdown(f"- 🥄 {ing}")

        st.markdown("---")
