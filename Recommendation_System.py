
# import streamlit as st
# import pandas as pd
# import numpy as np
# import random
# from datetime import datetime, timedelta
# from sklearn.metrics.pairwise import cosine_similarity
# from scipy.sparse import csr_matrix

# # Step 1: Define dataset size
# num_users = 500  # Number of users
# num_products = 20  # Limiting products to 20 grocery items
# num_interactions = 5000  # Number of interactions (ratings)

# # Step 2: Define grocery product names
# grocery_items = [
#     "Rice", "Wheat Flour", "Sugar", "Salt", "Milk", "Eggs", "Butter", "Cheese", "Yogurt", "Honey",
#     "Tea", "Coffee", "Pasta", "Noodles", "Cereal", "Oats", "Cooking Oil", "Bread", "Biscuits", "Juice"
# ]

# # Step 3: Generate user and product IDs
# user_ids = [f"U{i}" for i in range(1, num_users + 1)]
# product_ids = [f"P{i}" for i in range(1, num_products + 1)]
# product_mapping = dict(zip(product_ids, grocery_items))

# # Step 4: Generate random user-product interactions
# data = []
# for _ in range(num_interactions):
#     user = random.choice(user_ids)
#     product = random.choice(product_ids)
#     rating = random.randint(1, 5)  # Ratings from 1 to 5
#     timestamp = (datetime.now() - timedelta(days=random.randint(1, 365))).date()
#     data.append([user, product_mapping[product], rating, timestamp])

# # Step 5: Create DataFrame
# df = pd.DataFrame(data, columns=["User_ID", "Product_Name", "Rating", "Date"])

# # Step 6: Create a user-product matrix
# user_product_matrix = df.pivot_table(index='User_ID', columns='Product_Name', values='Rating', fill_value=0)

# # Step 7: Convert to sparse matrix
# sparse_matrix = csr_matrix(user_product_matrix)

# # Step 8: Compute similarity between products
# product_similarity = cosine_similarity(sparse_matrix.T)
# product_similarity_df = pd.DataFrame(product_similarity, index=user_product_matrix.columns, columns=user_product_matrix.columns)

# # Step 9: Function to get product recommendations (cached)
# @st.cache_data
# def get_recommendations(product_name, num_recommendations=5):
#     product_name = product_name.strip().title()  # Normalize input
#     if product_name not in product_similarity_df.index:
#         return ["Product not found"]
#     similar_products = product_similarity_df[product_name].sort_values(ascending=False).iloc[1:num_recommendations+1]
#     return list(similar_products.index)

# # Streamlit UI
# st.title("Grocery Product Recommendation System")
# st.write("Enter a grocery item to get recommendations.")

# # Display available products in the right sidebar
# st.sidebar.title("Available Products")
# st.sidebar.write(", ".join(grocery_items))

# # User Input
# product_input = st.text_input("Type the product name:")

# # Store recommendation results to ensure consistency across clicks
# if 'last_product' not in st.session_state:
#     st.session_state.last_product = None
#     st.session_state.last_recommendations = []

# if st.button("Get Recommendation"):
#     if product_input:
#         product_input = product_input.strip().title()
        
#         # Only update recommendations if the product changes
#         if product_input != st.session_state.last_product:
#             st.session_state.last_product = product_input
#             st.session_state.last_recommendations = get_recommendations(product_input)
        
#         # Display recommendations
#         if st.session_state.last_recommendations and "Product not found" not in st.session_state.last_recommendations:
#             st.write(f"Recommended products for {st.session_state.last_product}:")
#             for i, product in enumerate(st.session_state.last_recommendations, 1):
#                 st.write(f"{i}. {product}")
#         else:
#             st.warning("Invalid product! Please enter a valid product from the list.")
#     else:
#         st.warning("Please enter a product name.")






import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Step 1: Define dataset size
num_users = 500  # Number of users
num_products = 20  # Limiting products to 20 grocery items
num_interactions = 5000  # Number of interactions (ratings)

# Step 2: Define grocery product names
grocery_items = [
    "Rice", "Wheat Flour", "Sugar", "Salt", "Milk", "Eggs", "Butter", "Cheese", "Yogurt", "Honey",
    "Tea", "Coffee", "Pasta", "Noodles", "Cereal", "Oats", "Cooking Oil", "Bread", "Biscuits", "Juice"
]

# Step 3: Generate user and product IDs
user_ids = [f"U{i}" for i in range(1, num_users + 1)]
product_ids = [f"P{i}" for i in range(1, num_products + 1)]
product_mapping = dict(zip(product_ids, grocery_items))

# Step 4: Generate random user-product interactions
data = []
for _ in range(num_interactions):
    user = random.choice(user_ids)
    product = random.choice(product_ids)
    rating = random.randint(1, 5)  # Ratings from 1 to 5
    timestamp = (datetime.now() - timedelta(days=random.randint(1, 365))).date()
    data.append([user, product_mapping[product], rating, timestamp])

# Step 5: Create DataFrame
df = pd.DataFrame(data, columns=["User_ID", "Product_Name", "Rating", "Date"])

# Step 6: Create a user-product matrix
user_product_matrix = df.pivot_table(index='User_ID', columns='Product_Name', values='Rating', fill_value=0)

# Step 7: Convert to sparse matrix
sparse_matrix = csr_matrix(user_product_matrix)

# Step 8: Compute similarity between products
product_similarity = cosine_similarity(sparse_matrix.T)
product_similarity_df = pd.DataFrame(product_similarity, index=user_product_matrix.columns, columns=user_product_matrix.columns)

# Step 9: Function to get product recommendations (cached)
@st.cache_data
def get_recommendations(product_name, num_recommendations=5):
    product_name = product_name.strip().title()  # Normalize input
    if product_name not in product_similarity_df.index:
        return ["Product not found"]
    similar_products = product_similarity_df[product_name].sort_values(ascending=False).iloc[1:num_recommendations+1]
    return list(similar_products.index)

# Streamlit UI
st.set_page_config(page_title="Grocery Recommender", page_icon="🛒", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Grocery Product Recommendation System</h1>", unsafe_allow_html=True)
st.write("Enter a grocery item to get recommendations.")

# Display available products in the right sidebar
st.sidebar.markdown("<h2 style='color: #2196F3;'>Available Products</h2>", unsafe_allow_html=True)
st.sidebar.write(", ".join(grocery_items))

# User Input
product_input = st.text_input("**Type the product name:**", placeholder="E.g., Sugar, Milk, Eggs...")

# Store recommendation results to ensure consistency across clicks
if 'last_product' not in st.session_state:
    st.session_state.last_product = None
    st.session_state.last_recommendations = []

if st.button("🎯 Get Recommendation", help="Click to see recommended products"):
    if product_input:
        product_input = product_input.strip().title()
        
        # Only update recommendations if the product changes
        if product_input != st.session_state.last_product:
            st.session_state.last_product = product_input
            st.session_state.last_recommendations = get_recommendations(product_input)
        
        # Display recommendations
        if st.session_state.last_recommendations and "Product not found" not in st.session_state.last_recommendations:
            st.markdown(f"<h3 style='color: #FF5722;'>Recommended products for {st.session_state.last_product}:</h3>", unsafe_allow_html=True)
            for i, product in enumerate(st.session_state.last_recommendations, 1):
                st.markdown(f"<p style='color: #673AB7; font-size: 18px;'><b>{i}. {product}</b></p>", unsafe_allow_html=True)
        else:
            st.warning("🚨 Invalid product! Please enter a valid product from the list.")
    else:
        st.warning("⚠️ Please enter a product name.")
