import os
import re
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# APP CONFIGURATION
# ============================================================
DATA_FILE = "products_cleaned.csv"

# A recommendation must have at least this much TF-IDF relevance.
# Zero-similarity products are always rejected.
MIN_SIMILARITY = 0.03

REQUIRED_COLUMNS = {
    "product_id",
    "name",
    "main_category",
    "image",
    "ratings",
    "no_of_ratings",
    "actual_price",
}


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Product Finder",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PASTEL FRONT-END
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background: #FFF9F5;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(135deg, #E7F2FF 0%, #F8ECFF 52%, #FFF0E5 100%);
        border: 1px solid #E5DDF0;
        border-radius: 28px;
        padding: 2rem 2.3rem;
        margin-bottom: 1.35rem;
        box-shadow: 0 8px 24px rgba(100, 85, 125, 0.08);
    }

    .hero h1 {
        color: #403A55;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }

    .hero p {
        color: #6E687E;
        font-size: 1.02rem;
        margin: 0.65rem 0 0;
    }

    .section-card {
        background: #FFFFFF;
        border: 1px solid #EAE3F0;
        border-radius: 22px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 6px 18px rgba(100, 90, 120, 0.055);
    }

    .search-heading {
        background: linear-gradient(135deg, #F2EEFF 0%, #EEF7FF 100%);
        border: 1px solid #E4DDF0;
        border-radius: 18px;
        padding: 0.9rem 1.2rem;
        margin: 0.5rem 0 0.9rem;
    }

    .search-heading h3 {
        color: #403A55;
        margin: 0;
        font-size: 1.25rem;
    }

    .search-heading p {
        color: #716B7D;
        margin: 0.25rem 0 0;
        font-size: 0.9rem;
    }

    .product-card {
        background: #FFFFFF;
        border: 1px solid #E9E1EE;
        border-radius: 22px;
        padding: 1rem;
        margin: 0.7rem 0;
        box-shadow: 0 5px 15px rgba(100, 90, 120, 0.05);
    }

    .product-title {
        color: #3E3953;
        font-size: 1.16rem;
        font-weight: 750;
        line-height: 1.38;
        margin-bottom: 0.5rem;
    }

    .product-meta {
        color: #6B6579;
        font-size: 0.93rem;
        margin: 0.22rem 0;
    }

    .price {
        color: #76599E;
        font-size: 1.22rem;
        font-weight: 800;
    }

    .metric-box {
        background: #F7F2FC;
        border-radius: 14px;
        padding: 0.55rem 0.75rem;
        text-align: center;
        color: #5C526C;
        margin-top: 0.5rem;
    }

    .metric-value {
        color: #594577;
        font-size: 1.05rem;
        font-weight: 750;
    }

    .image-not-found {
        background: #F1EEF4;
        border: 1px solid #DED7E5;
        border-radius: 17px;
        min-height: 220px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        color: #817A8B;
        font-weight: 700;
        padding: 1rem;
    }

    .empty-state {
        background: #FFF1F1;
        border: 1px dashed #E1AAAA;
        border-radius: 18px;
        padding: 1.2rem;
        text-align: center;
        color: #825858;
        font-weight: 650;
        margin-top: 1rem;
    }

    .info-box {
        background: #EDF8F3;
        border: 1px solid #CFE9DC;
        border-radius: 16px;
        padding: 0.85rem 1rem;
        color: #4D7562;
        margin-bottom: 1rem;
    }

    /* Search input */
    div[data-testid="stTextInput"] input {
        background: #F7F4FC;
        border: 1px solid #DDD3EA;
        border-radius: 14px;
        color: #403A55;
        min-height: 3rem;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #BCA9E8;
        box-shadow: 0 0 0 1px #BCA9E8;
    }

    section[data-testid="stSidebar"] {
        background: #F5F0FF;
        border-right: 1px solid #E5DDF0;
    }

    .stButton > button {
        background: #BCA9E8;
        color: #30294A;
        border: none;
        border-radius: 14px;
        font-weight: 800;
        min-height: 3rem;
        box-shadow: 0 5px 12px rgba(118, 89, 158, 0.16);
    }

    .stButton > button:hover {
        background: #AC97DE;
        color: #30294A;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TEXT PREPROCESSING
# ============================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' was not found. Keep products_cleaned.csv "
            "in the same folder as app.py."
        )

    data = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    # Keep the exact product fields needed by the notebook/app.
    data = data.copy()

    numeric_columns = [
        "ratings",
        "no_of_ratings",
        "actual_price",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(
        subset=[
            "name",
            "main_category",
            "ratings",
            "no_of_ratings",
            "actual_price",
        ]
    ).reset_index(drop=True)

    return data


# ============================================================
# BUILD THE SAME CONTENT-BASED MODEL AS THE NOTEBOOK
# ============================================================
@st.cache_resource
def build_recommender(data):
    data = data.copy()

    data["price_log"] = np.log1p(data["actual_price"])
    data["rating_count_log"] = np.log1p(data["no_of_ratings"])
    data["rating_score"] = data["ratings"] / 5.0

    data["popularity_score"] = (
        data["rating_score"] * data["rating_count_log"]
    )

    data["price_bucket"] = pd.qcut(
        data["actual_price"],
        q=4,
        labels=["Budget", "Mid-Range", "Premium", "Luxury"],
        duplicates="drop",
    )

    data["clean_name"] = data["name"].apply(clean_text)

    data["product_text"] = (
        data["clean_name"]
        + " "
        + data["main_category"].astype(str).str.lower()
        + " "
        + data["price_bucket"].astype(str).str.lower()
    ).str.strip()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )

    tfidf_matrix = vectorizer.fit_transform(data["product_text"])

    return data, vectorizer, tfidf_matrix


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================
def recommend_products(
    data,
    vectorizer,
    tfidf_matrix,
    query,
    max_price,
    category,
    top_n,
):
    query_clean = clean_text(query)

    if not query_clean:
        return pd.DataFrame()

    query_vector = vectorizer.transform([query_clean])

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix,
    ).flatten()

    results = data.copy()
    results["text_similarity"] = similarities

    # --------------------------------------------------------
    # HARD CONSTRAINTS
    # --------------------------------------------------------
    if max_price is not None:
        results = results[
            results["actual_price"] <= max_price
        ]

    if category != "All Categories":
        results = results[
            results["main_category"].astype(str).str.lower()
            == category.lower()
        ]

    if results.empty:
        return results

    # --------------------------------------------------------
    # RELEVANCE GATE
    # --------------------------------------------------------
    # Never return products with zero/near-zero textual relevance.
    results = results[
        results["text_similarity"] >= MIN_SIMILARITY
    ]

    if results.empty:
        return results

    # --------------------------------------------------------
    # HEURISTIC RANKING
    # 70% text similarity
    # 20% rating
    # 10% popularity
    # --------------------------------------------------------
    rating_min = results["ratings"].min()
    rating_max = results["ratings"].max()

    if rating_max == rating_min:
        results["normalized_rating"] = 1.0
    else:
        results["normalized_rating"] = (
            (results["ratings"] - rating_min)
            / (rating_max - rating_min)
        )

    popularity_min = results["popularity_score"].min()
    popularity_max = results["popularity_score"].max()

    if popularity_max == popularity_min:
        results["normalized_popularity"] = 1.0
    else:
        results["normalized_popularity"] = (
            (results["popularity_score"] - popularity_min)
            / (popularity_max - popularity_min)
        )

    results["ranking_score"] = (
        0.70 * results["text_similarity"]
        + 0.20 * results["normalized_rating"]
        + 0.10 * results["normalized_popularity"]
    )

    results = results.sort_values(
        ["ranking_score", "text_similarity", "ratings"],
        ascending=[False, False, False],
    )

    return results.head(top_n).reset_index(drop=True)


# ============================================================
# IMAGE HANDLING
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_product_image(url):
    """Return a PIL image if the URL works; otherwise return None."""
    if not isinstance(url, str) or not url.strip():
        return None

    try:
        response = requests.get(
            url.strip(),
            timeout=5,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "Chrome/131.0 Safari/537.36"
                )
            },
        )
        response.raise_for_status()

        image = Image.open(BytesIO(response.content))
        image.load()
        return image.convert("RGB")

    except Exception:
        return None


def show_product_image(url):
    image = fetch_product_image(url)

    if image is not None:
        st.image(image, use_container_width=True)
    else:
        st.markdown(
            """
            <div class="image-not-found">
                <div>
                    <div style="font-size: 2rem;">🖼️</div>
                    Image not found
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# LOAD APP DATA
# ============================================================
try:
    df = load_data(DATA_FILE)
    df, vectorizer, tfidf_matrix = build_recommender(df)
except Exception as exc:
    st.error(str(exc))
    st.stop()


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>🛍️ Product Finder</h1>
        <p>
            Find relevant products using content-based search,
            budget/category constraints, ratings, and popularity.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## Search Controls")

    st.caption(
        "The system first applies your shopping constraints, "
        "then removes products that are not sufficiently relevant."
    )

    categories = ["All Categories"] + sorted(
        df["main_category"].dropna().unique().tolist()
    )

    category = st.selectbox(
        "Category",
        categories,
    )

    max_catalog_price = float(df["actual_price"].max())

    max_price = st.number_input(
        "Maximum budget (₹)",
        min_value=0.0,
        max_value=max_catalog_price,
        value=min(5000.0, max_catalog_price),
        step=100.0,
    )

    top_n = st.slider(
        "Number of recommendations",
        min_value=1,
        max_value=20,
        value=10,
    )

    st.markdown("---")
    st.markdown("### Recommendation Method")
    st.write("• TF-IDF + cosine similarity")
    st.write("• Hard budget/category constraints")
    st.write("• Minimum relevance threshold")
    st.write("• 70% text relevance")
    st.write("• 20% rating")
    st.write("• 10% popularity")


# ============================================================
# SEARCH AREA
# ============================================================
st.markdown(
    '''
    <div class="search-heading">
        <h3>Find your product</h3>
        <p>Enter what you need, then choose your category and budget.</p>
    </div>
    ''',
    unsafe_allow_html=True,
)

query = st.text_input(
    "What are you looking for?",
    placeholder="e.g. wireless headphones, sports shoes, digital camera",
)

search_clicked = st.button(
    "🔎 Find Products",
    type="primary",
    use_container_width=True,
)

# ============================================================
# RESULTS
# ============================================================
if search_clicked:
    if not query.strip():
        st.warning("Please enter a product search query.")
        st.stop()

    recommendations = recommend_products(
        data=df,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        query=query,
        max_price=max_price,
        category=category,
        top_n=top_n,
    )

    if recommendations.empty:
        st.markdown(
            f"""
            <div class="empty-state">
                No relevant products were found for
                <strong>{query}</strong> with the selected constraints.
                <br><br>
                Try increasing the budget, changing the category,
                or using a broader product description.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.subheader("Recommended Products")

        price_valid = (
            recommendations["actual_price"] <= max_price
        ).all()

        category_valid = (
            category == "All Categories"
            or (
                recommendations["main_category"].astype(str).str.lower()
                == category.lower()
            ).all()
        )

        relevance_valid = (
            recommendations["text_similarity"] >= MIN_SIMILARITY
        ).all()

        ranking_valid = (
            recommendations["ranking_score"].is_monotonic_decreasing
        )

        st.markdown(
            f"""
            <div class="info-box">
                Found <strong>{len(recommendations)}</strong>
                relevant recommendation(s) matching the current constraints.
            </div>
            """,
            unsafe_allow_html=True,
        )

        for _, product in recommendations.iterrows():
            st.markdown(
                '<div class="product-card">',
                unsafe_allow_html=True,
            )

            image_col, details_col = st.columns(
                [1, 2.7],
                gap="large",
            )

            with image_col:
                show_product_image(product["image"])

            with details_col:
                st.markdown(
                    f'<div class="product-title">{product["name"]}</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="product-meta">Category: '
                    f'<strong>{product["main_category"]}</strong></div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="price">₹{product["actual_price"]:,.0f}</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="product-meta">'
                    f'⭐ {product["ratings"]:.1f}/5 '
                    f'({product["no_of_ratings"]:,.0f} ratings)'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                metric1, metric2 = st.columns(2)

                with metric1:
                    st.markdown(
                        f"""
                        <div class="metric-box">
                            Text similarity<br>
                            <span class="metric-value">
                                {product["text_similarity"]:.3f}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with metric2:
                    st.markdown(
                        f"""
                        <div class="metric-box">
                            Ranking score<br>
                            <span class="metric-value">
                                {product["ranking_score"]:.3f}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Recommendation validation"):
            st.write(f"All products within budget: **{price_valid}**")
            st.write(f"All products in selected category: **{category_valid}**")
            st.write(f"All products meet relevance threshold: **{relevance_valid}**")
            st.write(f"Ranking order valid: **{ranking_valid}**")
            st.write(f"Minimum similarity threshold: **{MIN_SIMILARITY}**")

else:
    st.markdown(
        """
        <div class="section-card">
            <h3 style="color:#4A435D;">How to use</h3>
            <p style="color:#706A7D;">
                Enter a product request, choose an optional category,
                set your maximum budget, and click <b>Find Products</b>.
            </p>
            <p style="color:#706A7D;">
                Products with insufficient textual relevance are excluded,
                so the app does not fill the results with unrelated items.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div style="
        text-align:center;
        color:#817A8B;
        padding-top:1.5rem;
        font-size:0.85rem;
    ">
        Content-based recommendation • TF-IDF • Cosine Similarity
    </div>
    """,
    unsafe_allow_html=True,
)
