# Product Finder App

This Streamlit app is built from the supplied products_cleaned.csv dataset and follows the recommendation methodology developed.

## Features

Pastel-colored user interface
Natural-language product search
Category filtering
Maximum-budget filtering
Configurable number of recommendations
TF-IDF + cosine similarity
70/20/10 weighted heuristic ranking
Minimum relevance gate to prevent zero/near-zero relevance results
Handling of broken/missing product-image URLs
Product image fallback displaying Image not found
Recommendation validation panel

## Recommendation Logic

The app follows the notebook's content-based approach:

1. Clean product names and categories.
2. Build product text.
3. Create TF-IDF vectors using unigrams and bigrams.
4. Calculate cosine similarity between the query and products.
5. Apply maximum-price and category constraints.
6. Remove products below the minimum text-similarity threshold.
7. Rank remaining products using:
   70% text similarity
   20% normalized rating
   10% normalized popularity

The ranking weights are manually specified heuristics and are not learned from customer interaction history.

## Relevance Protection

The app does not return zero-similarity products. It also applies a small minimum similarity threshold MIN_SIMILARITY = 0.03 so that the recommendation list is not artificially filled when no relevant product matches the query.

If no product meets the relevance and shopping constraints, the app displays a clear no-results message.

## Image Handling

The app downloads product images server-side. If an image URL is empty, broken, times out, or cannot be decoded as an image, the product card displays:

Image not found

instead of showing a broken-image element.

## Limitation

This remains a content-based recommendation system. The dataset does not contain customer-level user-item interaction history, so it does not learn personalized user preferences.

## Url for the app
https://shopping-assistant-app.streamlit.app/
