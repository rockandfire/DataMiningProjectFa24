import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

def load_card_data(filename='magic_cards.csv'):
    return pd.read_csv(filename)

def prepare_features(df):
    # Process colors
    color_map = {'W': 'White', 'U': 'Blue', 'B': 'Black', 'R': 'Red', 'G': 'Green'}
    df['colorIdentity'] = df['colorIdentity'].fillna('').apply(lambda x: [color_map.get(c.strip(), c.strip()) for c in x.split(',')] if isinstance(x, str) else [])
    mlb_colors = MultiLabelBinarizer()
    color_features = pd.DataFrame(
        mlb_colors.fit_transform(df['colorIdentity']),
        columns=mlb_colors.classes_,
        index=df.index
    )
    
    # Process keywords
    df['keywords'] = df['keywords'].fillna('').apply(lambda x: [kw.strip() for kw in x.split(',')] if isinstance(x, str) else [])
    mlb_keywords = MultiLabelBinarizer()
    keyword_features = pd.DataFrame(
        mlb_keywords.fit_transform(df['keywords']),
        columns=mlb_keywords.classes_,
        index=df.index
    )
    
    # Process supertypes
    df['is_legendary'] = df['supertypes'].fillna('').str.contains('Legendary')
    
    # Combine features
    features = pd.concat([color_features, keyword_features, df[['is_legendary']]], axis=1)
    
    return features, mlb_colors, mlb_keywords

def cluster_cards_kmeans(features, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(features)

def cluster_cards_dbscan(features, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(features)

def cluster_cards_agglomerative(features, n_clusters=3):
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    return agglomerative.fit_predict(features)

def evaluate_clustering(features, labels):
    return silhouette_score(features, labels)

def find_related_cards(df, features, legendary_card_name, top_n=10):
    legendary_card = df[df['name'] == legendary_card_name].index[0]
    legendary_features = features.iloc[legendary_card].values.reshape(1, -1)
    
    similarities = cosine_similarity(legendary_features, features)
    similar_indices = similarities[0].argsort()[::-1][1:top_n+1]
    
    return df.iloc[similar_indices]

# Main workflow
df = load_card_data()
features, mlb_colors, mlb_keywords = prepare_features(df)

# Normalize features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Perform clustering with different algorithms
kmeans_clusters = cluster_cards_kmeans(normalized_features)
dbscan_clusters = cluster_cards_dbscan(normalized_features)
agglomerative_clusters = cluster_cards_agglomerative(normalized_features)

# Evaluate clustering results
kmeans_score = evaluate_clustering(normalized_features, kmeans_clusters)
dbscan_score = evaluate_clustering(normalized_features, dbscan_clusters)
agglomerative_score = evaluate_clustering(normalized_features, agglomerative_clusters)

print("Clustering Evaluation (Silhouette Score):")
print(f"K-Means: {kmeans_score:.4f}")
print(f"DBSCAN: {dbscan_score:.4f}")
print(f"Agglomerative: {agglomerative_score:.4f}")

# Add cluster labels to the dataframe
df['kmeans_cluster'] = kmeans_clusters
df['dbscan_cluster'] = dbscan_clusters
df['agglomerative_cluster'] = agglomerative_clusters

# Print sample of clustered cards
print("\nSample of clustered cards:")
print(df[['name', 'colorIdentity', 'keywords', 'is_legendary', 'kmeans_cluster', 'dbscan_cluster', 'agglomerative_cluster']].head(20))

# Find related cards for a legendary creature
legendary_card_name = "Reya Dawnbringer"  # An example from your data
if legendary_card_name in df['name'].values:
    related_cards = find_related_cards(df, features, legendary_card_name)
    print(f"\nCards related to {legendary_card_name}:")
    print(related_cards[['name', 'colorIdentity', 'keywords', 'is_legendary']])
else:
    print(f"Legendary creature '{legendary_card_name}' not found in the database.")

# Print some statistics
print("\nCluster Statistics (K-Means):")
print(df['kmeans_cluster'].value_counts())

print("\nMost common keywords:")
print(df['keywords'].explode().value_counts().head(10))

print("\nColor distribution:")
print(df['colorIdentity'].explode().value_counts())