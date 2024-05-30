import numpy as np
from openai import OpenAI
import io
from openpyxl.styles import PatternFill
from openpyxl import load_workbook
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import networkx as nx


st.set_page_config(layout='wide')

# Sidebar for user inputs
st.sidebar.title('Paramètres')

api_key = st.sidebar.text_input('OpenAI API Key', type='password')
embedding_model = st.sidebar.selectbox(
    'Embedding Model',
    options=["text-embedding-3-small",
             "text-embedding-3-large", "text-embedding-ada-002"]
)
clustering_model = st.sidebar.selectbox(
    "Choisissez le modèle de clustering",
    options=["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo", "gpt-4"]
)
eps = st.sidebar.slider('eps value for DBSCAN', 0.0, 1.0, 0.3)
min_samples = st.sidebar.slider('min_samples value for DBSCAN', 1, 100, 5)
cluster_label_prompt = st.sidebar.text_area(
    'Default Prompt for Cluster Labeling',
    'Labellise la liste de mots-clés suivante en lui donnant un nom concis en français de deux ou trois mots (ne fais aucun commentaire dans ta réponse) : '
)

# Explications des paramètres
st.sidebar.markdown("### Explication des paramètres")
st.sidebar.markdown("""
- **eps** : La distance maximale entre deux échantillons pour qu'ils soient considérés comme faisant partie du même cluster. Plus la valeur de eps est grande, plus les clusters seront grands.
- **min_samples** : Le nombre minimum d'échantillons dans un voisinage pour qu'un point soit considéré comme un point central. Plus la valeur de min_samples est grande, plus les clusters seront denses.
""")

# Function to get embeddings from OpenAI
openai_client = OpenAI(api_key=api_key)


def get_embeddings(keywords):
    response = openai_client.embeddings.create(
        model=embedding_model,
        input=keywords,
    )
    return [embedding.embedding for embedding in response.data]

# Function to read the Excel file


def read_excel(file):
    return pd.read_excel(file)

# Function to perform clustering and add columns


def add_cluster_columns(df, keyword_column):
    keywords = df[keyword_column].tolist()

    # Get embeddings from OpenAI
    embeddings = get_embeddings(keywords)
    vectors = np.array(embeddings)

    # Calculate cosine similarity
    cosine_sim_matrix = cosine_similarity(vectors)

    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    clustering.fit(vectors)
    df['Cluster'] = clustering.labels_

    cluster_names = []
    valid_clusters = []
    cluster_centers = {}
    noise_points = []

    for cluster in set(clustering.labels_):
        if cluster == -1:
            # Collect noise points
            noise_indices = df[df['Cluster'] == cluster].index.tolist()
            noise_points.extend(noise_indices)
            continue  # Ignore noise points
        cluster_indices = df[df['Cluster'] == cluster].index.tolist()
        cluster_keywords = [keywords[idx] for idx in cluster_indices]

        # Calculate cluster center
        cluster_center = np.mean([vectors[idx]
                                 for idx in cluster_indices], axis=0)
        cluster_centers[cluster] = cluster_center

        # Check if the cluster has at least 5 keywords
        if len(cluster_keywords) >= 5:
            valid_clusters.append(cluster)
            # Sort keywords by similarity to cluster center and take the top 50
            similarities = [(keywords[idx], cosine_similarity([vectors[idx]], [
                             cluster_center])[0][0]) for idx in cluster_indices]
            sorted_keywords = [keyword for keyword, similarity in sorted(
                similarities, key=lambda x: x[1], reverse=True)]
            cluster_name = get_cluster_name(sorted_keywords, max_keywords=50)
            cluster_names.append(cluster_name)

    # Clean cluster names
    cluster_names = [name.strip().replace('"', '').replace("'", '')
                     for name in cluster_names]

    # Create the new DataFrame with the required format
    max_length = max([len(df[df['Cluster'] == cluster])
                     for cluster in valid_clusters] + [len(noise_points)])
    new_data = {name: [''] * max_length for name in cluster_names}
    new_data.update(
        {f"Similarity {i+1}": [''] * max_length for i in range(len(valid_clusters))})
    new_data['Noise'] = [''] * max_length

    cluster_counts = {i: 0 for i in valid_clusters}
    for i, cluster in enumerate(valid_clusters):
        cluster_name = cluster_names[i]
        cluster_indices = df[df['Cluster'] == cluster].index.tolist()
        cluster_center = np.mean([vectors[idx]
                                 for idx in cluster_indices], axis=0)

        similarities = []
        for idx in cluster_indices:
            similarity = cosine_similarity(
                [vectors[idx]], [cluster_center])[0][0] * 100
            similarities.append((keywords[idx], similarity))

        # Sort the keywords in the cluster by similarity in descending order
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

        for idx, (keyword, similarity) in enumerate(similarities):
            new_data[cluster_name][idx] = keyword
            new_data[f"Similarity {i+1}"][idx] = similarity
            cluster_counts[cluster] += 1

    # Add noise points to the DataFrame
    for idx, noise_idx in enumerate(noise_points):
        new_data['Noise'][idx] = keywords[noise_idx]

    new_df = pd.DataFrame(new_data)

    # Convert all object columns to strings
    for col in new_df.select_dtypes(include=['object']).columns:
        new_df[col] = new_df[col].astype(str)

    # Convert similarity columns to numeric
    for col in new_df.columns:
        if 'Similarity' in col:
            new_df[col] = pd.to_numeric(new_df[col], errors='coerce')

    # Calculate cluster similarities and create DataFrame
    cluster_similarities = []
    for i, cluster1 in enumerate(valid_clusters):
        for j, cluster2 in enumerate(valid_clusters):
            if i != j:
                similarity = cosine_similarity([cluster_centers[cluster1]], [
                                               cluster_centers[cluster2]])[0][0] * 100
                cluster_similarities.append(
                    [cluster_names[i], cluster_names[j], similarity])

    cluster_sim_df = pd.DataFrame(cluster_similarities, columns=[
                                  'Cluster 1', 'Cluster 2', 'Cosine Similarity'])

    return new_df, cluster_names, valid_clusters, cosine_sim_matrix, vectors, cluster_sim_df

# Function to get cluster name using OpenAI


def get_cluster_name(keywords, max_keywords=50):
    prompt = cluster_label_prompt + ', '.join(keywords[:max_keywords])
    response = openai_client.chat.completions.create(
        model=clustering_model,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    cluster_name = response.choices[0].message.content.strip()
    return cluster_name

# Function to write the modified DataFrame to an Excel file


def write_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    processed_data = output.getvalue()
    return processed_data

# Function to create an interactive knowledge graph using streamlit-agraph


def create_knowledge_graph(cluster_names, valid_clusters, cosine_sim_matrix, df, keyword_column, vectors):
    G = nx.DiGraph()

    # Add nodes for clusters
    for cluster_name in cluster_names:
        G.add_node(cluster_name, label=cluster_name, node_type='cluster')

    # Determine the most representative cluster (the one with the most keywords)
    representative_cluster = max(
        valid_clusters, key=lambda cluster: len(df[df['Cluster'] == cluster]))

    # Add nodes and edges for keywords within clusters
    for cluster_name, cluster in zip(cluster_names, valid_clusters):
        cluster_indices = df[df['Cluster'] == cluster].index.tolist()
        cluster_center = np.mean([vectors[idx]
                                 for idx in cluster_indices], axis=0)
        similarities = []
        for idx in cluster_indices:
            similarity = cosine_similarity(
                [vectors[idx]], [cluster_center])[0][0] * 100
            keyword = df.iloc[idx][keyword_column]
            similarities.append((keyword, similarity))

        # Sort the keywords in the cluster by similarity in descending order
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

        # Only include the top keyword in the graph
        if similarities:
            keyword, similarity = similarities[0]
            label = f"{keyword} ({similarity:.2f}%)"
            color = 'red' if cluster == representative_cluster else 'blue'
            G.add_node(keyword, label=label, node_type='keyword', color=color)
            G.add_edge(cluster_name, keyword)

    # Add edges between clusters based on cosine similarity
    for i, cluster1 in enumerate(valid_clusters):
        for j, cluster2 in enumerate(valid_clusters):
            if i != j:
                similarity = cosine_similarity([cosine_sim_matrix[cluster1]], [
                                               cosine_sim_matrix[cluster2]])[0][0]
                if similarity > 0.7:  # You can adjust this threshold as needed
                    G.add_edge(cluster_names[i], cluster_names[j],
                               weight=similarity, label=f'{similarity:.2f}%')

    # Convert the NetworkX graph into a list of nodes and edges for streamlit-agraph
    nodes = [Node(id=node, label=G.nodes[node]['label'], size=25 if G.nodes[node]['node_type'] == 'cluster' else 15,
                  color=G.nodes[node].get('color', 'green')) for node in G.nodes]
    edges = [Edge(source=edge[0], target=edge[1],
                  label=G.edges[edge].get('label', '')) for edge in G.edges]

    # Create a configuration for the graph
    config = Config(
        width=2000,
        height=1000,
        directed=True,
        nodeHighlightBehavior=True,
        # This makes the graph static but allows dragging and dropping
        staticGraphWithDragAndDrop=True,
        physics=False,  # Disables the physics engine for more stability
    )

    return nodes, edges, config


# Streamlit app
st.title('Keywords Clustering')

with st.expander("À propos"):
    st.markdown("""
    Cette application a été créée sur la base d'un script développé par [Loïc Hélias](https://www.linkedin.com/in/loichelias/), leader SEO chez Décathlon, accessible [ici](https://github.com/lassomontana/check-keyword-similarity).
    ### À quoi sert cette application ?
    Cette application permet de traiter et de regrouper une liste de mots-clés en fonction de leur similarité sémantique. Elle utilise l'API OpenAI pour générer des embeddings et l'algorithme DBSCAN pour effectuer le clustering.

    ### Comment utiliser cette application ?
    1. **Téléchargez un fichier Excel** contenant les contenus textuels à analyser.
    2. **Sélectionnez la colonne** contenant les contenus textuels dans le fichier Excel.
    3. **Configurez les paramètres** dans la barre latérale :
        - Entrez votre clé API OpenAI.
        - Choisissez les modèles d'embedding et de clustering.
        - Définissez les valeurs de `eps` et `min_samples` pour l'algorithme DBSCAN.
        - Définissez la taille des chunks de contenu.
    4. **Cliquez sur "Lancer"** pour démarrer le traitement.
    5. Consultez et téléchargez les résultats :
        - Fichier Excel des données traitées.
        - Similarité des clusters sous forme de DataFrame.
        - Graphique de connaissance interactif des clusters.
    """)

uploaded_file = st.file_uploader("Upload an Excel file", type="xlsx")
if uploaded_file is not None:
    df = read_excel(uploaded_file)
    st.dataframe(df, use_container_width=True)

    keyword_column = st.selectbox(
        "Select the column containing keywords", options=df.columns)

    if st.button("Lancer"):
        new_df, cluster_names, valid_clusters, cosine_sim_matrix, vectors, cluster_sim_df = add_cluster_columns(
            df, keyword_column)
        st.session_state['new_df'] = new_df
        st.session_state['cluster_names'] = cluster_names
        st.session_state['valid_clusters'] = valid_clusters
        st.session_state['cosine_sim_matrix'] = cosine_sim_matrix
        st.session_state['vectors'] = vectors
        st.session_state['cluster_sim_df'] = cluster_sim_df

    if 'new_df' in st.session_state:
        st.write("Processed Data")
        st.dataframe(st.session_state['new_df'])

        processed_file = write_excel(st.session_state['new_df'])

        st.download_button(
            label="Download Processed Excel",
            data=processed_file,
            file_name="processed_keywords.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.write("Cluster Cosine Similarity DataFrame")
        st.dataframe(
            st.session_state['cluster_sim_df'], use_container_width=True)

        similarity_file = write_excel(st.session_state['cluster_sim_df'])

        st.download_button(
            label="Download Cosine Similarity Excel",
            data=similarity_file,
            file_name="cosine_similarity.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.write("Creating knowledge graph...")
        nodes, edges, config = create_knowledge_graph(
            st.session_state['cluster_names'], st.session_state['valid_clusters'], st.session_state['cosine_sim_matrix'], df, keyword_column, st.session_state['vectors'])
        agraph(nodes=nodes, edges=edges, config=config)
