import streamlit as st
from streamlit_quill import st_quill
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModel
import io
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from openai import OpenAI
import torch
import re
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Télécharger les ressources nécessaires de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Configurer la mise en page de Streamlit en mode large
st.set_page_config(layout="wide")

# Charger le modèle d'embedding par défaut
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Charger le modèle de résumé
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

st.sidebar.title('Paramètres')

api_key = st.sidebar.text_input('OpenAI API Key', type='password')
embedding_source = st.sidebar.selectbox(
    'Source des Embeddings',
    options=["OpenAI", "Hugging Face"]
)

if embedding_source == "OpenAI":
    embedding_model = st.sidebar.selectbox(
        'OpenAI Embedding Model',
        options=["text-embedding-3-large",
                 "text-embedding-3-small", "text-embedding-ada-002"]
    )
    hf_model_name = None  # Définir hf_model_name par défaut pour éviter l'erreur
else:
    embedding_model = None  # Définir embedding_model par défaut pour éviter l'erreur
    hf_model_name = st.sidebar.text_input(
        'Hugging Face Model Name', value='sentence-transformers/all-MiniLM-L6-v2')

# Titre de l'application
st.title("Semantify")
content = st_quill()

# Champ de texte pour entrer le contenu
user_input = content

# Slider pour ajuster le seuil de similarité cosinus
similarity_threshold = st.slider(
    "Seuil de similarité cosinus", 0.0, 1.0, 0.5, 0.05)

# CSS pour le wrapping des phrases
st.markdown("""
    <style>
    .wrap-text {
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to get embeddings from OpenAI or Hugging Face
openai_client = OpenAI(api_key=api_key)


def get_embeddings(keywords, embedding_source, api_key=None, embedding_model=None, hf_model_name=None):
    if embedding_source == "OpenAI":
        response = openai_client.embeddings.create(
            input=keywords,
            model=embedding_model
        )
        return [embedding.embedding for embedding in response.data]
    elif embedding_source == "Hugging Face":
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        model = AutoModel.from_pretrained(hf_model_name)
        inputs = tokenizer(keywords, padding=True,
                           truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()


# Fonction pour nettoyer le texte sans supprimer les ponctuations
def preprocess_text(text):
    # Mise en minuscule
    text = text.lower()
    # Tokenisation en mots
    tokens = word_tokenize(text)
    # Suppression des stopwords et lemmatisation
    stop_words = set(stopwords.words('french'))
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = [lemmatizer.lemmatize(
        word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(cleaned_tokens)

# Fonction pour diviser le texte en morceaux plus petits


def chunk_text(text, max_length=1024):
    tokens = text.split()
    chunks = [' '.join(tokens[i:i + max_length])
              for i in range(0, len(tokens), max_length)]
    return chunks

# Fonction pour créer un fichier Excel en mémoire


def to_excel(df1, df2, df3):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df1.to_excel(writer, index=True, sheet_name='Phrases')
    df2.to_excel(writer, index=True, sheet_name='Paragraphes')
    df3.to_excel(writer, index=True, sheet_name='NGrams')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

# Fonction pour calculer les scores de densité et de dilution sémantique


def calculate_semantic_scores(similarities, threshold=0.5):
    mean_similarity = similarities.mean()
    # Proportion de segments avec une similarité < threshold
    dilution_score = (similarities < threshold).sum() / len(similarities)
    return mean_similarity, dilution_score

# Fonction pour générer des N-grams à partir d'un texte


def generate_ngrams(text, n):
    vectorizer = CountVectorizer(ngram_range=(n, n))
    analyzer = vectorizer.build_analyzer()
    ngrams = analyzer(text)
    return ngrams

# Fonction pour calculer la luminosité d'une couleur


def luminance(color):
    r, g, b = mcolors.hex2color(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

# Fonction pour surligner les paragraphes en fonction de leur similarité cosinus


def highlight_paragraphs(paragraphs, similarities):
    norm = mcolors.Normalize(vmin=min(similarities), vmax=max(similarities))
    cmap = plt.get_cmap('Reds')
    highlighted_paragraphs = []
    for para, sim in zip(paragraphs, similarities):
        color = mcolors.rgb2hex(cmap(norm(sim)))
        text_color = "white" if luminance(color) < 0.5 else "black"
        highlighted_paragraphs.append(
            f'<span style="background-color:{color}; color:{text_color}">{para}</span>')
    return highlighted_paragraphs

# Fonction pour surligner les phrases en fonction de leur similarité cosinus


def highlight_sentences(sentences, similarities):
    norm = mcolors.Normalize(vmin=min(similarities), vmax=max(similarities))
    cmap = plt.get_cmap('Blues')
    highlighted_sentences = []
    for sent, sim in zip(sentences, similarities):
        color = mcolors.rgb2hex(cmap(norm(sim)))
        text_color = "white" if luminance(color) < 0.5 else "black"
        highlighted_sentences.append(
            f'<span style="background-color:{color}; color:{text_color}">{sent}</span>')
    return highlighted_sentences

# Fonction pour surligner les mots en fonction de leur similarité cosinus


def highlight_text_by_words(text, word_similarities):
    words = word_tokenize(text)
    norm = mcolors.Normalize(
        vmin=min(word_similarities.values()), vmax=max(word_similarities.values()))
    cmap = plt.get_cmap('Greens')
    highlighted_words = []
    for word in words:
        cleaned_word = preprocess_text(word)
        if cleaned_word in word_similarities:
            sim = word_similarities[cleaned_word]
            color = mcolors.rgb2hex(cmap(norm(sim)))
            text_color = "white" if luminance(color) < 0.5 else "black"
            highlighted_words.append(
                f'<span style="background-color:{color}; color:{text_color}">{word}</span>')
        else:
            highlighted_words.append(word)
    return " ".join(highlighted_words)

# Mettre en cache les DataFrames


@st.cache_data
def process_content(user_input, similarity_threshold, embedding_source, api_key, embedding_model, hf_model_name):
    # Diviser le contenu en phrases en considérant \n comme fin de phrase sauf si précédé par une ponctuation
    user_input = re.sub(r'(?<![.!?;:])\n', '. ', user_input)
    original_sentences = sent_tokenize(user_input)
    # Nettoyer chaque phrase
    cleaned_sentences = [preprocess_text(sentence)
                         for sentence in original_sentences]

    # Obtenir les embeddings des phrases
    sentence_embeddings = get_embeddings(
        cleaned_sentences, embedding_source, api_key, embedding_model, hf_model_name)

    # Diviser le contenu en paragraphes
    original_paragraphs = user_input.split('\n')
    original_paragraphs = [para.strip()
                           for para in original_paragraphs if para]
    cleaned_paragraphs = [preprocess_text(
        paragraph) for paragraph in original_paragraphs]

    # Obtenir les embeddings des paragraphes
    paragraph_embeddings = get_embeddings(
        cleaned_paragraphs, embedding_source, api_key, embedding_model, hf_model_name)

    # Déterminer le sujet principal (embedding du contenu entier)
    main_topic_embedding = get_embeddings([preprocess_text(
        user_input)], embedding_source, api_key, embedding_model, hf_model_name)[0]

    # Calculer la similarité cosinus entre chaque phrase et le sujet principal
    sentence_similarities = util.pytorch_cos_sim(
        sentence_embeddings, main_topic_embedding).flatten()

    # Créer un DataFrame pour afficher les résultats des phrases
    df_sentences = pd.DataFrame({
        'Sentence': original_sentences,
        'Similarity': sentence_similarities
    })

    # Ajouter une colonne de classement
    df_sentences['Rank'] = df_sentences['Similarity'].rank(
        method='dense', ascending=False).astype(int)

    # Calculer les scores de densité et de dilution sémantique pour les phrases
    sentence_density, sentence_dilution = calculate_semantic_scores(
        sentence_similarities, threshold=similarity_threshold)

    # Calculer les scores de densité et de dilution sémantique pour les 25% premiers segments des phrases
    num_25_percent_sentences = max(1, len(sentence_similarities) // 4)
    sentence_density_25, sentence_dilution_25 = calculate_semantic_scores(
        sentence_similarities[:num_25_percent_sentences], threshold=similarity_threshold)

    # Trier les phrases par similarité croissante
    df_sorted_sentences = df_sentences.sort_values(by='Similarity')

    # Ajuster l'index pour qu'il commence à 1
    df_sorted_sentences.index = df_sorted_sentences.index + 1
    df_sorted_sentences.index.name = 'Index'

    # Ajouter une classe CSS pour le wrapping des phrases
    df_sorted_sentences['Sentence'] = df_sorted_sentences['Sentence'].apply(
        lambda x: f'<div class="wrap-text">{x}</div>')

    # Calculer la similarité cosinus entre chaque paragraphe et le sujet principal
    paragraph_similarities = util.pytorch_cos_sim(
        paragraph_embeddings, main_topic_embedding).flatten()

    # Surligner les paragraphes en fonction de leur similarité cosinus
    highlighted_paragraphs = highlight_paragraphs(
        original_paragraphs, paragraph_similarities)

    # Surligner les phrases en fonction de leur similarité cosinus
    highlighted_sentences = highlight_sentences(
        original_sentences, sentence_similarities)

    # Créer un DataFrame pour afficher les résultats des paragraphes
    df_paragraphs = pd.DataFrame({
        'Paragraph': original_paragraphs,
        'Similarity': paragraph_similarities
    })

    # Ajouter une colonne de classement
    df_paragraphs['Rank'] = df_paragraphs['Similarity'].rank(
        method='dense', ascending=False).astype(int)

    # Calculer les scores de densité et de dilution sémantique pour les paragraphes
    paragraph_density, paragraph_dilution = calculate_semantic_scores(
        paragraph_similarities, threshold=similarity_threshold)

    # Calculer les scores de densité et de dilution sémantique pour les 25% premiers segments des paragraphes
    num_25_percent_paragraphs = max(1, len(paragraph_similarities) // 4)
    paragraph_density_25, paragraph_dilution_25 = calculate_semantic_scores(
        paragraph_similarities[:num_25_percent_paragraphs], threshold=similarity_threshold)

    # Trier les paragraphes par similarité croissante
    df_sorted_paragraphs = df_paragraphs.sort_values(by='Similarity')

    # Ajuster l'index pour qu'il commence à 1
    df_sorted_paragraphs.index = df_sorted_paragraphs.index + 1
    df_sorted_paragraphs.index.name = 'Index'

    # Ajouter une classe CSS pour le wrapping des paragraphes
    df_sorted_paragraphs['Paragraph'] = df_sorted_paragraphs['Paragraph'].apply(
        lambda x: f'<div class="wrap-text">{x}</div>')

    # Générer des N-grams et calculer leur similarité cosinus avec le sujet principal
    ngrams = generate_ngrams(preprocess_text(user_input), 1)

    # Compter les occurrences des N-grams
    ngram_counts = pd.Series(ngrams).value_counts().reset_index()
    ngram_counts.columns = ['N-Gram', 'Occurrences']

    # Obtenir les embeddings des N-grams uniques
    unique_ngrams = ngram_counts['N-Gram'].tolist()
    ngram_embeddings = get_embeddings(
        unique_ngrams, embedding_source, api_key, embedding_model, hf_model_name)
    ngram_similarities = util.pytorch_cos_sim(
        ngram_embeddings, main_topic_embedding).flatten()

    # Créer un dictionnaire pour les similarités des mots
    word_similarities = dict(zip(unique_ngrams, ngram_similarities))

    # Surligner les mots en fonction de leur similarité cosinus dans le contenu entier
    highlighted_text = highlight_text_by_words(user_input, word_similarities)

    # Ajouter les similarités au DataFrame des N-grams
    ngram_counts['Similarity'] = ngram_similarities

    # Ajouter une colonne de classement
    ngram_counts['Rank'] = ngram_counts['Similarity'].rank(
        method='dense', ascending=False).astype(int)

    # Trier les N-grams par similarité croissante
    df_sorted_ngrams = ngram_counts.sort_values(by='Similarity')

    return (df_sorted_sentences, sentence_density, sentence_dilution, sentence_density_25, sentence_dilution_25), (df_sorted_paragraphs, paragraph_density, paragraph_dilution, paragraph_density_25, paragraph_dilution_25), df_sorted_ngrams, highlighted_paragraphs, highlighted_sentences, highlighted_text


# Bouton pour lancer l'analyse
if st.button("Entrer"):
    if user_input:
        # Traiter le contenu et obtenir les DataFrames et scores
        (df_sorted_sentences, sentence_density, sentence_dilution, sentence_density_25, sentence_dilution_25), (df_sorted_paragraphs, paragraph_density,
                                                                                                                paragraph_dilution, paragraph_density_25, paragraph_dilution_25), df_sorted_ngrams, highlighted_paragraphs, highlighted_sentences, highlighted_text = process_content(user_input, similarity_threshold, embedding_source, api_key, embedding_model, hf_model_name)

        # Disposition en colonnes
        col1, col2 = st.columns(2)
        # Afficher le DataFrame des phrases
        with col1:
            # Afficher le contenu surligné sous l'éditeur de texte
            st.markdown("<h3>Contenu Surligné par Paragraphes</h3>",
                        unsafe_allow_html=True)
            st.markdown("".join(highlighted_paragraphs),
                        unsafe_allow_html=True)
            st.markdown("<h3>Analyse des Phrases</h3>", unsafe_allow_html=True)
            st.markdown(df_sorted_sentences.to_html(
                escape=False), unsafe_allow_html=True)
            st.write(f"Densité sémantique : {sentence_density:.2f}")
            st.write(f"Dilution sémantique : {sentence_dilution:.2f}")
            st.write(
                f"Densité sémantique (25% premiers) : {sentence_density_25:.2f}")
            st.write(
                f"Dilution sémantique (25% premiers) : {sentence_dilution_25:.2f}")

        # Afficher le DataFrame des paragraphes
        with col2:
            st.markdown("<h3>Contenu Surligné par Phrases</h3>",
                        unsafe_allow_html=True)
            st.markdown("".join(highlighted_sentences), unsafe_allow_html=True)
            st.markdown("<h3>Contenu Surligné par Mots</h3>",
                        unsafe_allow_html=True)
            st.markdown(highlighted_text, unsafe_allow_html=True)

            st.markdown("<h3>Analyse des Paragraphes</h3>",
                        unsafe_allow_html=True)
            st.markdown(df_sorted_paragraphs.to_html(
                escape=False), unsafe_allow_html=True)
            st.write(f"Densité sémantique : {paragraph_density:.2f}")
            st.write(f"Dilution sémantique : {paragraph_dilution:.2f}")
            st.write(
                f"Densité sémantique (25% premiers) : {paragraph_density_25:.2f}")
            st.write(
                f"Dilution sémantique (25% premiers) : {paragraph_dilution_25:.2f}")

        # Afficher le DataFrame des N-grams
        st.markdown("<h3>Analyse des N-Grams</h3>", unsafe_allow_html=True)
        st.markdown(df_sorted_ngrams.to_html(
            escape=False), unsafe_allow_html=True)

        # Ajouter un bouton de téléchargement pour les DataFrames au format Excel
        excel_data = to_excel(df_sorted_sentences,
                              df_sorted_paragraphs, df_sorted_ngrams)
        st.download_button(label="Télécharger les résultats au format Excel",
                           data=excel_data,
                           file_name='analyse_contenu.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    else:
        st.write("Veuillez entrer un contenu avant de cliquer sur 'Entrer'.")
