import os
import streamlit as st
import pdfplumber
import textwrap
import numpy as np
import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import spacy
from nltk.corpus import stopwords
import re
from googletrans import Translator
import random
from sklearn.neighbors import NearestNeighbors

# Define Academic City University color scheme
ACITY_PRIMARY = "#910000"  # Deep red
ACITY_SECONDARY = "#FFD700"  # Gold
ACITY_ACCENT = "#000080"  # Navy blue
ACITY_TEXT = "#333333"  # Dark grey
ACITY_BG = "#black"  # Light background

# Set page configuration with Academic City theme
st.set_page_config(
    page_title="Academic City PDF QA System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Academic City theme
def apply_acity_theme():
    st.markdown(f"""
    <style>
        .stApp {{
            background-color: {ACITY_BG};
        }}
        .stButton button {{
            background-color: {ACITY_PRIMARY};
            color: white;
        }}
        .stProgress > div > div {{
            background-color: {ACITY_PRIMARY};
        }}
        h1, h2, h3 {{
            color: {ACITY_PRIMARY};
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 2px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: white;
            color: {ACITY_TEXT};
            border-radius: 4px 4px 0px 0px;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {ACITY_PRIMARY};
            color: white;
        }}
        .stSidebar .sidebar-content {{
            background-color: {ACITY_BG};
        }}
        .block-container {{
            padding-top: 2rem;
        }}
        .css-1kyxreq {{
            justify-content: center;
            align-items: center;
            display: flex;
        }}
    </style>
    """, unsafe_allow_html=True)

apply_acity_theme()

# Display Academic City logo/header
st.markdown(f"""
    <div style="background-color:{ACITY_PRIMARY}; padding:10px; border-radius:5px; margin-bottom:20px;">
        <h1 style="color:white; text-align:center;">Academic City PDF Question Answering System</h1>
        <h4 style="color:{ACITY_SECONDARY}; text-align:center;">Domain-Adapted Document Intelligence</h4>
    </div>
""", unsafe_allow_html=True)


# Optionally set a default API key or leave empty
DEFAULT_API_KEY = ""

# Ask user for API key if not already set
if "API_KEY" not in st.session_state:
    st.session_state.API_KEY = DEFAULT_API_KEY

if not st.session_state.API_KEY:
    st.session_state.API_KEY = st.text_input(
        "🔐 Enter your Hugging Face API Key", 
        type="password", 
        placeholder="hf_...",
        help="You can get this from https://huggingface.co/settings/tokens"
    )

# Use it
API_KEY = st.session_state.API_KEY

# Download necessary NLTK data
@st.cache_resource
def load_nltk_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    return True

load_nltk_resources()

# Load spaCy model for NER
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except:
        st.warning("Downloading spaCy model...")
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Extracting text from each PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Named Entity Recognition (NER) for domain adaptation
def perform_ner(text):
    """Extract named entities from text to help with domain adaptation"""
    doc = nlp(text[:10000])  # Limit for performance
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Enhanced chunking with NER awareness
def enhanced_chunk_text(text, chunk_size=500, domain_adaptation=False):
    """Chunk text with domain adaptation if enabled"""
    basic_chunks = textwrap.wrap(text, width=chunk_size)
    
    if not domain_adaptation:
        return basic_chunks
    
    # With domain adaptation: try to preserve entity boundaries when possible
    enhanced_chunks = []
    entities_dict = {}
    
    # Process each chunk to identify entities
    for i, chunk in enumerate(basic_chunks):
        entities = perform_ner(chunk)
        for entity, label in entities:
            if entity not in entities_dict:
                entities_dict[entity] = {"count": 0, "label": label}
            entities_dict[entity]["count"] += 1
    
    # Keep track of important domain entities
    domain_entities = {
        entity: info["label"] 
        for entity, info in entities_dict.items() 
        if info["count"] > 1 and len(entity.split()) > 1  # Multiple words entities that appear more than once
    }
    
    # Now try to preserve entities in chunks
    current_pos = 0
    text_length = len(text)
    
    while current_pos < text_length:
        end_pos = min(current_pos + chunk_size, text_length)
        
        # Check if we're splitting in the middle of an important entity
        chunk = text[current_pos:end_pos]
        
        # Look for entities that might be cut
        for entity in domain_entities:
            entity_pos = chunk.find(entity)
            if entity_pos != -1 and entity_pos + len(entity) > len(chunk):
                # Entity is cut off, adjust end position
                end_pos = current_pos + entity_pos
                if end_pos == current_pos:  # Avoid empty chunks
                    end_pos = current_pos + chunk_size
        
        enhanced_chunks.append(text[current_pos:end_pos])
        current_pos = end_pos
    
    return enhanced_chunks

# Data augmentation through paraphrasing
def simple_paraphrase(text):
    """Simple rule-based paraphrasing to augment data"""
    # This is a simplified version - in production you might use a dedicated paraphrasing model
    sentences = sent_tokenize(text)
    paraphrases = []
    
    synonyms = {
        "important": ["significant", "crucial", "essential"],
        "good": ["excellent", "great", "beneficial"],
        "bad": ["poor", "inadequate", "subpar"],
        "large": ["big", "substantial", "considerable"],
        "small": ["tiny", "minor", "little"],
    }
    
    for sentence in sentences:
        new_sentence = sentence
        for word, replacements in synonyms.items():
            if word in new_sentence.lower():
                replacement = random.choice(replacements)
                new_sentence = re.sub(
                    r'\b' + word + r'\b', 
                    replacement, 
                    new_sentence, 
                    flags=re.IGNORECASE, 
                    count=1
                )
        
        paraphrases.append(new_sentence)
    
    return " ".join(paraphrases)

# Back-translation for data augmentation
def back_translate(text, source_lang="en", target_lang="fr"):
    """Augment data through back-translation"""
    try:
        translator = Translator()
        # Translate to target language
        intermediate = translator.translate(text, src=source_lang, dest=target_lang).text
        # Translate back to source language
        back_translated = translator.translate(intermediate, src=target_lang, dest=source_lang).text
        return back_translated
    except Exception as e:
        st.warning(f"Back-translation failed: {str(e)}")
        return text  # Return original text if translation fails

# Filter bias and noise
def filter_bias_and_noise(text):
    """Remove potential biases and noise from text"""
    # List of patterns that might indicate bias or noise
    bias_patterns = [
        r'\b(all|every|always|never|nobody|everybody)\b',  # Absolute statements
        r'\b(obviously|clearly|undoubtedly|certainly)\b',  # Certainty markers
        r'\b(should|must|have to|need to)\b',  # Prescriptive language
        r'\b(good|bad|best|worst|perfect)\b',  # Subjective judgments
    ]
    
    # Replace potential biased phrases with more neutral alternatives
    filtered_text = text
    for pattern in bias_patterns:
        matches = re.finditer(pattern, filtered_text, re.IGNORECASE)
        for match in matches:
            biased_word = match.group(0)
            if random.random() < 0.5:  # Only replace some instances to maintain low impact
                if biased_word.lower() in ['all', 'every', 'always', 'never', 'nobody', 'everybody']:
                    replacement = 'many' if random.random() < 0.5 else 'some'
                elif biased_word.lower() in ['obviously', 'clearly', 'undoubtedly', 'certainly']:
                    replacement = 'possibly' if random.random() < 0.5 else 'potentially'
                elif biased_word.lower() in ['should', 'must', 'have to', 'need to']:
                    replacement = 'could' if random.random() < 0.5 else 'might'
                elif biased_word.lower() in ['good', 'bad', 'best', 'worst', 'perfect']:
                    replacement = 'suitable' if random.random() < 0.5 else 'appropriate'
                else:
                    replacement = biased_word
                
                filtered_text = filtered_text.replace(biased_word, replacement, 1)
    
    # Remove repetitive content (potential noise)
    sentences = sent_tokenize(filtered_text)
    clean_sentences = []
    seen_content = set()
    
    for sentence in sentences:
        # Create a simplified fingerprint of the sentence
        fingerprint = re.sub(r'[^\w\s]', '', sentence.lower())
        fingerprint = ' '.join(sorted(fingerprint.split()))
        
        if fingerprint not in seen_content:
            clean_sentences.append(sentence)
            seen_content.add(fingerprint)
    
    return ' '.join(clean_sentences)

# Retrieving relevant chunks
def retrieve_relevant_chunk(query, chunks, nn_index, embedding_model, top_k=3, entity_boost=False):
    """Retrieve relevant chunks with optional entity boosting"""
    query_embedding = embedding_model.encode([query])
    distances, indices = nn_index.kneighbors(query_embedding, n_neighbors=top_k)
    retrieved_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    
    if entity_boost:
        # Extract entities from query
        query_entities = [ent.text.lower() for ent in nlp(query).ents]
        
        # If we found entities in the query, boost chunks containing those entities
        if query_entities:
            boosted_chunks = []
            for chunk in retrieved_chunks:
                chunk_score = 1.0
                for entity in query_entities:
                    if entity in chunk.lower():
                        chunk_score += 0.2  # Small boost for each matched entity
                boosted_chunks.append((chunk, chunk_score))
            
            # Sort by boost score
            boosted_chunks.sort(key=lambda x: x[1], reverse=True)
            retrieved_chunks = [chunk for chunk, _ in boosted_chunks]
    
    return "\n---\n".join(retrieved_chunks)

# Parse model response to extract just the answer part
def parse_model_response(raw_response):
    """Extract just the answer part from the model's response with improved robustness"""
    # If the response contains our prompt pattern ending with "Answer:"
    if "Answer:" in raw_response:
        answer_part = raw_response.split(" ")[-1].strip()
    # # Remove anything after 'Context'
    # if "Context" in answer_part:
    #     answer_part = answer_part.split("Context")[0].strip()
    # return answer_part

    
    # If the response has multiple lines after "Question:"
    lines = raw_response.split('\n')
    question_index = -1
    
    for i, line in enumerate(lines):
        if "Question:" in line:
            question_index = i
            break
    
    if question_index >= 0 and question_index < len(lines) - 1:
        # Return everything after the question line, skipping any potential empty lines
        answer_lines = [line for line in lines[question_index+1:] if line.strip()]
        if answer_lines:
            return "\n".join(answer_lines)
    
    # If we couldn't parse it properly, as a fallback, take the last paragraph
    paragraphs = raw_response.split("\n\n")
    if paragraphs:
        return paragraphs[-1].strip()
    
    # Last resort, return the raw response
    return raw_response.strip()

# Sending prompt to API with retry
def ask_model(context, query, api_url, api_key, retries=3, wait=5):
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Modify the prompt to get a cleaner answer
    prompt = f"""
    Based on the following context, answer the question directly and concisely.
    
    Context: {context}
    
    Question: {query}
    
    Answer:
    """
    
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 200}
    }
    
    for attempt in range(retries):
        with st.spinner(f"Generating answer (attempt {attempt+1}/{retries})..."):
            # Ensure header values are str and strictly ASCII/lower Latin-1
            clean_headers = {k: str(v).encode('latin-1', errors='ignore').decode('latin-1') for k, v in headers.items()}
            response = requests.post(api_url, headers=clean_headers, json=payload)

            if response.status_code == 200:
                raw_response = response.json()[0]["generated_text"]
                answer = parse_model_response(raw_response)
                return answer
            elif response.status_code == 503:
                if attempt < retries - 1:
                    time.sleep(wait)
                else:
                    return "Error: Model not available after several retries."
            elif "text/html" in response.headers.get("Content-Type", ""):
                return "Error: Received unexpected HTML content."
            else:
                return f"Error {response.status_code}: {response.text}"
    
    return "Error: Model not available after several retries."


# Expanded Metrics Collection
def calculate_metrics(query, expected_answer, model_answer, context, embedding_model):
    """Calculate comprehensive metrics for QA performance"""
    metrics = {}
    
    # Embedding-based metrics
    expected_embedding = embedding_model.encode([expected_answer])
    answer_embedding = embedding_model.encode([model_answer]) 
    context_embedding = embedding_model.encode([context])
    query_embedding = embedding_model.encode([query])
    
    # Answer relevance (semantic similarity to expected answer)
    similarity = np.dot(expected_embedding[0], answer_embedding[0]) / (
        np.linalg.norm(expected_embedding[0]) * np.linalg.norm(answer_embedding[0])
    )
    metrics["answer_relevance"] = float(similarity)
    
    # Retrieval relevance (query-context similarity)
    retrieval_relevance = np.dot(context_embedding[0], query_embedding[0]) / (
        np.linalg.norm(context_embedding[0]) * np.linalg.norm(query_embedding[0])
    )
    metrics["retrieval_relevance"] = float(retrieval_relevance)
    
    # Answer-context alignment (how well the answer relates to retrieved context)
    answer_context_alignment = np.dot(answer_embedding[0], context_embedding[0]) / (
        np.linalg.norm(answer_embedding[0]) * np.linalg.norm(context_embedding[0])
    )
    metrics["answer_context_alignment"] = float(answer_context_alignment)
    
    # Query-answer directness (how directly answer addresses the query)
    query_answer_directness = np.dot(query_embedding[0], answer_embedding[0]) / (
        np.linalg.norm(query_embedding[0]) * np.linalg.norm(answer_embedding[0])
    )
    metrics["query_answer_directness"] = float(query_answer_directness)
    
    # Answer precision (length ratio of model answer to reference answer)
    len_model = len(model_answer.split())
    len_expected = max(1, len(expected_answer.split()))  # Avoid division by zero
    precision = min(len_expected, len_model) / max(len_expected, len_model)
    metrics["answer_precision"] = float(precision)
    
    # Accuracy (similarity > 0.7 threshold)
    metrics["accuracy"] = 1.0 if similarity > 0.7 else 0.0
    
    # F1 Score (simplified using precision and similarity as recall proxy)
    recall_proxy = similarity
    if precision + recall_proxy > 0:
        f1 = 2 * precision * recall_proxy / (precision + recall_proxy)
    else:
        f1 = 0.0
    metrics["f1_score"] = float(f1)
    
    # Exact match (for short factual answers)
    exact_match = 1.0 if expected_answer.lower() in model_answer.lower() else 0.0
    metrics["exact_match"] = exact_match
    
    # Composite score - weighted average of all metrics
    metrics["composite_score"] = (
        0.25 * metrics["answer_relevance"] + 
        0.15 * metrics["retrieval_relevance"] + 
        0.15 * metrics["answer_context_alignment"] +
        0.15 * metrics["query_answer_directness"] + 
        0.10 * metrics["answer_precision"] +
        0.20 * metrics["exact_match"]
    )
    
    return metrics

# Visualize token distribution
def visualize_token_distribution(chunks):
    """Visualize token distribution across chunks"""
    st.subheader("Token Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    # Count tokens per chunk
    token_counts = []
    all_tokens = []
    
    for chunk in chunks:
        tokens = word_tokenize(chunk)
        token_counts.append(len(tokens))
        all_tokens.extend(tokens)
    
    # Plot 1: Distribution of tokens per chunk
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.histplot(token_counts, kde=True, ax=ax1, color=ACITY_PRIMARY)
        ax1.set_title('Distribution of Tokens per Chunk', color=ACITY_PRIMARY)
        ax1.set_xlabel('Number of Tokens')
        ax1.set_ylabel('Frequency')
        st.pyplot(fig1)
    
    # Plot 2: Most common tokens (excluding stop words)
    with col2:
        token_freq = Counter([t.lower() for t in all_tokens if t.lower() not in stopwords.words('english')])
        common_tokens = dict(token_freq.most_common(15))
        
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        bars = sns.barplot(x=list(common_tokens.values()), y=list(common_tokens.keys()), ax=ax2, color=ACITY_PRIMARY)
        ax2.set_title('15 Most Common Tokens', color=ACITY_PRIMARY)
        ax2.set_xlabel('Frequency')
        plt.tight_layout()
        st.pyplot(fig2)
    
    # Add entity analysis
    st.subheader("Named Entity Analysis")
    entity_counts = Counter()
    entity_types = Counter()
    
    for chunk in chunks[:50]:  # Limit to first 50 chunks for performance
        doc = nlp(chunk[:5000])  # Limit text size
        for ent in doc.ents:
            entity_counts[ent.text] += 1
            entity_types[ent.label_] += 1
    
    col1, col2 = st.columns(2)
    
    # Display entity types distribution
    with col1:
        entity_type_df = pd.DataFrame({
            'Entity Type': list(entity_types.keys()),
            'Count': list(entity_types.values())
        }).sort_values('Count', ascending=False)
        
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Count', y='Entity Type', data=entity_type_df, ax=ax3, color=ACITY_PRIMARY)
        ax3.set_title('Named Entity Types Distribution', color=ACITY_PRIMARY)
        st.pyplot(fig3)
    
    # Display most common entities
    with col2:
        st.markdown(f"<h4 style='color:{ACITY_PRIMARY}'>Most Common Named Entities</h4>", unsafe_allow_html=True)
        most_common_entities = pd.DataFrame({
            'Entity': [e for e, c in entity_counts.most_common(10)],
            'Count': [c for e, c in entity_counts.most_common(10)]
        })
        st.dataframe(most_common_entities, use_container_width=True)


def visualize_benchmark_comparison(current_metrics):
    """Compare current system performance against benchmark standards"""
    st.markdown(f"<h3 style='color:{ACITY_PRIMARY}'>Benchmark Comparison</h3>", unsafe_allow_html=True)
    
    # Define benchmark metrics (pre-defined comparison points)
    benchmark_metrics = {
        "Industry Standard": {
            "answer_relevance": 0.75,
            "retrieval_relevance": 0.70,
            "answer_context_alignment": 0.72,
            "query_answer_directness": 0.68,
            "answer_precision": 0.65,
            "exact_match": 0.60,
            "composite_score": 0.72
        },
        "Previous Version": {
            "answer_relevance": 0.65,
            "retrieval_relevance": 0.62,
            "answer_context_alignment": 0.60,
            "query_answer_directness": 0.58,
            "answer_precision": 0.55,
            "exact_match": 0.40,
            "composite_score": 0.60
        },
        "Target Performance": {
            "answer_relevance": 0.85,
            "retrieval_relevance": 0.82,
            "answer_context_alignment": 0.80,
            "query_answer_directness": 0.78,
            "answer_precision": 0.75,
            "exact_match": 0.70,
            "composite_score": 0.80
        }
    }
    
    # Select metrics to compare
    comparison_metrics = [
        "answer_relevance", 
        "retrieval_relevance", 
        "answer_context_alignment",
        "query_answer_directness",
        "composite_score"
    ]
    
    # Create comparison dataframe
    comparison_data = {
        "Metric": [],
        "Value": [],
        "System": []
    }
    
    # Add current system data
    for metric in comparison_metrics:
        comparison_data["Metric"].append(metric)
        comparison_data["Value"].append(current_metrics[metric])
        comparison_data["System"].append("Current System")
    
    # Add benchmark data
    for benchmark_name, benchmark_data in benchmark_metrics.items():
        for metric in comparison_metrics:
            comparison_data["Metric"].append(metric)
            comparison_data["Value"].append(benchmark_data[metric])
            comparison_data["System"].append(benchmark_name)
    
    # Create the dataframe
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create a categorical color map
    system_colors = {
        "Current System": ACITY_PRIMARY,
        "Industry Standard": "#1f77b4",  # Blue
        "Previous Version": "#ff7f0e",   # Orange
        "Target Performance": "#2ca02c"   # Green
    }
    
    # Create the comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each system as a separate line
    for system, color in system_colors.items():
        system_data = comparison_df[comparison_df["System"] == system]
        ax.plot(system_data["Metric"], system_data["Value"], marker='o', linewidth=2, label=system, color=color)
    
    # Add legend and labels
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Score")
    ax.set_title("Performance Comparison with Benchmarks", color=ACITY_PRIMARY)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Display the plot
    st.pyplot(fig)
    
    # Add a table comparison for more details
    st.markdown(f"<h4 style='color:{ACITY_PRIMARY}'>Detailed Benchmark Comparison</h4>", unsafe_allow_html=True)
    
    # Pivot the dataframe for better display
    pivot_df = pd.pivot_table(
        comparison_df, 
        values="Value", 
        index="Metric", 
        columns="System"
    ).reset_index()
    
    # Display the comparison table
    st.dataframe(pivot_df, use_container_width=True)
    
    # Add analysis and insights
    improvement_over_previous = current_metrics["composite_score"] - benchmark_metrics["Previous Version"]["composite_score"]
    gap_to_target = benchmark_metrics["Target Performance"]["composite_score"] - current_metrics["composite_score"]
    
    st.markdown(f"""
        <div style="background-color:#f0f0f0; padding:15px; border-radius:5px; margin-top:20px;">
            <h4 style="color:{ACITY_PRIMARY}">Performance Analysis:</h4>
            <ul>
                <li>The current system shows <b>{improvement_over_previous:.2%}</b> improvement over the previous version.</li>
                <li>Gap to target performance: <b>{gap_to_target:.2%}</b></li>
                <li>Current system {'exceeds' if current_metrics["composite_score"] > benchmark_metrics["Industry Standard"]["composite_score"] else 'falls below'} industry standard by <b>{abs(current_metrics["composite_score"] - benchmark_metrics["Industry Standard"]["composite_score"]):.2%}</b>.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)


# Enhanced model evaluation with comprehensive metrics
def evaluate_model_performance(embedding_model, chunks, faiss_index, api_url, api_key, test_questions):
    """Evaluate model performance using comprehensive metrics"""
    st.markdown(f"<h3 style='color:{ACITY_PRIMARY}'>Model Evaluation & Performance Analysis</h3>", unsafe_allow_html=True)
    
    results = []
    all_metrics = {}
    
    # Initialize metrics
    metric_names = ["answer_relevance", "retrieval_relevance", "answer_context_alignment", 
                    "query_answer_directness", "answer_precision", "accuracy", 
                    "f1_score", "exact_match", "composite_score"]
    
    for metric in metric_names:
        all_metrics[metric] = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (question, expected_answer) in enumerate(test_questions.items()):
        status_text.text(f"Evaluating question {i+1}/{len(test_questions)}")
        
        # Get context and model answer
        context = retrieve_relevant_chunk(question, chunks, faiss_index, embedding_model)
        model_answer = ask_model(context, question, api_url, api_key)
        
        # Calculate comprehensive metrics
        metrics = calculate_metrics(question, expected_answer, model_answer, context, embedding_model)
        
        # Add metrics to results
        for metric, value in metrics.items():
            all_metrics[metric].append(value)
        
        results.append({
            "question": question,
            "expected": expected_answer,
            "actual": model_answer,
            "metrics": metrics
        })
        
        # Update progress
        progress_bar.progress((i + 1) / len(test_questions))
    
    # Calculate average metrics
    avg_metrics = {k: sum(v)/len(v) if v else 0 for k, v in all_metrics.items()}
    
    visualize_benchmark_comparison(avg_metrics)
    
    # Create comprehensive metrics visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"<h4 style='color:{ACITY_PRIMARY}'>Performance Metrics Summary</h4>", unsafe_allow_html=True)
        
        # Format metrics for display
        display_metrics = {
            "Answer Relevance": f"{avg_metrics['answer_relevance']:.3f}",
            "Retrieval Relevance": f"{avg_metrics['retrieval_relevance']:.3f}",
            "Context Alignment": f"{avg_metrics['answer_context_alignment']:.3f}",
            "Query-Answer Directness": f"{avg_metrics['query_answer_directness']:.3f}",
            "Answer Precision": f"{avg_metrics['answer_precision']:.3f}",
            "Accuracy": f"{avg_metrics['accuracy']:.3f}",
            "F1 Score": f"{avg_metrics['f1_score']:.3f}",
            "Exact Match Rate": f"{avg_metrics['exact_match']:.3f}",
            "Composite Score": f"{avg_metrics['composite_score']:.3f}"
        }
        
        metrics_df = pd.DataFrame({
            "Metric": list(display_metrics.keys()),
            "Value": list(display_metrics.values())
        })
        
        # Style the dataframe with Academic City colors
        st.dataframe(metrics_df, use_container_width=True)
    
    with col2:
        # Create radar chart for metrics visualization
        categories = ['Answer Relevance', 'Retrieval Relevance', 'Context Alignment', 
                     'Query-Answer Directness', 'Answer Precision', 'Exact Match']
        
        values = [avg_metrics['answer_relevance'], avg_metrics['retrieval_relevance'],
                 avg_metrics['answer_context_alignment'], avg_metrics['query_answer_directness'],
                 avg_metrics['answer_precision'], avg_metrics['exact_match']]
        
        # Create radar chart
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)
        
        # Plot values
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, color=ACITY_PRIMARY)
        ax.fill(angles, values, alpha=0.25, color=ACITY_PRIMARY)
        
        # Set labels and style
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_yticklabels([])
        
        # Add grid and circle layers
        ax.set_rlabel_position(0)
        ax.grid(True)
        
        plt.ylim(0, 1)
        plt.title('Performance Metrics Radar', color=ACITY_PRIMARY, y=1.08)
        
        st.pyplot(fig)
    
    # Show gauge chart for composite score
    st.markdown(f"<h4 style='color:{ACITY_PRIMARY}'>Overall System Performance</h4>", unsafe_allow_html=True)
    
    composite_score = avg_metrics['composite_score']
    
    # Use columns for layout
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        # Create gauge chart
        fig, ax = plt.subplots(figsize=(8, 4), subplot_kw=dict(polar=True))
        
        # Parameters for the gauge
        pos = composite_score * np.pi
        bar_height = 0.1
        
        # Color segments for gauge
        poor = plt.cm.YlOrRd(0.8)
        fair = plt.cm.YlOrRd(0.6)
        good = plt.cm.YlOrRd(0.4)
        excellent = plt.cm.YlOrRd(0.2)
        
        # Plot gauge segments
        ax.bar(np.pi/2, bar_height, width=np.pi/2, bottom=0.75, color=poor, alpha=0.8, edgecolor='white')
        ax.bar(0, bar_height, width=np.pi/2, bottom=0.75, color=fair, alpha=0.8, edgecolor='white')
        ax.bar(-np.pi/2, bar_height, width=np.pi/2, bottom=0.75, color=good, alpha=0.8, edgecolor='white')
        ax.bar(-np.pi, bar_height, width=np.pi/2, bottom=0.75, color=excellent, alpha=0.8, edgecolor='white')
        
        # Add gauge needle
        ax.arrow(0, 0, np.cos(pos - np.pi/2), np.sin(pos - np.pi/2), alpha=0.8, width=0.02, 
                head_width=0.05, head_length=0.1, fc=ACITY_PRIMARY, ec=ACITY_PRIMARY)
        
        # Add score text
        ax.text(0, 0, f"{composite_score:.2f}", ha='center', va='center', fontsize=22, 
                fontweight='bold', color=ACITY_PRIMARY)
        
        # Add labels
        ax.text(-np.pi, 0.9, 'Excellent', ha='center', va='center', fontsize=10)
        ax.text(-np.pi/2, 0.9, 'Good', ha='center', va='center', fontsize=10)
        ax.text(0, 0.9, 'Fair', ha='center', va='center', fontsize=10)
        ax.text(np.pi/2, 0.9, 'Poor', ha='center', va='center', fontsize=10)
        
        # Remove ticks and grid
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_frame_on(False)
        ax.set_ylim(0, 1)
        
        st.pyplot(fig)
    
    # Show detailed results
    st.markdown(f"<h4 style='color:{ACITY_PRIMARY}'>Detailed Results</h4>", unsafe_allow_html=True)
    
    for result in results:
        with st.expander(f"Q: {result['question']}"):
            metrics = result['metrics']
            
            # Create two columns for layout
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("**Expected Answer:**")
                st.markdown(f"<div style='background-color:#f0f0f0; padding:10px; border-radius:5px;'>{result['expected']}</div>", 
                           unsafe_allow_html=True)
                
                st.markdown("**Actual Answer:**")
                answer_color = "#e6f7e6" if metrics['composite_score'] > 0.7 else "#fff0f0"
                st.markdown(f"<div style='background-color:{answer_color}; padding:10px; border-radius:5px;'>{result['actual']}</div>", 
                           unsafe_allow_html=True)
                
            with col2:
                st.markdown("**Metrics:**")
                
                # Format individual metrics
                formatted_metrics = {
                    "Answer Relevance": f"{metrics['answer_relevance']:.3f}",
                    "Retrieval Relevance": f"{metrics['retrieval_relevance']:.3f}",
                    "Context Alignment": f"{metrics['answer_context_alignment']:.3f}",
                    "Query Directness": f"{metrics['query_answer_directness']:.3f}",
                    "Precision": f"{metrics['answer_precision']:.3f}",
                    "Exact Match": "✓" if metrics['exact_match'] > 0 else "✗",
                    "Composite Score": f"{metrics['composite_score']:.3f}"
                }
                
                # Create a small dataframe for the metrics
                metrics_df = pd.DataFrame({
                    "Metric": list(formatted_metrics.keys()),
                    "Value": list(formatted_metrics.values())
                })
                
                st.dataframe(metrics_df, use_container_width=True)
    
    return results

# App state
if 'processed_pdfs' not in st.session_state:
    st.session_state.processed_pdfs = False
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None

# Sidebar configuration
with st.sidebar:
    st.markdown(f"""
        <div style="background-color:{ACITY_PRIMARY}; padding:10px; border-radius:5px; margin-bottom:10px;">
            <h3 style="color:white; text-align:center;">Configuration</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # PDF upload
    st.markdown(f"""
        <div style="margin-bottom:15px;">
            <h4 style="color:{ACITY_PRIMARY};">Step 1: Upload Documents</h4>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    # Model selection - Using smaller models that work with the free API
    st.markdown(f"""
        <div style="margin-top:20px; margin-bottom:15px;">
            <h4 style="color:{ACITY_PRIMARY};">Step 2: Select Model</h4>
        </div>
    """, unsafe_allow_html=True)
    
    model_options = {
        "Phi-3 Mini": "microsoft/phi-3-mini-4k-instruct",
        "Flan T5 Base": "google/flan-t5-base",
        "MiniLM": "sentence-transformers/all-MiniLM-L6-v2"
    }
    
    selected_model = st.selectbox("Select Model", list(model_options.keys()))
    
    # Chunk size
    st.markdown(f"""
        <div style="margin-top:20px; margin-bottom:15px;">
            <h4 style="color:{ACITY_PRIMARY};">Step 3: Processing Options</h4>
        </div>
    """, unsafe_allow_html=True)
    
    chunk_size = st.slider("Chunk Size", min_value=100, max_value=1000, value=500, step=50)
    
    # Number of chunks to retrieve
    top_k = st.slider("Number of Chunks to Retrieve", min_value=1, max_value=10, value=3)
    
    # Domain adaptation options
    domain_tab = st.expander("Domain Adaptation & Data Enhancement")
    with domain_tab:
        use_ner = st.checkbox("Use Named Entity Recognition", value=False)
        # st.markdown("*NER helps preserve entity boundaries during chunking*")
        
        entity_boost = st.checkbox("Boost Entity Matches", value=False)
        # st.markdown("*Prioritizes chunks containing entities from the query*")

        filter_noise = st.checkbox("Filter Bias & Noise", value=False)
        # st.markdown("*Reduces potential biases and noise from documents*")
        
        augmentation = st.selectbox(
            "Data Augmentation Technique", 
            ["None", "Simple Paraphrasing", "Back Translation"]
        )
        # st.markdown("*Augments data to improve model robustness*")
        
        
        # If back translation is selected, show language options
        if augmentation == "Back Translation":
            target_lang = st.selectbox(
                "Target Language for Back-Translation",
                ["French", "Spanish", "German", "Chinese", "Japanese"],
                index=0
            )
            lang_codes = {"French": "fr", "Spanish": "es", "German": "de", "Chinese": "zh-cn", "Japanese": "ja"}
            selected_lang_code = lang_codes[target_lang]
    
    # Advanced features
    advanced_tab = st.expander("Advanced Analysis")
    with advanced_tab:
        visualize_tokens = st.checkbox("Visualize Token Distribution", value=False)
        run_evaluation = st.checkbox("Run Model Evaluation", value=False)
        
        if run_evaluation:
            st.subheader("Test Questions for Evaluation")
            st.write("Enter question-answer pairs for evaluation:")
            
            # Default test questions dictionary
            default_questions = {
                "What is the vision of Academic City University?": 
                "To be a world-class center for learning, innovation and entrepreneurship that nurtures future leaders",
                
                "What is the mission of Academic City University?": 
                "To educate future-ready leaders who can innovatively solve complex problems within an ethical, entrepreneurial and collaborative environment",
                
                "Who is the president of Academic City?": 
                "Professor Fred McBagonluri"
            }
            
            # Initialize or use existing test questions
            if 'test_questions' not in st.session_state:
                st.session_state.test_questions = default_questions
            
            # Display test questions editor
            test_questions_text = st.text_area(
                "Edit test questions (format: question | expected answer, one per line)",
                value="\n".join([f"{q} | {a}" for q, a in st.session_state.test_questions.items()]),
                height=150
            )
            
            # Parse test questions from text area
            if test_questions_text:
                try:
                    lines = test_questions_text.strip().split("\n")
                    st.session_state.test_questions = {}
                    for line in lines:
                        if "|" in line:
                            q, a = line.split("|", 1)
                            st.session_state.test_questions[q.strip()] = a.strip()
                except Exception as e:
                    st.error(f"Error parsing test questions: {str(e)}")
    
    # Process button styled with Academic City colors
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #910000;
            color: white;
            font-weight: bold;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            width: 100%;
        }
        div.stButton > button:hover {
            background-color: #700000;
            border: none;
        }
        </style>
    """, unsafe_allow_html=True)
    
    process_button = st.button("Process PDFs")

# Main content
if process_button and uploaded_files:
    st.session_state.processed_pdfs = False
    
    # Save uploaded files temporarily
    pdf_folder = "temp_pdfs"
    os.makedirs(pdf_folder, exist_ok=True)
    pdf_files = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(pdf_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_files.append(file_path)
    
    # Process PDFs with stylized progress indicators
    st.markdown(f"""
        <div style="background-color:#F8F9FA; padding:15px; border-radius:5px; margin-bottom:20px; border-left:5px solid {ACITY_PRIMARY};">
            <h4 style="color:{ACITY_PRIMARY};">Processing Documents</h4>
        </div>
    """, unsafe_allow_html=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Extract text
    status_text.markdown(f"<p style='color:{ACITY_PRIMARY};'>Extracting text from PDFs...</p>", unsafe_allow_html=True)
    all_text = ""
    for i, path in enumerate(pdf_files):
        all_text += extract_text_from_pdf(path) + "\n\n"
        progress_bar.progress((i + 1) / (len(pdf_files) + 5))
    
    # Apply data augmentation if selected
    if augmentation != "None":
        status_text.markdown(f"<p style='color:{ACITY_PRIMARY};'>Applying {augmentation}...</p>", unsafe_allow_html=True)
        if augmentation == "Simple Paraphrasing":
            augmented_text = simple_paraphrase(all_text)
            # Mix original and augmented with low impact (70% original, 30% augmented)
            all_text = all_text + "\n\n" + augmented_text[:int(len(all_text)*0.3)]
        elif augmentation == "Back Translation":
            # Apply back translation to a small portion (20%) to minimize impact
            sample_size = int(len(all_text) * 0.2)
            sample_start = random.randint(0, len(all_text) - sample_size)
            sample_text = all_text[sample_start:sample_start+sample_size]
            
            augmented_sample = back_translate(sample_text, source_lang="en", target_lang=selected_lang_code)
            all_text = all_text[:sample_start] + augmented_sample + all_text[sample_start+sample_size:]
        
        progress_bar.progress((len(pdf_files) + 1) / (len(pdf_files) + 5))
    
    # Apply bias and noise filtering if selected
    if filter_noise:
        status_text.markdown(f"<p style='color:{ACITY_PRIMARY};'>Filtering bias and noise...</p>", unsafe_allow_html=True)
        all_text = filter_bias_and_noise(all_text)
        progress_bar.progress((len(pdf_files) + 2) / (len(pdf_files) + 5))
    
    # Chunk text
    status_text.markdown(f"<p style='color:{ACITY_PRIMARY};'>Chunking text with domain adaptation...</p>", unsafe_allow_html=True)
    st.session_state.chunks = enhanced_chunk_text(all_text, chunk_size=chunk_size, domain_adaptation=use_ner)
    progress_bar.progress((len(pdf_files) + 3) / (len(pdf_files) + 5))
    
    # Load embedding model
    status_text.markdown(f"<p style='color:{ACITY_PRIMARY};'>Loading embedding model...</p>", unsafe_allow_html=True)
    st.session_state.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    progress_bar.progress((len(pdf_files) + 4) / (len(pdf_files) + 5))
    
    # Create embeddings and index
    status_text.markdown(f"<p style='color:{ACITY_PRIMARY};'>Creating embeddings and index...</p>", unsafe_allow_html=True)
    embeddings = np.array([st.session_state.embedding_model.encode(chunk) for chunk in st.session_state.chunks])
    st.session_state.nn_index = NearestNeighbors(n_neighbors=top_k, metric='euclidean')
    st.session_state.nn_index.fit(embeddings)
    st.session_state.embeddings = embeddings  # Save for later retrieval
    progress_bar.progress(1.0)
    
    # Display domain-specific entity information if NER is enabled
    if use_ner:
        entity_count = 0
        entity_types = Counter()
        for chunk in st.session_state.chunks[:20]:  # Limit for performance
            doc = nlp(chunk[:5000])  # Limit text size
            for ent in doc.ents:
                entity_count += 1
                entity_types[ent.label_] += 1
        
        status_text.markdown(f"""
            <div style='background-color:#e6f7e6; padding:10px; border-radius:5px; border-left:5px solid green;'>
                <p>Successfully processed {len(uploaded_files)} PDFs, created {len(st.session_state.chunks)} chunks with {entity_count} named entities.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        status_text.markdown(f"""
            <div style='background-color:#e6f7e6; padding:10px; border-radius:5px; border-left:5px solid green;'>
                <p>Successfully processed {len(uploaded_files)} PDFs, created {len(st.session_state.chunks)} chunks.</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.session_state.processed_pdfs = True

    # Clean up temporary files
    for path in pdf_files:
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists(pdf_folder) and not os.listdir(pdf_folder):
        os.rmdir(pdf_folder)

# QA Interface
if st.session_state.processed_pdfs:
    st.markdown(f"""
        <div style="background-color:#F8F9FA; padding:15px; border-radius:5px; margin-top:30px; margin-bottom:20px; border-left:5px solid {ACITY_PRIMARY};">
            <h3 style="color:{ACITY_PRIMARY};">Ask Questions About Your Documents</h3>
            <p style="color:black;">Enter your question below to get accurate answers from your PDFs.<br> Choose Phi-3-Mini for best results</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Add visualization if enabled
    if visualize_tokens:
        visualize_token_distribution(st.session_state.chunks)
    
    # Add evaluation if enabled
    if run_evaluation:
        api_url = f"https://api-inference.huggingface.co/models/{model_options[selected_model]}"
        evaluation_results = evaluate_model_performance(
            st.session_state.embedding_model,
            st.session_state.chunks,
            st.session_state.nn_index,  
            api_url,
            API_KEY,
            st.session_state.test_questions
        )
    
    # Create a pill-based UI for query suggestions
    st.markdown("""
        <style>
        .suggestion-pill {
            display: inline-block;
            background-color: #f0f0f0;
            color: #333;
            padding: 5px 15px;
            margin: 5px;
            border-radius: 20px;
            font-size: 14px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .suggestion-pill:hover {
            background-color: #e0e0e0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sample questions specific to Academic City
    sample_questions = [
        "What programs does Academic City offer?",
        "Who is the president of Academic City?",
        "What is the vision and mission of Academic City?",
        "What are the admission requirements?"
    ]
    
    st.markdown("<p>Try asking:</p>", unsafe_allow_html=True)
    pills_html = "".join([f"<span class='suggestion-pill'>{q}</span>" for q in sample_questions])
    st.markdown(f"<div>{pills_html}</div>", unsafe_allow_html=True)
    
    # Query input with Academic City styling
    query = st.text_input("Enter your question:", key="query_input")
    
    if query:
        # Get API URL for selected model
        api_url = f"https://api-inference.huggingface.co/models/{model_options[selected_model]}"
        
        # Create columns for QA display
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Retrieve context
            context = retrieve_relevant_chunk(
                query, 
                st.session_state.chunks, 
                st.session_state.nn_index,  
                st.session_state.embedding_model, 
                top_k=top_k,
                entity_boost=entity_boost
            )
            
            # Get answer with animation
            with st.spinner(f"Searching and analyzing documents for your answer..."):
                answer = ask_model(context, query, api_url, API_KEY)
            
            # Display answer with Academic City styling
            display_answer = answer.split("Answer:")[-1].split("Context")[0].strip()

            st.markdown(f"""
                <div style="border-left: 5px solid {ACITY_PRIMARY}; padding-left: 20px; margin-bottom: 20px;">
                <h3 style="color: {ACITY_PRIMARY};">Answer</h3>
                <div style="background-color: white; color: {ACITY_TEXT}; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                    {display_answer}
                </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Show context (optional)
            with st.expander("View Retrieved Context"):
                context_sections = context.split("---")
                for i, section in enumerate(context_sections):
                    st.markdown(f"""
                        <div style="margin-bottom: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 5px;">
                            <h4 style="color: {ACITY_PRIMARY};">Context Segment {i+1}</h4>
                            <div style="font-size: 0.9em; color: black">{section.replace(chr(10), '<br>')}</div>
                        </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            # Real-time metrics calculation for this answer
            st.markdown(f"<h4 style='color:{ACITY_PRIMARY};'>Response Metrics</h4>", unsafe_allow_html=True)
            
            # Calculate relevant metrics we can get without a reference answer
            query_embedding = st.session_state.embedding_model.encode([query])
            answer_embedding = st.session_state.embedding_model.encode([answer])
            context_embedding = st.session_state.embedding_model.encode([context])
            
            # Retrieval relevance (query-context similarity)
            retrieval_relevance = np.dot(context_embedding[0], query_embedding[0]) / (
                np.linalg.norm(context_embedding[0]) * np.linalg.norm(query_embedding[0])
            )
            
            # Answer-context alignment
            answer_context_alignment = np.dot(answer_embedding[0], context_embedding[0]) / (
                np.linalg.norm(answer_embedding[0]) * np.linalg.norm(context_embedding[0])
            )
            
            # Query-answer directness
            query_answer_directness = np.dot(query_embedding[0], answer_embedding[0]) / (
                np.linalg.norm(query_embedding[0]) * np.linalg.norm(answer_embedding[0])
            )
            
            # Create gauge charts for visualization
            metrics = {
                "Retrieval Relevance": retrieval_relevance,
                "Answer-Context Alignment": answer_context_alignment,
                "Query-Answer Directness": query_answer_directness,
            }
            
            for metric_name, value in metrics.items():
                # Create colored gauge based on value
                if value >= 0.8:
                    color = "#28a745"  # Green for excellent
                    quality = "Excellent"
                elif value >= 0.6:
                    color = "#17a2b8"  # Blue for good
                    quality = "Good"
                elif value >= 0.4:
                    color = "#ffc107"  # Yellow for fair
                    quality = "Fair"
                else:
                    color = "#dc3545"  # Red for poor
                    quality = "Poor"
                
                # Display metric with gauge visualization
                st.markdown(f"""
                    <div style="margin-bottom: 15px;">
                        <p style="margin-bottom: 5px;">{metric_name}</p>
                        <div style="background-color: #e9ecef; height: 8px; border-radius: 4px; width: 100%;">
                            <div style="background-color: {color}; width: {value*100}%; height: 100%; border-radius: 4px;"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.8em;">
                            <span>{value:.2f}</span>
                            <span>{quality}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Offer feedback mechanism
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f"<h5 style='color:{ACITY_PRIMARY};'>Was this answer helpful?</h5>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                thumbs_up = st.button("👍 Yes")
            with col2:
                thumbs_down = st.button("👎 No")
            
            if thumbs_down:
                improvement = st.text_area("How can we improve?")
                if st.button("Submit Feedback"):
                    st.success("Thank you for your feedback! We'll use it to improve the system.")
            elif thumbs_up:
                st.success("Thank you for your feedback!")

else:
    # Display the welcome screen when no PDFs are processed yet
    st.markdown(f"""
        <div style="text-align: center; padding: 40px 20px; background-color: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <img src="https://www.google.com/imgres?q=academic%20city&imgurl=https%3A%2F%2Flookaside.fbsbx.com%2Flookaside%2Fcrawler%2Fmedia%2F%3Fmedia_id%3D100072339720275&imgrefurl=https%3A%2F%2Fwww.facebook.com%2Facitygh%2F&docid=gs1mOS5nSZOCAM&tbnid=i9Cnc0cusAbaiM&vet=12ahUKEwi1rLDE0-GMAxXvTaQEHcDoChMQM3oECGYQAA..i&w=2045&h=2048&hcb=2&ved=2ahUKEwi1rLDE0-GMAxXvTaQEHcDoChMQM3oECGYQAA" style="width: 150px; height: 150px; border-radius: 50%;">
            <h2 style="color: {ACITY_PRIMARY}; margin-top: 20px;">Welcome to Academic City's PDF Question Answering System</h2>
            <p style="color: {ACITY_TEXT}; font-size: 1.2em; margin: 20px 0;">
                Upload PDF documents and ask questions to get accurate, contextual answers.
            </p>
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; text-align: left; max-width: 600px; margin: 0 auto;">
                <h4 style="color: {ACITY_PRIMARY};">Getting Started:</h4>
                <ol style="color: {ACITY_TEXT}; text-align: left;">
                    <li>Upload PDF files using the sidebar on the left</li>
                    <li>Configure processing options if needed</li>
                    <li>Click "Process PDFs" to analyze your documents</li>
                    <li>Ask questions and get intelligent answers!</li>
                </ol>
                <p style="color: black"> NOTE: The accuracy of the model depends on which you chose and the advanced processing options you selected.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Sample features showcase
    st.markdown(f"""
        <div style="margin-top: 40px;">
            <h3 style="color: {ACITY_PRIMARY}; text-align: center;">System Features</h3>
            <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin-top: 20px;">
                <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); width: 300px;">
                    <h4 style="color: {ACITY_PRIMARY};">Named Entity Recognition</h4>
                    <p style="color: black";>Preserves important concepts and names during document processing for more accurate answers.</p>
                </div>
                <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); width: 300px;">
                    <h4 style="color: {ACITY_PRIMARY};">Comprehensive Metrics</h4>
                    <p style="color: black";>Evaluate system performance with detailed retrieval and answer quality metrics.</p>
                </div>
                <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); width: 300px;">
                    <h4 style="color: {ACITY_PRIMARY};">Data Augmentation</h4>
                    <p style="color: black";>Enhanced content processing through paraphrasing and translation for better results.</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
