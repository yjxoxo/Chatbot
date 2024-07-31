import time
import asyncio
import pymysql
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import faiss
import spacy
import torch
import os
from fastapi.staticfiles import StaticFiles
import uvicorn
from uvicorn import Config, Server
import logging
from rapidfuzz import fuzz, process
import pickle

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Ensure the spaCy model for Korean is installed
try:
    nlp = spacy.load("ko_core_news_sm")
except OSError:
    spacy.cli.download("ko_core_news_sm")
    nlp = spacy.load("ko_core_news_sm")

# Database connection details
conn = pymysql.connect(
    host="133.186.215.75",
    port=10000,
    user="kdxuser",
    password="71tnf7oqkfQn!U$er~",
    database="db_kdx"
)

# SQL query to fetch data
query = """
SELECT T.PRODUCT_ID AS id, T.TITLE, T.DESCRIPTION
FROM (
    SELECT p.PRODUCT_ID, p.TITLE, p.DESCRIPTION
    FROM db_kdx.PRODUCT p
    LEFT JOIN db_kdx.PRODUCT_KEYWORD k ON p.PRODUCT_ID = k.PRODUCT_ID
    JOIN db_kdx.USER u ON p.CREATOR_USER_ID = u.USER_ID
    LEFT JOIN db_kdx.CATEGORY c ON p.CATEGORY_ID = c.CATEGORY_ID
    LEFT JOIN db_kdx.CORP_INFO ci ON ci.CORP_ID = u.CORP_ID
    WHERE 1=1
      AND p.IS_ENABLE = 1
      AND p.STATUS != 'deleted'
      AND NOW() <= p.SALE_END_DATE
      AND p.LINK_TYPE = 'inlink'
      AND p.IS_ENABLE = '1'
    GROUP BY p.PRODUCT_ID, p.TITLE, p.DESCRIPTION
) T;
"""

# Fetch data from the database
df = pd.read_sql(query, conn)
conn.close()

# Refined system prompt
system_prompt = (
    "제공된 내용을 분석합니다."
    "내용의 언급된 주요 개념, 주제, 실체를 파악하여 논의된 핵심 아이디어와 주제를 가장 잘 나타내는 5개의 키워드를 추출합니다."
    "모두 한국어로 진행됩니다."
)

# Data preprocessing
titles = df['TITLE'].tolist()
descriptions = df['DESCRIPTION'].tolist()
texts = [f"{title} {desc}" for title, desc in zip(titles, descriptions)]

# Load BGE-M3 model
embedding_model_name = "BAAI/bge-m3"
embedding_model = SentenceTransformer(embedding_model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_model.to(device)

# Load Qwen2-7B-Instruct model
qwen_model_name = "spow12/Ko-Qwen2-7B-Instruct"
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
qwen_model = AutoModelForCausalLM.from_pretrained(qwen_model_name)
qwen_model.eval()
qwen_model.to(device)

# File paths for saving and loading embeddings and FAISS index
embeddings_file = "./data_embeddings_0729_server.npy"
faiss_index_file = "./faiss_index_0729_server.bin"
embeddings_cache_file = "./embeddings_cache.pkl"

# Define the file to save questions
QUESTIONS_FILE = "questions_log.txt"

# Load or initialize embeddings cache
if os.path.exists(embeddings_cache_file):
    with open(embeddings_cache_file, "rb") as f:
        embeddings_cache = pickle.load(f)
else:
    embeddings_cache = {}

# Function to save the embeddings cache
def save_embeddings_cache():
    with open(embeddings_cache_file, "wb") as f:
        pickle.dump(embeddings_cache, f)

# Function to save the question to a txt file
def save_question(question):
    with open(QUESTIONS_FILE, "a") as file:
        file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {question}\n")

# Function to load predefined responses from CSV
def load_predefined_responses(csv_file):
    df = pd.read_csv(csv_file)
    predefined_responses = dict(zip(df['question'], df['response']))
    return predefined_responses

# Load predefined responses from CSV
predefined_responses = load_predefined_responses("predefined_responses.csv")

# Function to find the best match for a user question using RapidFuzz
def find_predefined_response(question, predefined_responses):
    questions = list(predefined_responses.keys())
    best_match, score, _ = process.extractOne(question, questions, scorer=fuzz.ratio)
    if score > 80:  # Adjust threshold as needed
        return predefined_responses[best_match]
    return None

# Function to compute embeddings for a list of texts
async def get_embeddings(texts, model, batch_size=10):  # Reduced batch size for potentially faster response
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
        embeddings.extend(batch_embeddings.cpu().numpy())
    return np.array(embeddings)

# Save embeddings and FAISS index
def save_embeddings_and_index(embeddings, index):
    np.save(embeddings_file, embeddings)
    faiss.write_index(index, faiss_index_file)
    save_embeddings_cache()  # Save embeddings cache

# Load embeddings and FAISS index
def load_embeddings_and_index():
    embeddings = np.load(embeddings_file)
    index = faiss.read_index(faiss_index_file)
    return embeddings, index

# Check if embeddings and FAISS index already exist
if os.path.exists(embeddings_file) and os.path.exists(faiss_index_file):
    data_embeddings, faiss_index = load_embeddings_and_index()
else:
    # Compute embeddings for data descriptions
    data_embeddings = asyncio.run(get_embeddings(texts, embedding_model))
    
    # Normalize embeddings for cosine similarity
    data_embeddings = data_embeddings / np.linalg.norm(data_embeddings, axis=1, keepdims=True)
    
    # Create FAISS index for data embeddings
    dimension = data_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(data_embeddings)
    
    # Save embeddings and FAISS index
    save_embeddings_and_index(data_embeddings, faiss_index)

# Store previous recommendations to handle duplicate questions
previous_recommendations = {}

# Function to analyze intent using Qwen2-7B-Instruct model
async def analyze_intent(question, tokenizer, model, temperature=0.1, max_new_tokens=150):
    prompt_with_question = f"{system_prompt}\nQuestion: {question}\n"
    inputs = tokenizer(prompt_with_question, return_tensors='pt')
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature)
    intent = tokenizer.decode(outputs[0], skip_special_tokens=True)
    intent = intent.split(prompt_with_question)[-1].strip()
    return intent

# Function to extract keywords from the analyzed intent using spaCy
def extract_keywords(intent):
    doc = nlp(intent)
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] and token.is_alpha]
    keywords = list(set(keywords))  # Remove duplicates
    keywords = [keyword for keyword in keywords if keyword.strip()]
    logger.info(f"Extracted keywords: {keywords}")
    return keywords

# Function to analyze question and find contributing words
async def analyze_question_and_find_contributing_words(question, model, faiss_index, data_embeddings, titles, descriptions, previous_recommendations):
    intent = await analyze_intent(question, qwen_tokenizer, qwen_model)
    print(f"Analyzed intent: {intent}")
    keywords = extract_keywords(intent)
    print(f"Extracted keywords: {keywords}")

    intent_embedding = await get_embeddings([intent], model)
    intent_embedding = intent_embedding / np.linalg.norm(intent_embedding, axis=1, keepdims=True)
    D, I = faiss_index.search(intent_embedding, len(titles))

    previous_indices = previous_recommendations.get(question, [])
    best_match_index = None
    for idx in I[0]:
        if idx not in previous_indices:
            best_match_index = idx
            previous_indices.append(idx)
            break
    previous_recommendations[question] = previous_indices

    if best_match_index is None:
        return None, None, None, [], intent

    keyword_embeddings = await get_embeddings(keywords, model)
    keyword_embeddings = keyword_embeddings / np.linalg.norm(keyword_embeddings, axis=1, keepdims=True)
    similarities = cosine_similarity(keyword_embeddings, data_embeddings[best_match_index].reshape(1, -1)).flatten()

    logger.info(f"Recommended data score: {similarities[0]}")

    contributing_words = sorted(zip(keywords, similarities), key=lambda x: x[1], reverse=True)

    best_title = titles[best_match_index]
    best_description = descriptions[best_match_index]
    best_id = df.loc[df['TITLE'] == best_title, 'id'].values[0]

    return best_id, best_title, best_description, contributing_words, intent


# Function to get chatbot response
async def chatbot(question):
    save_question(question)  # Save the question to the log file
    
    predefined_response = find_predefined_response(question, predefined_responses)
    if predefined_response:
        logger.info(f"Predefined response found for question: {question}")
        return "predefined", "", "", [], predefined_response
    
    if question in embeddings_cache:
        question_embedding = embeddings_cache[question]
        logger.info(f"Using cached embedding for question: {question}")
    else:
        question_embedding = await get_embeddings([question], embedding_model)
        question_embedding = question_embedding / np.linalg.norm(question_embedding, axis=1, keepdims=True)
        embeddings_cache[question] = question_embedding
        logger.info(f"Caching embedding for question: {question}")

    D, I = faiss_index.search(question_embedding, 1)
    threshold_score = 0.4  # Lower the threshold score

    logger.info(f"Question embedding cosine similarity scores: {D[0]}")

    if D[0][0] > threshold_score:
        best_match_index = I[0][0]
        best_title = titles[best_match_index]
        best_description = descriptions[best_match_index]
        best_id = df.loc[df['TITLE'] == best_title, 'id'].values[0]

        logger.info(f"High-scoring match found with score {D[0][0]} for question: {question}")

        intent = f"{question}에 관련된 주요키워드"
        return str(best_id), best_title, best_description, [], intent
    
    logger.info(f"No high-scoring match found for question: {question}, falling back to model.")

    best_id, best_title, best_description, contributing_words, intent = await analyze_question_and_find_contributing_words(
        question, embedding_model, faiss_index, data_embeddings, titles, descriptions, previous_recommendations
    )
    intent = f"{question}에 관련된 주요키워드"
    return str(best_id), best_title, best_description, contributing_words, intent


class Question(BaseModel):
    question: str

class Response(BaseModel):
    id: str = None
    title: str
    description: str = None
    contributing_words: List[Dict[str, Any]] = None
    response_time: float
    intent: str

@app.post("/chatbot", response_model=Response)
async def ask_question(question: Question):
    start_time = time.time()
    try:
        # Set a timeout for the chatbot function
        best_id, best_title, best_description, contributing_words, intent = await asyncio.wait_for(
            chatbot(question.question), timeout=70.0
        )
        end_time = time.time()
        response_time = end_time - start_time
        
        logger.info(f"Response time: {response_time:.2f} seconds")
        
        response = {
            "id": best_id if best_id else "",
            "title": best_title if best_title else "",
            "description": best_description if best_description else "",
            "contributing_words": [{"word": word, "similarity": float(score)} for word, score in contributing_words] if contributing_words else [],
            "response_time": response_time,
            "intent": intent if intent else ""
        }
        logger.info(f"Response generated: {response}")
        return response
    except asyncio.TimeoutError:
        logger.error("관련된 데이터가 없습니다.")
        raise HTTPException(status_code=408, detail="Relevant data cannot be found within the time limit.")
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def get_index():
    with open("static/index1.html", "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read(), status_code=200)

if __name__ == "__main__":
    app.mount("/static", StaticFiles(directory="static"), name="static")

    config = Config(app, host="localhost", port=8887, log_level="info")
    server = Server(config)
    server.run()
