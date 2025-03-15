# import json
# import os
# import numpy as np
# import faiss
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import openai
# from fastapi.middleware.cors import CORSMiddleware  # Add this import
# from langchain_openai import OpenAIEmbeddings

# # 🔹 Load OpenAI API Key
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")

# if not openai_api_key:
#     raise ValueError("❌ OPENAI_API_KEY is missing! Set it as an environment variable.")

# # ✅ Initialize OpenAI Client
# client = openai.OpenAI(api_key=openai_api_key)

# # ✅ FastAPI App
# app = FastAPI(title="Chatbot API", version="1.0")

# # Add CORS middleware here
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

# # ✅ Load JSON Data
# def load_json(file_path="final_persons.json"):
#     with open(file_path, "r", encoding="utf-8") as file:
#         data = json.load(file)
#     return data["persons"]

# # ✅ Initialize or Load FAISS Index
# def create_or_load_vector_store():
#     json_data = load_json()
    
#     if os.path.exists("faiss_index.bin"):
#         index = faiss.read_index("faiss_index.bin")
#         with open("metadata.json", "r", encoding="utf-8") as file:
#             metadata = json.load(file)
#         texts = metadata["texts"]
#     else:
#         embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
#         texts = [entry["full_name"][:200] for entry in json_data]
#         vectors = np.array([embeddings_model.embed_query(text) for text in texts], dtype=np.float32)

#         index = faiss.IndexFlatL2(vectors.shape[1])
#         index.add(vectors)

#         faiss.write_index(index, "faiss_index.bin")
#         with open("metadata.json", "w", encoding="utf-8") as file:
#             json.dump({"texts": texts, "json_data": json_data}, file, indent=4)

#     return index, texts, json_data

# index, texts, json_data = create_or_load_vector_store()

# # ✅ Find Best Match Function
# def find_best_matches(query, top_k=3):
#     embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
#     query_vector = np.array([embeddings_model.embed_query(query[:200])], dtype=np.float32)

#     _, I = index.search(query_vector, k=top_k)
    
#     results = []
#     for match_index in I[0]:
#         if match_index != -1:
#             results.append(json_data[match_index])

#     return results if results else [{"response": "No relevant data found."}]

# # ✅ OpenAI GPT-4 Response
# def get_chat_response(user_query):
#     best_matches = find_best_matches(user_query, top_k=3)
    
#     best_match_texts = "\n".join([json.dumps(match, indent=4) for match in best_matches])

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": "You are a genealogy assistant providing accurate historical data."},
#                 {"role": "user", "content": f"User asked: {user_query}. Best matches found:\n{best_match_texts}"}
#             ]
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"❌ OpenAI API Error: {str(e)}"

# # ✅ API Endpoint for Chatbot
# class QueryRequest(BaseModel):
#     query: str

# @app.post("/chat")
# async def chat_with_bot(request: QueryRequest):
#     response_text = get_chat_response(request.query)
#     return {"response": response_text}

# @app.get("/")
# async def root():
#     return {"message": "Chatbot API is Running!"}





import json
import os
import numpy as np
import faiss
import redis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import OpenAIEmbeddings

# 🔹 Load Environment Variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

# ✅ Check API Key
if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY is missing! Set it as an environment variable.")

# ✅ OpenAI Client
client = openai.OpenAI(api_key=openai_api_key)

# ✅ Redis Client (For Caching)
redis_client = redis.Redis.from_url(redis_url, decode_responses=True)

# ✅ FastAPI App
app = FastAPI(title="Fast Genealogy Chatbot", version="2.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load JSON Data
def load_json(file_path="final_persons.json"):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data["persons"]

# ✅ Load FAISS Index with HNSW
def create_or_load_vector_store():
    json_data = load_json()
    
    if os.path.exists("faiss_index.bin"):
        index = faiss.read_index("faiss_index.bin")
        with open("metadata.json", "r", encoding="utf-8") as file:
            metadata = json.load(file)
        texts = metadata["texts"]
    else:
        embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
        texts = [entry["full_name"][:200] for entry in json_data]
        vectors = np.array([embeddings_model.embed_query(text) for text in texts], dtype=np.float32)

        # Use HNSW index instead of L2 for speed
        d = vectors.shape[1]
        index = faiss.IndexHNSWFlat(d, 32)  # HNSW with 32 neighbors
        index.add(vectors)

        faiss.write_index(index, "faiss_index.bin")
        with open("metadata.json", "w", encoding="utf-8") as file:
            json.dump({"texts": texts, "json_data": json_data}, file, indent=4)

    return index, texts, json_data

index, texts, json_data = create_or_load_vector_store()

# ✅ Find Best Match Using FAISS
def find_best_matches(query, top_k=3):
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    query_vector = np.array([embeddings_model.embed_query(query[:200])], dtype=np.float32)

    _, I = index.search(query_vector, k=top_k)
    
    results = []
    for match_index in I[0]:
        if match_index != -1:
            results.append(json_data[match_index])

    return results if results else [{"response": "No relevant data found."}]

# ✅ OpenAI GPT-4 Response with Redis Caching
def get_chat_response(user_query):
    # Check cache first
    cached_response = redis_client.get(user_query)
    if cached_response:
        return cached_response  # Instant response if cached

    best_matches = find_best_matches(user_query, top_k=3)
    best_match_texts = "\n".join([json.dumps(match, indent=4) for match in best_matches])

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use GPT-3.5 for speed; fallback to GPT-4 if needed
            messages=[
                {"role": "system", "content": "You are a genealogy assistant providing accurate historical data."},
                {"role": "user", "content": f"User asked: {user_query}. Best matches found:\n{best_match_texts}"}
            ]
        )
        final_response = response.choices[0].message.content

        # Cache the response in Redis for 24 hours
        redis_client.setex(user_query, 86400, final_response)
        return final_response
    except Exception as e:
        return f"❌ OpenAI API Error: {str(e)}"

# ✅ API Endpoint for Chatbot
class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_with_bot(request: QueryRequest):
    response_text = get_chat_response(request.query)
    return {"response": response_text}

@app.get("/")
async def root():
    return {"message": "Chatbot API is Running with HNSW & Redis!"}
