import os
import glob
import sqlite3
import pandas as pd
import httpx
import asyncio
import re
from datetime import datetime
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(collection_name="knowledge_base", embedding_function=embedding_model, persist_directory="./chroma_db")

db_seeds = Chroma(
    collection_name="seed_db",
    embedding_function=embedding_model,
    persist_directory="./chroma_seeds"
)

db_states = Chroma(collection_name="state_db", embedding_function=embedding_model, persist_directory="./chroma_states")

db_market = Chroma(
    collection_name="market_db",
    embedding_function=embedding_model,
    persist_directory="./chroma_market"
)

# CSV ingestion
def ingest_all_csvs(folder_path="data_csv", chunk_size=5):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            new_texts = []
            for start_idx in range(0, len(df), chunk_size):
                chunk_rows = df.iloc[start_idx: start_idx + chunk_size]
                chunk_texts = [" | ".join([f"{col}: {row[col]}" for col in chunk_rows.columns]) for _, row in chunk_rows.iterrows()]
                new_texts.append("\n".join(chunk_texts))
            db.add_texts(new_texts)
        except Exception as e:
            print(f"Error ingesting {csv_file}: {e}")
    db.persist()



def ingest_pdfs(folder_path="data_pdf"):
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            documents = loader.load()
            texts = [doc.page_content for doc in documents]
            db_seeds.add_texts(texts)
        except Exception as e:
            print(f"Error ingesting {pdf_file}: {e}")
    db_seeds.persist()


def ingest_seed_csvs(folder_path="data_pdf", chunk_size=5):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            new_texts = []
            for start_idx in range(0, len(df), chunk_size):
                chunk_rows = df.iloc[start_idx: start_idx + chunk_size]
                chunk_texts = [
                    " | ".join([f"{col}: {row[col]}" for col in chunk_rows.columns])
                    for _, row in chunk_rows.iterrows()
                ]
                new_texts.append("\n".join(chunk_texts))
            db_seeds.add_texts(new_texts)   # 👈 add into seed_db, not general db
        except Exception as e:
            print(f"Error ingesting seed CSV {csv_file}: {e}")
    db_seeds.persist()

# ================= INGEST TXT FILES STATE-WISE =================
def ingest_state_txts(folder_path="data_states"):
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    for txt_file in txt_files:
        try:
            state_name = os.path.basename(txt_file).replace(".txt", "")
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Store with metadata: which state this belongs to
            db_states.add_texts([content], metadatas=[{"state": state_name}])
        except Exception as e:
            print(f"Error ingesting {txt_file}: {e}")
    db_states.persist()


# Dynamically build the absolute path for the DB file
# Assuming 'agri_market.db' is in the same directory as 'main.py'
db_file_path = os.path.join(os.path.dirname(__file__), "agri_market.db")

def ingest_sqlite_db(db_path, table_name, chunk_size=200):
    """Ingests data from a SQLite database table into the market vector store."""
    if not os.path.exists(db_path):
        print(f"Database file not found at {db_path}, skipping ingestion.")
        return
    try:
        print(f"Starting ingestion of market data from {db_path}, table: {table_name}")
        conn = sqlite3.connect(db_path)
        
        # First check if table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not cursor.fetchone():
            print(f"Table '{table_name}' does not exist in {db_path}. Available tables:")
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            for table in tables:
                print(f"- {table[0]}")
            return
            
        # Proceed with ingestion
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        print(f"Read {len(df)} rows from {table_name}")
        
        # Delete any existing data first to avoid duplicates
        try:
            db_market.delete_collection()
            print("Cleared existing market data before ingestion")
        except Exception as e:
            print(f"Note: Could not clear existing market data: {e}")
        
        # Ingest data in smaller chunks to avoid memory issues
        new_texts = []
        metadatas = []
        ids = []
        
        for start_idx in range(0, len(df), chunk_size):
            chunk_rows = df.iloc[start_idx: start_idx + chunk_size]
            for idx, row in chunk_rows.iterrows():
                # Create a text representation of the row
                row_text = " | ".join([f"{col}: {row[col]}" for col in chunk_rows.columns if pd.notna(row[col])])
                new_texts.append(row_text)
                
                # Create metadata for better searching
                metadata = {"source": "market_db"}
                for col in chunk_rows.columns:
                    if pd.notna(row[col]):
                        # Only add string or numeric values to metadata
                        if isinstance(row[col], (str, int, float)):
                            metadata[col] = str(row[col])
                metadatas.append(metadata)
                
                # Create a unique ID
                ids.append(f"market_{idx}")
                
            print(f"Processed rows {start_idx} to {min(start_idx + chunk_size, len(df))}")
        
        # Add to vector database with metadata in smaller batches
        print(f"Adding {len(new_texts)} texts to vector database in batches...")
        batch_size = 5000  # Process 5000 texts at a time
        total_added = 0
        
        for batch_start in range(0, len(new_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(new_texts))
            print(f"Adding batch {batch_start} to {batch_end} to vector database...")
            
            batch_texts = new_texts[batch_start:batch_end]
            batch_metadatas = metadatas[batch_start:batch_end]
            batch_ids = ids[batch_start:batch_end]
            
            db_market.add_texts(texts=batch_texts, metadatas=batch_metadatas, ids=batch_ids)
            db_market.persist()  # Persist after each batch
            
            total_added += len(batch_texts)
            print(f"Added {total_added} texts so far ({(total_added/len(new_texts)*100):.1f}% complete)")
        
        conn.close()
        print(f"Successfully ingested {len(df)} rows from table '{table_name}' in {db_path} into db_market.")
    except Exception as e:
        print(f"Error ingesting SQLite DB {db_path}: {e}")
        import traceback
        print(traceback.format_exc())


# --- OPTIMIZATION: Ingest data only if the database is not already populated ---
def check_and_ingest(force_market_ingestion=False):
    """
    Check if vector databases exist and ingest data if they don't.
    
    Parameters:
        force_market_ingestion (bool): If True, will re-ingest market data regardless of whether
                                       the database exists already. Default is False, which
                                       means market data will only be ingested once when the 
                                       database file doesn't exist.
    """
    # A simple check for the existence of the SQLite file Chroma uses.
    # This prevents re-ingesting data on every server startup.
    
    # Check for states data
    if not os.path.exists(os.path.join("./chroma_states", "chroma.sqlite3")):
        print("Database for states not found. Ingesting state-specific data...")
        ingest_state_txts()
    else:
        print("State data found, skipping ingestion.")

    # Check for general knowledge base data
    if not os.path.exists(os.path.join("./chroma_db", "chroma.sqlite3")):
        print("Database for general knowledge not found. Ingesting general CSV data...")
        ingest_all_csvs()
    else:
        print("General knowledge base found, skipping ingestion.")

    # Check for seed data
    if not os.path.exists(os.path.join("./chroma_seeds", "chroma.sqlite3")):
        print("Database for seeds not found. Ingesting seed-related PDFs and CSVs...")
        ingest_pdfs()
        ingest_seed_csvs()
    else:
        print("Seed data found, skipping ingestion.")

    # Check for market data - can be forced to reingest even if file exists
    market_db_path = os.path.join("./chroma_market", "chroma.sqlite3")
    if not os.path.exists(market_db_path) or force_market_ingestion:
        print("Ingesting market data from agri_market.db...")
        
        # If the directory exists but we're forcing reingestion, try to remove the existing database
        if os.path.exists(market_db_path) and force_market_ingestion:
            try:
                # Clear the collection instead of deleting files
                db_market.delete_collection()
                print("Cleared existing market database for fresh ingestion")
            except Exception as e:
                print(f"Warning: Could not clear existing market collection: {e}")
                # Continue anyway as the ingest function will handle this
        
        ingest_sqlite_db(db_file_path, "market_prices")
    else:
        print("Market price data found, skipping ingestion.")

# Only ingest market data if it doesn't already exist
check_and_ingest(force_market_ingestion=False)

# Function to manually refresh market data if needed in the future
def refresh_market_data():
    """
    Manual utility function to force reingestion of market data.
    Call this function whenever you want to refresh the market data.
    Example usage:
    ```
    # In a Python terminal or script:
    from main import refresh_market_data
    refresh_market_data()
    ```
    """
    print("Manually refreshing market data...")
    try:
        # Clear the existing collection
        db_market.delete_collection()
        print("Cleared existing market database for fresh ingestion")
    except Exception as e:
        print(f"Warning: Could not clear existing market collection: {e}")
    
    # Re-ingest the data
    ingest_sqlite_db(db_file_path, "market_prices")
    print("Market data refresh complete")

# SQLite
conn = sqlite3.connect("chat_history.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    role TEXT,
    content TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

def store_message(user_id, role, content):
    cursor.execute("INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)", (user_id, role, content))
    conn.commit()

def get_user_messages(user_id, limit=10):
    """Retrieve recent messages for a given user."""
    cursor.execute(
        "SELECT role, content FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?", 
        (user_id, limit)
    )
    messages = cursor.fetchall()
    # Return messages in reverse order (oldest first)
    return [{"role": msg[0], "content": msg[1]} for msg in reversed(messages)]

# Weatherbit API
WEATHERBIT_KEY = os.getenv("WEATHERBIT_KEY") 

async def reverse_geocode(lat: float, lon: float):
    """Get city, district (if available), state using OpenCage API."""
    try:
        url = "https://api.opencagedata.com/geocode/v1/json"
        params = {"q": f"{lat},{lon}", "key": os.getenv("OPENCAGE_KEY")}
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(url, params=params)
            data = res.json()

            if "results" in data and len(data["results"]) > 0:
                comp = data["results"][0]["components"]

                city = comp.get("city") or comp.get("town") or comp.get("village")
                state = comp.get("state")
                # District may come under different keys
                district = comp.get("state_district") or comp.get("county") or comp.get("suburb")

                print(f"City: {city}, District: {district}, State: {state}")
                return {"city": city, "district": district, "state": state}
    except Exception as e:
        print(f"Error in reverse geocode: {e}")
    return {"city": None, "district": None, "state": None}


async def get_soil_moisture(lat: float, lon: float):
    try:
        url = "https://api.weatherbit.io/v2.0/forecast/agweather"
        params = {"lat": lat, "lon": lon, "key": WEATHERBIT_KEY}
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.get(url, params=params)
            data = res.json()

            import json
            # Print entire data for debugging
            print("===== FULL WEATHERBIT DATA =====")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            print("===== END DATA =====")

            if "data" in data and len(data["data"]) > 0:
                # Latest forecast (first element)
                latest = data["data"][0]
                # Longest forecast (last element)
                longest = data["data"][-1]

                print("===== LATEST FORECAST =====")
                print(json.dumps(latest, indent=2, ensure_ascii=False))
                print("===== LONGEST FORECAST =====")
                print(json.dumps(longest, indent=2, ensure_ascii=False))

                # Pick soil moisture fields
                moisture_0_10 = latest.get("soilm_0_10cm", "N/A")
                moisture_10_40 = latest.get("soilm_10_40cm", "N/A")
                moisture_40_100 = latest.get("soilm_40_100cm", "N/A")
                moisture_100_200 = latest.get("soilm_100_200cm", "N/A")
                temp = latest.get("temp_2m_avg", "N/A")
                precip = latest.get("precip", "N/A")
                
                return (
                    f"Latest Soil Moisture (mm): 0-10cm: {moisture_0_10}, "
                    f"10-40cm: {moisture_10_40}, 40-100cm: {moisture_40_100}, "
                    f"100-200cm: {moisture_100_200} | Temp: {temp}°C | Precip: {precip}mm"
                )

            return "Soil moisture data unavailable."
    except Exception as e:
        print(f"Error fetching soil moisture: {e}")
        return f"Error fetching soil moisture: {e}"


# Weatherbit Daily Forecast & Irrigation Advice
async def get_weather_forecast(lat: float, lon: float, days: int = 10):
    try:
        url = "https://api.weatherbit.io/v2.0/forecast/daily"
        params = {"lat": lat, "lon": lon, "days": days, "key": WEATHERBIT_KEY}

        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.get(url, params=params)
            data = res.json()

            import json
            print("===== FULL DAILY WEATHER DATA =====")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            print("===== END DATA =====")

            forecast_summary = []
            for day in data.get("data", []):
                date = day.get("datetime")
                temp = day.get("temp")
                precip = day.get("precip")
                rh = day.get("rh")
                wind = day.get("wind_spd")
                weather_desc = day.get("weather", {}).get("description", "")

                # Simple irrigation logic: irrigate if precipitation < 5mm and RH < 60%
                irrigation_needed = "Yes" if (precip is not None and precip < 5 and rh is not None and rh < 60) else "No"

                forecast_summary.append(
                    f"{date}: Temp={temp}°C, Precip={precip}mm, RH={rh}%, Wind={wind}m/s, "
                    f"Weather='{weather_desc}', Irrigation Needed={irrigation_needed}"
                )

            return "\n".join(forecast_summary) if forecast_summary else "Weather data unavailable."

    except Exception as e:
        print(f"Error fetching daily weather forecast: {e}")
        return f"Error fetching daily weather forecast: {e}"


def extract_city_from_query(query):
    match = re.search(r'\b(?:in|at)\s+([A-Za-z ]+)', query.lower())
    if match:
        return match.group(1).strip().title()
    return None


# @app.post("/ask")
# async def ask(user_id: str = Form(...), query: str = Form(...), lat: float = Form(...), lon: float = Form(...), k: int = Form(5)):
#     store_message(user_id, "user", query)

#     # Define keyword groups
#     weather_keywords = ["rain", "weather", "temperature", "forecast", "irrigate", "soil", "moisture"]
#     seed_keywords = ["seed", "variety", "crop", "disease resistance", "trial", "recommendation"]

#     if any(word in query.lower() for word in weather_keywords):
#         # WEATHER AGENT
#         soil_moisture = await get_soil_moisture(lat, lon)
#         weather_forecast = await get_weather_forecast(lat, lon)
#         retriever = db.as_retriever(search_kwargs={"k": k})
#         context_docs = retriever.get_relevant_documents(query)
#         weather_context = " ".join([d.page_content for d in context_docs])

#         ai_prompt = f"""
#         You are an agriculture assistant.
#         Soil & Weather Info: {soil_moisture}
#         Weather Forecast: {weather_forecast}
#         Crop Info: {weather_context}
#         Question: {query}
#         Answer concisely for a farmer.
#         """

#     elif any(word in query.lower() for word in seed_keywords):
#         # SEED VARIETY AGENT
#         retriever = db_seeds.as_retriever(search_kwargs={"k": k})
#         context_docs = retriever.get_relevant_documents(query)
#         seed_context = " ".join([d.page_content for d in context_docs])

#         ai_prompt = f"""
#         You are an agriculture assistant specializing in seed varieties.
#         Knowledge: {seed_context}
#         Question: {query}
#         Answer with variety recommendations, disease resistance, and official guidelines.
#         """


#     else:
#         # DEFAULT AGENT
#         ai_prompt = f"""
#         You are an agriculture assistant.
#         The farmer asked: {query}
#         Answer briefly.
#         """

#     # Get answer from Groq
#     response = client.chat.completions.create(
#         model="llama3-8b-8192",
#         messages=[
#             {"role": "system", "content": "You are a helpful agriculture assistant."},
#             {"role": "user", "content": ai_prompt}
#         ]
#     )
#     ai_content = response.choices[0].message.content

#     store_message(user_id, "assistant", ai_content)
#     return {"query": query, "response": ai_content}

# @app.post("/ask")
# async def ask(user_id: str = Form(...), query: str = Form(...), lat: float = Form(...), lon: float = Form(...), k: int = Form(5)):
#     store_message(user_id, "user", query)

#     # Step 1: Gather retrievals from *both* databases
#     retriever_main = db.as_retriever(search_kwargs={"k": k})
#     retriever_seeds = db_seeds.as_retriever(search_kwargs={"k": k})

#     main_docs = retriever_main.get_relevant_documents(query)
#     seed_docs = retriever_seeds.get_relevant_documents(query)

#     main_context = " ".join([d.page_content for d in main_docs])
#     seed_context = " ".join([d.page_content for d in seed_docs])

#     # Step 2: Check if query requires weather
#     weather_context = ""
#     if any(word in query.lower() for word in ["rain", "weather", "temperature", "forecast", "irrigate", "soil", "moisture"]):
#         soil_moisture = await get_soil_moisture(lat, lon)
#         weather_forecast = await get_weather_forecast(lat, lon)
#         weather_context = f"Soil & Moisture Data: {soil_moisture}\nForecast: {weather_forecast}"

#     # Step 3: Construct hybrid prompt
#     ai_prompt = f"""
#     You are an agriculture assistant for farmers. 
#     Use the following knowledge sources to answer:

#     ✅ General Agri Knowledge: {main_context}
#     🌱 Seed Variety Info: {seed_context}
#     🌦️ Weather & Soil Info: {weather_context}

#     Question: {query}

#     Rules:
#     - If seed recommendation is relevant, mention variety + traits (yield, disease resistance).
#     - If weather is relevant, explain irrigation / planting timing clearly.
#     - Merge multiple sources into ONE farmer-friendly answer.
#     - Keep it concise and practical.
#     """

#     # Step 4: Get response from Groq
#     response = client.chat.completions.create(
#         model="llama3-8b-8192",
#         messages=[
#             {"role": "system", "content": "You are a helpful agriculture assistant."},
#             {"role": "user", "content": ai_prompt}
#         ]
#     )

#     ai_content = response.choices[0].message.content
#     store_message(user_id, "assistant", ai_content)

#     return {"query": query, "response": ai_content}
# @app.post("/ask")
# async def ask(user_id: str = Form(...), query: str = Form(...), lat: float = Form(...), lon: float = Form(...), k: int = Form(5)):
#     store_message(user_id, "user", query)

#     # Step 1: Gather retrievals from *both* databases
#     retriever_main = db.as_retriever(search_kwargs={"k": k})
#     retriever_seeds = db_seeds.as_retriever(search_kwargs={"k": k})

#     main_docs = retriever_main.get_relevant_documents(query)
#     seed_docs = retriever_seeds.get_relevant_documents(query)

#     main_context = " ".join([d.page_content for d in main_docs])
#     seed_context = " ".join([d.page_content for d in seed_docs])

#     # Step 2: Check if query requires weather
#     weather_context = ""
#     if any(word in query.lower() for word in ["rain", "weather", "temperature", "forecast", "irrigate", "soil", "moisture"]):
#         soil_moisture = await get_soil_moisture(lat, lon)
#         weather_forecast = await get_weather_forecast(lat, lon)
#         weather_context = f"Soil & Moisture Data: {soil_moisture}\nForecast: {weather_forecast}"

#     # Step 3: Construct hybrid prompt
#     ai_prompt = f"""
#     You are an agriculture assistant for farmers. 
#     Use the following knowledge sources to answer:

#     ✅ General Agri Knowledge: {main_context}
#     🌱 Seed Variety Info: {seed_context}
#     🌦️ Weather & Soil Info: {weather_context}

#     Question: {query}

#     Rules:
#     - If seed recommendation is relevant, mention variety + traits (yield, disease resistance).
#     - If weather is relevant, explain irrigation / planting timing clearly.
#     - Merge multiple sources into ONE farmer-friendly answer.
#     - Keep it concise and practical.
#     """

#     # Step 4: Get response from Groq
#     response = client.chat.completions.create(
#         model="llama3-8b-8192",
#         messages=[
#             {"role": "system", "content": "You are a helpful agriculture assistant."},
#             {"role": "user", "content": ai_prompt}
#         ]
#     )

#     ai_content = response.choices[0].message.content
#     store_message(user_id, "assistant", ai_content)

#     return {"query": query, "response": ai_content}
@app.post("/ask")
async def ask(
    user_id: str = Form(...),
    query: str = Form(...),
    lat: float = Form(...),
    lon: float = Form(...),
    k: int = Form(5)
):
    # Store user message first
    store_message(user_id, "user", query)
    
    # Define the new system prompt for the agent with stronger anti-hallucination guidance
    system_prompt = """You are an agricultural assistant. Your goal is to provide accurate, context-aware answers about soil, crops, and farming.

CRITICAL RULES:
1. If the user's question is general or missing key details (e.g., 'When should I sell my crop?', 'What pesticides should I use?'), do not assume the crop or condition.
2. Instead, politely ask follow-up questions to collect the missing information first (e.g., 'Which crop are you asking about?', 'What stage of growth is your crop in?').
3. NEVER invent or hallucinate information that isn't provided in the context.
4. If you don't have specific data on a topic, clearly state that you don't have this information rather than making something up.
5. For market prices, weather conditions, and specific crop data, ONLY use information explicitly provided in the context.
6. Once all necessary details are clarified, then provide a helpful, precise answer based on facts.
7. If the question is complete and specific, answer directly with factual information.

Always keep the conversation clear, simple, and farmer-friendly."""
    
    # --- Greeting and Simple Query Handling ---
    greetings = ["hi", "hii", "hello", "hey", "hola", "greetings", "namaste", "good morning", "good afternoon", "good evening"]
    normalized_query = query.strip().lower()

    if normalized_query in greetings or any(greeting in normalized_query.split() for greeting in greetings):
        # Personalized greeting response
        ai_content = "Hello! I'm AgriSense, your agriculture assistant. How can I help with your farming today? I can provide weather forecasts, crop recommendations, or answer questions about agricultural practices."
        store_message(user_id, "assistant", ai_content)
        return {"query": query, "response": ai_content}

    # --- Capability Inquiry Handling ---
    capability_queries = ["what can you do", "what are your capabilities", "help me", "help", "what do you know", "how can you help"]
    if normalized_query in capability_queries or any(cap_query in normalized_query for cap_query in capability_queries):
        ai_content = """I can help you with:
- Weather forecasts and soil moisture conditions for your location
- Crop variety recommendations based on your region and climate
- State-specific agricultural guidelines and practices
- Market prices and best selling times for your crops
- Pest and disease management advice
- Irrigation scheduling based on weather forecasts
- General farming questions and techniques

Just ask me a specific question about your farm or crops!"""
        store_message(user_id, "assistant", ai_content)
        return {"query": query, "response": ai_content}
    
    # Get location details
    location_info = await reverse_geocode(lat, lon)
    state = location_info.get("state")
    district = location_info.get("district")
    city = location_info.get("city")

    # Check for vague or general queries that need clarification
    general_crop_patterns = [
        r'when should I (sell|plant|harvest|water) my crop',
        r'what (pesticides|fertilizer|nutrients) (should I use|do I need|are best)',
        r'how (often|much) (should I|do I need to) (water|irrigate|spray)',
        r'(is it|when is) (the right|a good) time to (harvest|sell|plant)',
        r'how to (protect|treat) my crop',
        r'what is (affecting|wrong with) my crop',
        r'which crop (should|can) (I|we) plant (this season|now|this month)',
        r'what crop (to|should|can) (I|we) (plant|grow|cultivate) (this season|now)',
        r'best crop for (this season|planting now)',
        r'recommend (a|some) crop (to plant|to grow)'
    ]
    
    # Check if query is too general and needs clarification
    if any(re.search(pattern, query.lower()) for pattern in general_crop_patterns):
        # Check previous messages to see if this is a follow-up to a clarification request
        previous_messages = get_user_messages(user_id, limit=4)
        
        # Extract crop name from the current query or recent messages
        mentioned_crop = None
        crops_list = ["wheat", "rice", "maize", "cotton", "soybean", "potato", "tomato", 
                      "onion", "sugarcane", "pulses", "chickpea", "mustard", "groundnut"]
        
        # First check current query
        for crop in crops_list:
            if crop in query.lower():
                mentioned_crop = crop
                break
                
        # If no crop in current query, check previous messages
        if not mentioned_crop and len(previous_messages) > 0:
            for msg in previous_messages:
                if msg['role'] == 'user':
                    for crop in crops_list:
                        if crop in msg['content'].lower():
                            mentioned_crop = crop
                            break
                    if mentioned_crop:
                        break
        
        # If we found a crop mention, process the query with that crop
        if mentioned_crop:
            print(f"Found crop in conversation: {mentioned_crop}")
            # Use the specific crop information to provide a response
            ai_prompt = f"""
            The user is asking about {mentioned_crop}. Their query is: "{query}"
            
            Please provide a specific, helpful response about {mentioned_crop} based on their question.
            Include information about:
            1. Best practices for {mentioned_crop} cultivation
            2. Current season suitability for {mentioned_crop} in {state or 'their region'}
            3. Basic care instructions
            
            Make your answer direct, practical, and farmer-friendly.
            """
            
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": ai_prompt}
                ],
                temperature=0.2,
                max_tokens=250
            )
            
            ai_content = response.choices[0].message.content
        else:
            # No crop mentioned - use the LLM to generate a follow-up question
            # Make the clarification request more specific
            ai_prompt = f"""
            The user asked: "{query}"
            
            This question needs clarification before I can give a helpful answer.
            Create a friendly response that:
            1. Acknowledges their question
            2. Asks which specific crop they're interested in
            3. Asks about their location or growing conditions if relevant
            4. Maintains a helpful, conversational tone
            
            Keep your response concise and focused on getting the necessary details.
            """
            
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful agricultural assistant who needs specific details to provide accurate advice."},
                    {"role": "user", "content": ai_prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            ai_content = response.choices[0].message.content
            
        store_message(user_id, "assistant", ai_content)
        return {"query": query, "response": ai_content}

    # Enhanced crop name extraction with better pattern matching
    crop_patterns = [
        r'\b(?:my|the)\s+([a-zA-Z ]+?)(?:\s+crop|\s+yield|\s+harvest|\s+field|\s+plantation|\s+farm|\s+seeds)\b',
        r'\bplanting\s+([a-zA-Z ]+)\b',
        r'\bgrow(?:ing)?\s+([a-zA-Z ]+)\b',
        r'\b([a-zA-Z ]+)\s+(?:farm|field|cultivation)\b'
    ]
    
    crop_name = None
    for pattern in crop_patterns:
        crop_match = re.search(pattern, query.lower())
        if crop_match:
            crop_name = crop_match.group(1).strip()
            break
    
    # If no crop found but query contains crop-related words, try direct matching
    if not crop_name and any(word in query.lower() for word in ["crop", "harvest", "yield", "sell", "market", "plant", "farm", "sow"]):
        crop_candidates = ["rice", "wheat", "corn", "maize", "soybean", "cotton", "sugarcane", "potato", "tomato", 
                          "onion", "chilli", "pepper", "mustard", "pulses", "dal", "groundnut", "peanut", "chickpea",
                          "millet", "bajra", "jowar", "ragi", "barley"]
        for crop in crop_candidates:
            if crop in query.lower():
                crop_name = crop
                break

    print(f"Identified crop: {crop_name}")

    # Special handler for queries about "crops in my area" - ensures direct, concise answers
    area_crop_patterns = [
        r'what crops? (are |can be |)(better|best|good|suitable|recommended) (for|in) (my|this) area',
        r'which crops? (should|can) (i|we|one|farmers|a farmer) (grow|plant|cultivate) (in|at) (my|this) area',
        r'crops? (for|in|suited to) (my|this) (area|region|location|place)',
        r'best crops? (to|for) plant(ing)? (now|currently) (in|at) (my|this) (region|area|location)'
    ]
    
    if any(re.search(pattern, query.lower()) for pattern in area_crop_patterns):
        # Check if we have previous messages from this user to determine context
        previous_messages = get_user_messages(user_id, limit=5)
        has_soil_info = any("soil" in msg['content'].lower() for msg in previous_messages if msg['role'] == 'user')
        has_location_specifics = any(re.search(r'(in|at|near|from) ([A-Za-z\s]+)', msg['content']) for msg in previous_messages if msg['role'] == 'user')
        
        # If we don't have specific information, ask follow-up questions
        if not has_soil_info or not has_location_specifics:
            ai_prompt = f"""
            The user is asking about suitable crops for their region, but we need more information.
            
            Current information:
            - Location: {state or city or 'Unknown'}
            - Soil type: {"Unknown" if not has_soil_info else "Mentioned in previous messages"}
            
            Generate a friendly response asking for their specific location and soil type (if not already provided).
            Mention that this information will help provide more accurate crop recommendations.
            Also ask if irrigation is available, as this affects crop selection.
            Keep the response conversational and farmer-friendly.
            """
            
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful agricultural assistant who needs specific details to provide accurate advice."},
                    {"role": "user", "content": ai_prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            ai_content = response.choices[0].message.content
            store_message(user_id, "assistant", ai_content)
            return {"query": query, "location": location_info, "response": ai_content}
        
        # If we have enough information from previous messages, proceed with recommendations
        # Get state-specific crop recommendations with higher relevance count
        retriever_state = db_states.as_retriever(search_kwargs={"k": 3})
        state_query = f"main crops grown in {state if state else 'this region'}"
        
        if state:
            state_docs = retriever_state.get_relevant_documents(state_query)
            state_context = " ".join([d.page_content for d in state_docs])
        else:
            state_context = ""
            
        # Also get general crop data
        retriever_main = db.as_retriever(search_kwargs={"k": 3})
        main_docs = retriever_main.get_relevant_documents(f"suitable crops for {state if state else 'cultivation'}")
        main_context = " ".join([d.page_content for d in main_docs])
        
        # Extract soil type from previous messages if available
        soil_type = "unknown"
        for msg in previous_messages:
            if msg['role'] == 'user':
                soil_match = re.search(r'soil (is|type) (clay|loamy|sandy|black|red|alluvial)', msg['content'].lower())
                if soil_match:
                    soil_type = soil_match.group(2)
                    break
        
        # Get the current month for seasonal recommendations
        current_month = datetime.now().strftime("%B")
        
        # Ultra-focused prompt for area crop recommendations
        ai_prompt = f"""
        You are responding to a farmer who wants to know what crops to plant in their region.

        **Farmer Information:**
        - Location: {state or city or 'Unspecified'}
        - Current month: {current_month}
        - Soil type: {soil_type}
        
        **Available Data:**
        - State crop information: {state_context}
        - General crop suitability: {main_context}
        
        Based on this information, provide a list of 3-5 crops that would be most suitable for planting now in their region.
        Include brief notes about why these crops are suitable based on the current season and their soil type.
        Format your response in a clear, organized way with bullet points.
        """
        
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are an agricultural expert providing crop recommendations based on location, season, and soil conditions. Be specific, informative, and helpful to farmers."},
                {"role": "user", "content": ai_prompt}
            ],
            temperature=0.3,
            max_tokens=250
        )
        ai_content = response.choices[0].message.content
        
        # Check if response has content
        if not ai_content or len(ai_content.split()) < 10:
            # Fallback for empty or very short responses
            crops_for_season = {
                "January": ["Wheat", "Mustard", "Potato", "Gram", "Peas"],
                "February": ["Wheat", "Mustard", "Gram", "Peas", "Summer vegetables"],
                "March": ["Summer vegetables", "Zaid crops", "Early paddy", "Groundnut", "Maize"],
                "April": ["Jute", "Cotton", "Summer vegetables", "Groundnut", "Maize"],
                "May": ["Cotton", "Jute", "Early paddy", "Summer vegetables", "Summer pulses"],
                "June": ["Paddy", "Cotton", "Maize", "Soybean", "Arhar"],
                "July": ["Paddy", "Maize", "Arhar", "Soybean", "Groundnut"],
                "August": ["Paddy", "Maize", "Arhar", "Soybean", "Vegetables"],
                "September": ["Paddy", "Vegetables", "Pulses", "Oilseeds", "Maize"],
                "October": ["Potato", "Vegetables", "Wheat", "Mustard", "Gram"],
                "November": ["Wheat", "Mustard", "Gram", "Potato", "Vegetables"],
                "December": ["Wheat", "Mustard", "Gram", "Potato", "Winter vegetables"]
            }
            
            current_month = datetime.now().strftime("%B")
            recommended_crops = crops_for_season.get(current_month, ["Paddy", "Wheat", "Maize", "Pulses", "Vegetables"])
            
            ai_content = f"Thank you! Since {state or city} in {current_month} has typical seasonal conditions, and your soil is {soil_type}, the best crops to plant right now include:\n\n"
            for crop in recommended_crops[:5]:
                ai_content += f"• {crop}\n"
                
            # Add a note about soil suitability
            soil_notes = {
                "sandy": "These recommendations work well in your sandy soil, though ensure good irrigation and organic matter addition.",
                "loamy": "Your loamy soil is excellent for these crops as it provides good drainage and nutrient retention.",
                "clay": "With your clay soil, ensure proper drainage for these crops and consider raised beds if waterlogging is common.",
                "black": "Your black soil is rich in nutrients and good for water retention, making it excellent for these crops.",
                "red": "For your red soil, consider adding organic matter to improve fertility for these recommended crops.",
                "alluvial": "Your alluvial soil is naturally fertile and well-suited for these recommended crops."
            }
            
            ai_content += f"\n{soil_notes.get(soil_type, '')}"
        
        store_message(user_id, "assistant", ai_content)
        return {"query": query, "location": location_info, "response": ai_content}

    # Special logic for handling unclear or vague queries
    if len(query.split()) < 3 and not any(keyword in query.lower() for keyword in ["weather", "rain", "crop", "soil", "plant", "harvest"]):
        # Check if this is a simple follow-up to a previous question
        previous_messages = get_user_messages(user_id, limit=4)
        
        # Check if there's a context from previous conversation
        if len(previous_messages) >= 2:
            last_assistant_msg = None
            for msg in reversed(previous_messages):
                if msg['role'] == 'assistant':
                    last_assistant_msg = msg['content']
                    break
            
            # Check if the last assistant message was asking for clarification
            if last_assistant_msg and ("could you please" in last_assistant_msg.lower() or 
                                      "can you provide" in last_assistant_msg.lower() or
                                      "what specific" in last_assistant_msg.lower()):
                
                # This is likely a response to our clarification request
                # Let's handle specific common keywords
                if any(crop in query.lower() for crop in ["wheat", "rice", "cotton", "maize", "potato"]):
                    crop_name = next((crop for crop in ["wheat", "rice", "cotton", "maize", "potato"] if crop in query.lower()))
                    
                    ai_prompt = f"""
                    The user has indicated they're interested in {crop_name}. Provide helpful information about:
                    1. Current season suitability for {crop_name} in {state or 'their region'}
                    2. Basic cultivation practices for {crop_name}
                    3. Common considerations for this crop
                    
                    Keep your response direct, practical and farmer-friendly.
                    """
                    
                    response = client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": ai_prompt}
                        ],
                        temperature=0.2,
                        max_tokens=250
                    )
                    
                    ai_content = response.choices[0].message.content
                    store_message(user_id, "assistant", ai_content)
                    return {"query": query, "response": ai_content}
                
                elif "irrigation" in query.lower():
                    # Handle irrigation query
                    ai_prompt = f"""
                    The user is asking about irrigation scheduling. Provide helpful information about:
                    1. General irrigation principles for crops in {state or 'their region'} during the current season
                    2. How to determine when crops need water
                    3. Efficient irrigation practices
                    
                    Keep your response direct, practical and farmer-friendly.
                    """
                    
                    response = client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": ai_prompt}
                        ],
                        temperature=0.2,
                        max_tokens=250
                    )
                    
                    ai_content = response.choices[0].message.content
                    store_message(user_id, "assistant", ai_content)
                    return {"query": query, "response": ai_content}
        
        # If no relevant context or not a follow-up, provide the generic guidance
        ai_content = """I'd like to help you with your farming questions! To provide the most useful information, could you please tell me:

1. Which crop(s) you're working with
2. Your specific question or concern
3. Your location (if relevant)

For example, you might ask:
• "When should I plant wheat in Bihar?"
• "How do I treat leaf spots on my tomato plants?"
• "What's the best irrigation method for cotton in sandy soil?"
"""
        store_message(user_id, "assistant", ai_content)
        return {"query": query, "response": ai_content}

    # Special logic for cold tolerance and temperature questions
    if any(term in query.lower() for term in ["temperature drop", "cold", "frost", "freezing", "winter", "chilling"]):
        # Retrieve cold tolerance info from seed DB with enhanced query
        retriever_cold = db_seeds.as_retriever(search_kwargs={"k": 3})
        cold_query = f"{crop_name if crop_name else ''} cold tolerance temperature resistance frost"
        cold_docs = retriever_cold.get_relevant_documents(cold_query)
        cold_context = " ".join([d.page_content for d in cold_docs])

        # Get next week's minimum temperature forecast
        forecast = await get_weather_forecast(lat, lon, days=7)
        min_temps = []
        for line in forecast.split('\n'):
            match = re.search(r'Temp=([0-9.]+)', line)
            if match:
                min_temps.append(float(match.group(1)))
        next_week_min_temp = min(min_temps) if min_temps else None

        # Enhanced prompt for cold tolerance questions
        ai_prompt = f"""
        You are an agricultural expert helping a farmer with concerns about cold temperatures.
        
        **CRITICAL INSTRUCTIONS:**
        1. Answer ONLY the user's question about cold tolerance/temperature effects.
        2. Be concise and practical in your advice.
        3. Only use information from the context - never make up data or recommendations.
        4. If context is insufficient, clearly state what information you don't have.
        
        **Farmer's Location:** {city or ''}, {district or ''}, {state or ''}
        **Crop Identified:** {crop_name or "Not specified"}
        **Farmer's Question:** "{query}"
        
        **Context:**
        - **Cold Tolerance Data:** {cold_context}
        - **Next 7 Days Minimum Temperature Forecast:** {next_week_min_temp}°C
        
        Format your answer as practical advice that a farmer can immediately use.
        """

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ai_prompt}
            ],
            temperature=0.2  # Lower temperature for more factual responses
        )
        ai_content = response.choices[0].message.content

        # Check if the response contains sufficient information
        if len(ai_content.split()) < 20 or "I don't have" in ai_content or "insufficient" in ai_content.lower():
            # Fallback to a more generic but helpful response about cold protection
            if crop_name:
                ai_content += f"\n\nGeneral cold protection measures for crops include covering plants with row covers at night, ensuring adequate soil moisture before temperature drops, applying mulch around plants, and creating wind barriers. Consider these methods to protect your {crop_name} if temperatures fall below freezing."

        store_message(user_id, "assistant", ai_content)
        return {"query": query, "location": location_info, "response": ai_content}

    # Enhanced logic for market price and selling questions
    if any(word in query.lower() for word in ["sell", "price", "market", "rates", "cost", "worth", "value", "mandi"]):
        # Check if this is a general query about "market prices for common crops" without specifying a crop
        is_general_price_query = ("common crops" in query.lower() or 
                                "crops" in query.lower() and "price" in query.lower() and not crop_name)
        
        # First, try to get specific market price data with enhanced query
        retriever_market = db_market.as_retriever(search_kwargs={"k": 8})
        
        if is_general_price_query:
            # For general queries, get prices for common crops
            market_query = f"price market rates for common crops in India {state if state else ''}"
            common_crops = ["rice", "wheat", "maize", "soybean", "cotton", "potato", "onion", "tomato"]
            
            # Create a consolidated market context with information about common crops
            consolidated_market_context = ""
            for crop in common_crops:
                crop_docs = retriever_market.get_relevant_documents(f"{crop} price market rates")
                if crop_docs:
                    crop_context = " ".join([d.page_content for d in crop_docs])
                    consolidated_market_context += f"\n{crop}: {crop_context}"
            
            market_context = consolidated_market_context
        else:
            # For specific crop queries
            market_query = f"{crop_name if crop_name else ''} price market rates {state if state else ''} {district if district else ''}"
            market_docs = retriever_market.get_relevant_documents(market_query)
            market_context = " ".join([d.page_content for d in market_docs])
        
        # Get weather forecast for next 7 days to help with selling decision
        weather_forecast = await get_weather_forecast(lat, lon, days=7)
        
        if is_general_price_query:
            # Special handling for general crop price queries
            ai_prompt = f"""
            You are responding to a general query about current market prices for common crops.
            
            **CRITICAL INSTRUCTIONS:**
            1. ONLY mention price information that is EXPLICITLY mentioned in the context.
            2. If you don't have current price data, clearly state this fact.
            3. DO NOT mention weather conditions unless the user specifically asked about them.
            4. DO NOT invent or hallucinate price data that isn't in the context.
            5. If price data is limited, suggest other resources the farmer could check.
            6. Format your answer as a clear list of crops and their current market prices.
            7. If no data is available, provide general advice on where farmers can check current prices.
            
            **Farmer's Location:** {city or ''}, {state or ''}
            **Farmer's Question:** "{query}"
            
            **Available Market Price Information:**
            {market_context}
            
            Provide only factual information about crop prices based strictly on the context provided.
            """
        else:
            # Original prompt for specific crop price queries
            ai_prompt = f"""
            You are an agricultural market advisor helping a farmer make an informed selling decision.
            
            **CRITICAL INSTRUCTIONS:**
            1. Use ONLY data from the context when discussing specific prices or market trends.
            2. If you don't have market data for the specific crop or location, clearly acknowledge this.
            3. Provide practical advice about timing of sales based on weather and market information.
            4. If the query isn't specifically about selling but about general pricing, focus on providing the available price information.
            5. Never invent prices or trends not supported by the context.
            
            **Farmer's Location:** {city or ''}, {district or ''}, {state or ''}
            **Farmer's Question:** "{query}"
            **Crop (if identified):** {crop_name or "Not specified"}
            
            **Context:**
            - **Market Price Data:** {market_context}
            - **Weather Forecast (next 7 days):** {weather_forecast}
            
            Respond with practical advice that helps the farmer make an informed decision about their crop sales.
            """

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are an agricultural expert providing strictly factual information about crop prices. Never make up data."},
                {"role": "user", "content": ai_prompt}
            ],
            temperature=0.1  # Even lower temperature for more factual responses
        )
        ai_content = response.choices[0].message.content

        # Extensive post-processing to catch hallucinations in market advice
        # Check for invented price references
        if "current price" in ai_content.lower() and "current price" not in market_context.lower():
            ai_content = ai_content.replace("current price", "typical price")
        
        # Check for currency symbols and numbers that aren't in the data
        if "₹" in ai_content and "₹" not in market_context:
            ai_content = re.sub(r'₹\s*[\d,]+(\.\d+)?', "[price data not available]", ai_content)
        
        # Check for invented weather impacts on market prices
        if is_general_price_query and ("thunderstorm" in ai_content.lower() or "rainfall" in ai_content.lower() or "weather forecast" in ai_content.lower()):
            # Remove weather references from general market price queries
            weather_phrases = [
                r'based on the weather forecast.*?\.', 
                r'the weather conditions.*?\.', 
                r'due to (the )?expected rainfall.*?\.', 
                r'thunderstorms? (might|may|could|will).*?\.', 
                r'looking at the weather forecast.*?\.', 
                r'monitor the weather.*?\.', 
                r'heavy rainfall.*?\.'
            ]
            
            for phrase in weather_phrases:
                ai_content = re.sub(phrase, '', ai_content, flags=re.IGNORECASE | re.DOTALL)
        
        # Check for hallucinated specific crop references in general queries
        if is_general_price_query and not any(crop in query.lower() for crop in ["rice", "wheat", "maize"]):
            # If a specific crop is prominently mentioned but wasn't in the query, remove it
            crops_not_mentioned = []
            for crop in ["rice", "wheat", "maize", "soybean", "cotton"]:
                if crop not in query.lower() and f"for {crop}" in ai_content.lower():
                    crops_not_mentioned.append(crop)
            
            if crops_not_mentioned:
                # Add a disclaimer if specific crops were hallucinated
                ai_content += "\n\nNote: I've provided general market price information. For specific prices of particular crops in your area, I recommend checking with your local agricultural extension office or mandi."
        
        # Final fallback for completely fabricated responses
        if is_general_price_query and not any(word in market_context.lower() for word in ["price", "market", "rate", "cost"]):
            ai_content = """I don't currently have access to up-to-date market prices for common crops. For the most accurate and current market prices, I recommend:

1. Checking your local Agricultural Produce Market Committee (APMC) or mandi
2. Contacting your district agriculture office
3. Using agricultural price apps like AgMarknet or Kisan Suvidha
4. Consulting local farmer producer organizations
5. Checking agricultural news websites for recent market trends

Market prices fluctuate daily based on supply, demand, and quality, so it's best to check these resources for the most current information."""

        store_message(user_id, "assistant", ai_content)
        return {"query": query, "location": location_info, "response": ai_content}
    
    # Enhanced logic for weather and irrigation questions
    if any(word in query.lower() for word in ["rain", "weather", "temperature", "forecast", "irrigate", "irrigation", "watering", "soil", "moisture", "wet", "dry", "drought"]):
        # Get detailed weather and soil moisture data
        soil_moisture = await get_soil_moisture(lat, lon)
        weather_forecast = await get_weather_forecast(lat, lon, days=10)  # Extended forecast
        
        # Additional context specific to irrigation needs
        retriever_main = db.as_retriever(search_kwargs={"k": 3})
        if crop_name:
            irrigation_query = f"{crop_name} irrigation water requirements"
            irrigation_docs = retriever_main.get_relevant_documents(irrigation_query)
            irrigation_context = " ".join([d.page_content for d in irrigation_docs])
        else:
            irrigation_context = ""
        
        # Enhanced prompt for weather-related questions
        ai_prompt = f"""
        You are an agricultural weather expert helping a farmer with weather-related decisions.
        
        **CRITICAL INSTRUCTIONS:**
        1. Focus specifically on the weather/irrigation question being asked.
        2. When discussing rainfall, use only the precipitation values from the weather forecast.
        3. For irrigation advice, consider both the upcoming weather forecast and soil moisture readings.
        4. Be specific about which days will likely have rain based on Precip values > 0mm.
        5. If asked about irrigation, give practical advice on whether and when to irrigate based on the forecast.
        
        **Farmer's Location:** {city or ''}, {district or ''}, {state or ''}
        **Crop (if identified):** {crop_name or "Not specified"}
        **Farmer's Question:** "{query}"
        
        **Context:**
        - **Soil Moisture Data:** {soil_moisture}
        - **Weather Forecast (next 10 days):** {weather_forecast}
        - **Crop Irrigation Info:** {irrigation_context}
        
        Provide a practical response that helps the farmer make an immediate decision about weather-related farming activities.
        """
        
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ai_prompt}
            ],
            temperature=0.2  # Lower temperature for more factual responses
        )
        ai_content = response.choices[0].message.content
        
        # Process weather data to add visual forecast indicator for improved readability
        if "forecast" in query.lower() or "weather" in query.lower() or "rain" in query.lower():
            try:
                forecast_lines = weather_forecast.split('\n')
                pattern = r'(.*): Temp=([\d.]+)°C, Precip=([\d.]+)mm, RH=([\d.]+)%'
                
                summary = "**Weather Summary:**\n"
                for i, line in enumerate(forecast_lines[:5]):  # First 5 days only
                    match = re.search(pattern, line)
                    if match:
                        date, temp, precip, humidity = match.groups()
                        emoji = "🌧️" if float(precip) > 0 else "☀️"
                        summary += f"{emoji} {date}: {temp}°C, Rain: {precip}mm, Humidity: {humidity}%\n"
                
                if "Irrigation Needed=Yes" in weather_forecast:
                    summary += "\n⚠️ **Irrigation may be needed** based on the forecast."
                
                # Check if this is a specific crop-related weather question
                previous_messages = get_user_messages(user_id, limit=4)
                mentioned_crop = None
                
                # Check current query for crop mentions
                crops_list = ["wheat", "rice", "maize", "cotton", "soybean", "potato", "tomato", 
                             "onion", "sugarcane", "pulses", "chickpea", "mustard", "groundnut"]
                
                for crop in crops_list:
                    if crop in query.lower():
                        mentioned_crop = crop
                        break
                
                # If no crop in current query, check previous messages
                if not mentioned_crop and len(previous_messages) > 0:
                    for msg in previous_messages:
                        if msg['role'] == 'user':
                            for crop in crops_list:
                                if crop in msg['content'].lower():
                                    mentioned_crop = crop
                                    break
                            if mentioned_crop:
                                break
                
                # If we have a specific crop, provide weather implications for that crop
                if mentioned_crop:
                    # Prepare a more specific response about weather for this crop
                    crop_weather_ai_prompt = f"""
                    The user is asking about weather conditions for {mentioned_crop}. Their query is: "{query}"
                    
                    Based on the current weather forecast:
                    {summary}
                    
                    Provide specific advice about:
                    1. How these weather conditions may affect {mentioned_crop} cultivation
                    2. Any precautions the farmer should take given the forecast
                    3. Optimal timing for planting or harvesting based on the weather (if relevant)
                    
                    Keep your response practical, specific, and farmer-friendly.
                    """
                    
                    crop_weather_response = client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[
                            {"role": "system", "content": "You are an expert agricultural advisor specializing in crop-specific weather implications."},
                            {"role": "user", "content": crop_weather_ai_prompt}
                        ],
                        temperature=0.2,
                        max_tokens=250
                    )
                    
                    # Replace the generic response with the crop-specific one
                    ai_content = crop_weather_response.choices[0].message.content
                    
                    # Make sure the weather summary is included
                    if not any(emoji in ai_content for emoji in ["🌧️", "☀️"]):
                        ai_content += f"\n\n{summary}"
                else:
                    # Append generic weather summary at the end of the AI response if not already included
                    if not any(emoji in ai_content for emoji in ["🌧️", "☀️"]):
                        ai_content += f"\n\n{summary}"
            except Exception as e:
                print(f"Error formatting weather summary: {e}")
        
        store_message(user_id, "assistant", ai_content)
        return {"query": query, "location": location_info, "response": ai_content}

    # Enhanced seed/variety recommendation logic
    if any(term in query.lower() for term in ["seed", "variety", "varieties", "which crop", "what crop", "recommend", "suggestion", "plant", "sow", "grow"]):
        # Get both general and seed-specific information
        retriever_main = db.as_retriever(search_kwargs={"k": 3})
        retriever_seeds = db_seeds.as_retriever(search_kwargs={"k": 4})
        
        # Improve query with location context
        state_specific_query = f"crops suitable for {state if state else 'this region'}"
        
        # If crop name is identified, use it, otherwise use a more general query
        if crop_name:
            seed_query = f"{crop_name} varieties recommended for {state if state else 'cultivation'}"
        else:
            seed_query = f"recommended crop varieties for {state if state else 'farming'}"
        
        main_docs = retriever_main.get_relevant_documents(state_specific_query)
        seed_docs = retriever_seeds.get_relevant_documents(seed_query)
        
        main_context = " ".join([d.page_content for d in main_docs])
        seed_context = " ".join([d.page_content for d in seed_docs])
        
        # Get state-specific crop recommendations
        state_query = f"crops suitable for {state if state else 'cultivation'} farming"
        retriever_state = db_states.as_retriever(search_kwargs={"k": 2})
        if state:
            state_docs = retriever_state.get_relevant_documents(state_query)
            state_context = " ".join([d.page_content for d in state_docs])
        else:
            state_context = ""
        
        # Get local weather to help with seasonal variety selection (minimal info)
        weather_forecast = await get_weather_forecast(lat, lon, days=3)
        
        # Extract current month/season for seasonal recommendations
        import datetime
        current_month = datetime.datetime.now().strftime("%B")
        
        # Special prompt for seed variety recommendations with emphasis on brevity
        ai_prompt = f"""
        You are an agricultural expert providing crop recommendations.
        
        **CRITICAL INSTRUCTIONS:**
        1. Be EXTREMELY CONCISE - limit to 3-5 crop recommendations maximum.
        2. Answer DIRECTLY what crops are suitable for the farmer's location.
        3. DO NOT provide extensive details about each crop unless specifically asked.
        4. Format as a simple, short list with minimal explanation.
        5. DO NOT include yield numbers, disease resistance details, or extensive characteristics unless specifically requested.
        6. If multiple varieties exist, mention ONLY the names without details.
        7. Never make up crop recommendations - if no data exists for a region, say so.
        
        **Current Month:** {current_month}
        **Farmer's Location:** {city or ''}, {district or ''}, {state or ''}
        **Farmer's Question:** "{query}"
        
        **Context:**
        - **Regional Crop Data:** {main_context}
        - **Seed Variety Information:** {seed_context}
        - **State-Specific Guidelines:** {state_context}
        
        Your response must be a simple, direct list of recommended crops suitable for this location right now.
        """
        
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ai_prompt}
            ],
            temperature=0.1,  # Lower temperature for more predictable, factual responses
            max_tokens=250  # Limit response length to ensure conciseness
        )
        ai_content = response.choices[0].message.content
        
        # Post-process to ensure brevity - if response is too long, truncate and simplify
        if len(ai_content.split()) > 100:
            # Try to extract just the crop list
            crop_list_pattern = r"(?:recommend|suitable|best|good)\s+crops[^:]*:([^\.]+)"
            crop_match = re.search(crop_list_pattern, ai_content, re.IGNORECASE)
            
            if crop_match:
                crops = crop_match.group(1).strip()
                ai_content = f"Based on your location, the following crops are recommended: {crops}"
            else:
                # Fallback - just keep the first paragraph
                first_para = ai_content.split('\n\n')[0]
                ai_content = first_para
        
        # Remove any hallucinated yield statistics
        ai_content = re.sub(r'\d+(?:-\d+)?\s*(?:tons|quintals|kg)(?:/hectare|/acre)?', "good yield", ai_content)
        
        # Remove any made-up disease resistance claims if they're not in the context
        if "disease resistance" not in seed_context and "disease resistance" in ai_content.lower():
            ai_content = re.sub(r'(?:resistant|resistance) to [^\.]+', "disease resistance", ai_content)
        
        store_message(user_id, "assistant", ai_content)
        return {"query": query, "location": location_info, "response": ai_content}

    # General retrieval and prompt construction for other types of questions
    retriever_main = db.as_retriever(search_kwargs={"k": 3})  # Increased for better coverage
    retriever_seeds = db_seeds.as_retriever(search_kwargs={"k": 2})
    retriever_market = db_market.as_retriever(search_kwargs={"k": 2})
    
    main_query = query
    seed_query = f"{crop_name if crop_name else ''} {query}"
    
    main_docs = retriever_main.get_relevant_documents(main_query)
    seed_docs = retriever_seeds.get_relevant_documents(seed_query)
    market_docs = retriever_market.get_relevant_documents(query)

    main_context = " ".join([d.page_content for d in main_docs])
    seed_context = " ".join([d.page_content for d in seed_docs])
    market_context = " ".join([d.page_content for d in market_docs])

    state_context = ""
    if state:
        retriever_state = db_states.as_retriever(search_kwargs={"k": 3, "filter": {"state": state}})
        state_docs = retriever_state.get_relevant_documents(query)
        state_context = " ".join([d.page_content for d in state_docs])

    weather_context = ""
    if any(word in query.lower() for word in ["rain", "weather", "temperature", "forecast", "irrigate", "soil", "moisture"]):
        soil_moisture = await get_soil_moisture(lat, lon)
        weather_forecast = await get_weather_forecast(lat, lon)
        weather_context = f"Soil & Moisture: {soil_moisture}\nForecast: {weather_forecast}"

    # Improved general prompt with better guidance and emphasis on brevity
    ai_prompt = f"""
    You are AgriSense, a specialized agriculture assistant helping farmers with direct, practical advice.
    
    **CRITICAL INSTRUCTIONS:**
    1. Answer the farmer's question DIRECTLY in as few words as possible.
    2. Be extremely concise - aim for 2-3 sentences when possible.
    3. Use only information from the context provided - never invent data.
    4. If you don't have specific information, say "I don't have enough information about X" and stop.
    5. No lengthy explanations - provide simple, actionable advice only.
    6. Never include statistics, numbers, or percentages unless they appear in the context.
    
    **Farmer's Location:** {state or ''}, {district or ''}, {city or ''}
    **Crop (if identified):** {crop_name or "Not specified"}
    **Farmer's Question:** "{query}"
    
    **Available Context:**
    - **General Agricultural Knowledge:** {main_context}
    - **Seed Variety Information:** {seed_context}
    - **State-Specific Guidelines:** {state_context}
    - **Market Information:** {market_context}
    - **Weather Information:** {weather_context}
    
    Provide a brief, focused answer using only the information from the context. If the context doesn't contain relevant information, simply state "I don't have specific information about that" instead of inventing an answer.
    """

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ai_prompt}
            ],
            temperature=0.1,  # Lower temperature for more predictable, factual responses
            max_tokens=200  # Strict token limit to force brevity
        )
        ai_content = response.choices[0].message.content
        
        # Enhanced post-processing to catch common hallucinations
        hallucination_patterns = [
            (r'\b(\d+)\s*kg\s*per\s*hectare\b', "appropriate amount per hectare"),
            (r'\b(\d+)\s*kg\s*per\s*acre\b', "appropriate amount per acre"),
            (r'\b(\d+)%\s*yield\s*increase\b', "potential yield improvement"),
            (r'\b(\d+)\s*days\s*to\s*harvest\b', "typical time to harvest"),
            (r'\b₹\s*[\d,]+(\.\d+)?\b', "market price")
        ]
        
        # Automatic response simplification if too verbose
        if len(ai_content.split()) > 75:  # If response has more than 75 words
            # Keep first paragraph only
            paragraphs = ai_content.split('\n\n')
            if len(paragraphs) > 1:
                ai_content = paragraphs[0]
        
        # Remove hallucinated statistics
        if not any(pattern in ai_prompt.lower() for pattern, _ in hallucination_patterns):
            for pattern, replacement in hallucination_patterns:
                if re.search(pattern, ai_content) and not re.search(pattern, main_context + seed_context + market_context):
                    ai_content = re.sub(pattern, replacement, ai_content)
        
        # Check for overconfident statements without supporting context
        if "best variety" in ai_content.lower() and "best variety" not in ai_prompt.lower():
            ai_content = ai_content.replace("best variety", "suitable variety")
            
        if "water requirement" in ai_content.lower() and "water requirement" not in ai_prompt.lower():
            ai_content = ai_content.replace("water requirement", "typical water needs")
            
        # Remove any overly technical terminology
        technical_terms = [
            (r'photosynthesis rate', 'growth'),
            (r'genetic modification', 'breeding'),
            (r'physiological stress', 'stress'),
            (r'agronomic practices', 'farming methods'),
            (r'phenological stages', 'growth stages')
        ]
        
        for tech_term, simple_term in technical_terms:
            ai_content = re.sub(tech_term, simple_term, ai_content, flags=re.IGNORECASE)
        
        # Remove any additional hallucination indicators
        if "according to research" in ai_content.lower() and "according to research" not in (main_context + seed_context + state_context).lower():
            ai_content = ai_content.replace("According to research", "Generally")
            
        store_message(user_id, "assistant", ai_content)
        
    except Exception as e:
        ai_content = "I'm sorry, I encountered an error processing your question. Please try again with more details about your specific farming situation, crop, or concern."
        print(f"Error in /ask endpoint: {e}")
        store_message(user_id, "assistant", ai_content)
        
    return {"query": query, "location": location_info, "response": ai_content}
@app.get("/history")
async def get_history(user_id: str, limit: int = 50):
    cursor.execute("SELECT role, content, timestamp FROM messages WHERE user_id = ? ORDER BY id ASC LIMIT ?", (user_id, limit))
    rows = cursor.fetchall()
    history = [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]
    return {"history": history}

@app.delete("/message/{message_id}")
async def delete_message(message_id: int):
    """Delete a specific message from chat history"""
    cursor.execute("DELETE FROM messages WHERE id = ?", (message_id,))
    conn.commit()
    return {"success": True, "message": "Message deleted"}

@app.get("/test_crops_query")
async def test_crops_query():
    """Test endpoint for crop recommendation improvement"""
    test_queries = [
        "what crops are better in my area",
        "which crops should I plant in my area",
        "recommended crops for my region",
        "best crops for my land"
    ]
    
    # Use a fixed location for testing
    test_lat = 28.6139  # Delhi coordinates
    test_lon = 77.2090
    
    results = {}
    
    for query in test_queries:
        # Process through the ask endpoint logic without storing in DB
        location_info = await reverse_geocode(test_lat, test_lon)
        
        # Trigger the area crop patterns
        area_crop_patterns = [
            r'what crops? (are |can be |)(better|best|good|suitable|recommended) (for|in) (my|this) area',
            r'which crops? (should|can) (i|we|one|farmers|a farmer) (grow|plant|cultivate) (in|at) (my|this) area',
            r'crops? (for|in|suited to) (my|this) (area|region|location|place)'
        ]
        
        if any(re.search(pattern, query.lower()) for pattern in area_crop_patterns):
            state = location_info.get("state")
            city = location_info.get("city")
            
            # Get state-specific crop recommendations
            retriever_state = db_states.as_retriever(search_kwargs={"k": 3})
            state_query = f"main crops grown in {state if state else 'this region'}"
            
            if state:
                state_docs = retriever_state.get_relevant_documents(state_query)
                state_context = " ".join([d.page_content for d in state_docs])
            else:
                state_context = ""
                
            # Also get general crop data
            retriever_main = db.as_retriever(search_kwargs={"k": 3})
            main_docs = retriever_main.get_relevant_documents(f"suitable crops for {state if state else 'cultivation'}")
            main_context = " ".join([d.page_content for d in main_docs])
            
            # Ultra-focused prompt for area crop recommendations
            ai_prompt = f"""
            Answer ONLY with a simple list of crops suitable for the farmer's area.

            **CRITICAL RULES:**
            1. List ONLY 3-5 crops total, no more
            2. NO explanations - just name the crops
            3. DO NOT mention yield, disease resistance, or characteristics
            4. Format as a simple bullet list
            5. If no data is available, say "I don't have enough information about crops suited for {state or city or 'your area'}"

            **Location:** {state or city or 'Unspecified'}
            
            **Available Data:**
            - State crop information: {state_context}
            - General crop suitability: {main_context}
            
            Respond with ONLY the names of 3-5 suitable crops in a simple bullet list.
            """
            
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are an agricultural assistant providing crop recommendations for Indian farmers."},
                    {"role": "user", "content": ai_prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            ai_content = response.choices[0].message.content
            
            # Test the simplification post-processing
            if not ai_content.strip().startswith("•") and not ai_content.strip().startswith("-") and not ai_content.strip().startswith("*"):
                crops_mentioned = re.findall(r'\b(?:rice|wheat|maize|cotton|pulses|sugarcane|millet|soybean|groundnut|potato|onion|tomato|mustard|barley|sorghum|ragi|jowar|bajra)\b', ai_content.lower())
                
                if crops_mentioned:
                    unique_crops = list(set(crops_mentioned))[:5]
                    ai_content = "Based on your location, these crops are suitable:\n" + "\n".join([f"• {crop.title()}" for crop in unique_crops])
                elif "don't have" not in ai_content.lower() and "no information" not in ai_content.lower():
                    ai_content = f"I don't have specific crop recommendations for {state or city or 'your area'}. Please provide more details about your soil type and farming conditions."
            
            results[query] = {
                "location": f"{city}, {state}",
                "response": ai_content,
                "word_count": len(ai_content.split())
            }
    
    return {"test_results": results}

@app.post("/flag_hallucination")
async def flag_hallucination(user_id: str = Form(...), message_content: str = Form(...)):
    """Flag a message as containing hallucinated content"""
    # Log the hallucination for future model improvements
    cursor.execute(
        "INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)", 
        (user_id, "system", f"HALLUCINATION_FLAG: {message_content}")
    )
    conn.commit()
    
    # Return a corrected message
    correction = "I apologize for providing inaccurate information earlier. Without specific data about your location and conditions, I can't make precise crop recommendations. Please provide more details about your soil type, climate, and farming goals so I can assist you better."
    return {"success": True, "correction": correction}