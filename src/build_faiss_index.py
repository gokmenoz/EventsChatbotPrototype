import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from utils import parse_date_range, fetch_events

# -------- Settings --------
city = "Berlin"
query = f"What's happening in {city} this weekend?"
start, end = parse_date_range(query)

print(f"📅 Fetching events in {city} from {start} to {end}")
events = fetch_events(city, start, end)

# -------- Embedding --------
print(f"🧠 Embedding {len(events)} events...")
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [e["title"] + "\n\n" + e.get("description", "") for e in events]
embeddings = model.encode(texts, convert_to_numpy=True)

# -------- Build FAISS Index --------
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

os.makedirs("faiss_index", exist_ok=True)
faiss.write_index(index, "faiss_index/index.faiss")
with open("faiss_index/chunks.pkl", "wb") as f:
    pickle.dump(events, f)

print(f"✅ Saved faiss_index with {len(events)} events")