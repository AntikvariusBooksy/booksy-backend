import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

load_dotenv()
api_key_openai = os.getenv("OPENAI_API_KEY")
api_key_pinecone = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=api_key_pinecone)
index = pc.Index("booksy-index")
client_ai = OpenAI(api_key=api_key_openai)

def xray_search(query):
    print(f"\n🔎 KERESÉS ERRE: '{query}'")
    print("-" * 40)
    
    # Vektorizálás
    response = client_ai.embeddings.create(input=query, model="text-embedding-3-small")
    vec = response.data[0].embedding
    
    # Nyers keresés (minden szűrő nélkül)
    results = index.query(vector=vec, top_k=5, include_metadata=True)
    
    for match in results['matches']:
        m = match['metadata']
        title = m.get('title', 'Nincs cím')
        stock = m.get('stock', 'Nincs infó')
        
        # Kiírjuk a pontos értéket, idézőjelek között, hogy lássuk, ha van szóköz!
        print(f"📚 KÖNYV: {title}")
        print(f"   KÉSZLET ADAT (Stock): '{stock}'") 
        print("-" * 40)

if __name__ == "__main__":
    xray_search("kertészet")
    xray_search("erdély")
    xray_search("regény")