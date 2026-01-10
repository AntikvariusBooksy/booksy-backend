import os
import difflib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# --- KONFIGURÁCIÓ ---
load_dotenv()
INDEX_NAME = "booksy-index"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class BooksyBrain:
    def __init__(self):
        self.api_key_openai = os.getenv("OPENAI_API_KEY")
        self.api_key_pinecone = os.getenv("PINECONE_API_KEY")
        self.client_ai = OpenAI(api_key=self.api_key_openai)
        self.pc = Pinecone(api_key=self.api_key_pinecone)
        self.index = self.pc.Index(INDEX_NAME)

        self.store_policy = """
        [SZÁLLÍTÁS: Feldolgozás (2-4 nap raktár / 7-30 nap külső) + Futár (24-48h RO, 2-4 nap HU).]
        """
        
        self.system_prompt = f"""
        Te Booksy vagy, Sales Agent.
        TUDÁSBÁZIS: {self.store_policy}
        SZABÁLYOK:
        - Ha könyvet ajánlasz, írj egy nagyon rövid, vonzó bevezetőt.
        - TILTOTT: Ne sorold fel a könyveket listában (1. Cím - Ár...), mert a rendszer kártyákon jeleníti meg őket alattad!
        - Csak a tartalomra fókuszálj.
        - Ha nincs találat, legyél udvarias.
        """

    def generate_search_params(self, user_input):
        try:
            response = self.client_ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Döntsd el: 'SEARCH' (könyv) vagy 'INFO'. Válasz: 'hu | SEARCH | kulcsszó'"},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.1
            )
            parts = response.choices[0].message.content.split('|')
            return parts[0].strip().lower(), parts[1].strip(), parts[2].strip()
        except:
            return "hu", "SEARCH", user_input

    def search_books(self, query_text, lang_filter):
        try:
            response = self.client_ai.embeddings.create(input=query_text, model="text-embedding-3-small")
            search_results = self.index.query(
                vector=response.data[0].embedding,
                top_k=20, include_metadata=True,
                filter={"stock": "instock", "lang": lang_filter}
            )
            return search_results
        except:
            return {"matches": []}

    def process_message(self, user_input):
        detected_lang, intent, keywords = self.generate_search_params(user_input)
        
        context_text = ""
        found_products = [] # Itt gyűjtjük a kártyák adatait

        if intent == "SEARCH":
            results = self.search_books(keywords, detected_lang)
            seen_titles = []
            
            # --- BIZTONSÁGI ZÁR ---
            if not results.get('matches'):
                msg = "Din păcate, nu am găsit cărți." if detected_lang == 'ro' else "Sajnos nem találtam ilyen könyvet."
                return {"reply": msg, "products": []}

            for match in results['matches']:
                if match['score'] < 0.35: continue
                meta = match['metadata']
                title = str(meta.get('title', 'N/A'))
                
                # Duplikáció szűrés
                is_dup = False
                for seen in seen_titles:
                    if difflib.SequenceMatcher(None, title.lower(), seen.lower()).ratio() > 0.85:
                        is_dup = True; break
                if is_dup: continue
                seen_titles.append(title)
                
                # ADATGYŰJTÉS A KÁRTYÁKHOZ
                product_data = {
                    "title": title,
                    "price": meta.get('price', 'N/A'),
                    "url": meta.get('url', '#'),
                    "image": meta.get('image_url', '') # Itt van a kép linkje!
                }
                found_products.append(product_data)
                
                context_text += f"- {title}\n" # Az AI-nak csak a címet adjuk
                if len(found_products) >= 6: break
            
            if not found_products:
                msg = "Din păcate, nu am găsit nimic relevant." if detected_lang == 'ro' else "Sajnos nem találtam releváns könyvet."
                return {"reply": msg, "products": []}

        else:
            context_text = "HASZNÁLD A TUDÁSBÁZIST!"

        # Válasz generálás
        lang_instruction = "Reply in ROMANIAN only!" if detected_lang == 'ro' else "Reply in HUNGARIAN only!"
        
        response = self.client_ai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "system", "content": lang_instruction},
                {"role": "user", "content": f"User: {user_input}\nContext (Found Books): {context_text}"}
            ],
            temperature=0.3
        )
        
        # VISSZATÉRÉS: Szöveg + Terméklista
        return {
            "reply": response.choices[0].message.content,
            "products": found_products
        }

bot = BooksyBrain()

@app.get("/")
def home(): return {"status": "Booksy V2 (With Cards)"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    return bot.process_message(request.message)