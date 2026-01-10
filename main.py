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

    def generate_search_params(self, user_input):
        try:
            # OKOS ELEMZÉS: Nyelv + Szándék + Keresési Hatókör (Scope)
            response = self.client_ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """
                     Analyze the user's query.
                     1. Language: 'hu' or 'ro'.
                     2. Intent: 'SEARCH' (books) or 'INFO' (shipping, contact).
                     3. Scope: If user says "minden nyelven", "toate limbile", "all books", "bármilyen nyelv" -> 'ALL'. Otherwise -> 'SPECIFIC'.
                     4. Keywords: extract search terms.
                     
                     Output Format: LANG | SCOPE | INTENT | KEYWORDS
                     Example: hu | ALL | SEARCH | Harry Potter
                     Example: ro | SPECIFIC | SEARCH | Sadoveanu
                     """},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.1
            )
            parts = response.choices[0].message.content.split('|')
            return parts[0].strip().lower(), parts[1].strip(), parts[2].strip(), parts[3].strip()
        except:
            return "hu", "SPECIFIC", "SEARCH", user_input

    def search_books(self, query_text, lang_filter, scope):
        try:
            response = self.client_ai.embeddings.create(input=query_text, model="text-embedding-3-small")
            
            # --- FILTER LOGIKA ---
            filter_criteria = {"stock": "instock"}
            
            # Csak akkor szűrünk nyelvre, ha a Scope NEM 'ALL'
            if scope != 'ALL' and lang_filter in ['hu', 'ro']:
                filter_criteria["lang"] = lang_filter
            
            search_results = self.index.query(
                vector=response.data[0].embedding,
                top_k=25, 
                include_metadata=True,
                filter=filter_criteria
            )
            return search_results
        except Exception as e:
            print(f"Keresési hiba: {e}")
            return {"matches": []}

    def process_message(self, user_input):
        # 1. Elemzés (Most már a SCOPE-ot is kinyerjük)
        detected_lang, scope, intent, keywords = self.generate_search_params(user_input)
        
        context_text = ""
        found_products = [] 
        
        # Lábléc szövegek (A kötelező tájékoztató)
        footer_hu = "\n\n💡 *Tipp: Jelenleg a nyelvednek megfelelő könyveket keresem. Ha mindent látni szeretnél, írd hozzá: „minden nyelven”!*"
        footer_ro = "\n\n💡 *Sfat: Caut cărți în limba ta. Dacă vrei să vezi toate limbile, adaugă: „toate limbile”!*"

        if intent == "SEARCH":
            # 2. Keresés (átadjuk a scope-ot is)
            results = self.search_books(keywords, detected_lang, scope)
            seen_titles = []
            
            # Ha nincs találat
            if not results.get('matches'):
                msg = "Nu am găsit rezultate." if detected_lang == 'ro' else "Sajnos nem találtam ilyen könyvet."
                # Itt is kiírjuk a tippet, hátha azért nem talált, mert rossz nyelven kereste
                tip = footer_ro if detected_lang == 'ro' else footer_hu
                return {"reply": msg + tip, "products": []}

            for match in results['matches']:
                if match['score'] < 0.40: continue
                meta = match['metadata']
                title = str(meta.get('title', 'N/A'))
                
                # Duplikáció szűrés
                is_dup = False
                for seen in seen_titles:
                    if difflib.SequenceMatcher(None, title.lower(), seen.lower()).ratio() > 0.85:
                        is_dup = True; break
                if is_dup: continue
                seen_titles.append(title)
                
                # Termék adat
                product_data = {
                    "title": title,
                    "price": meta.get('price', 'N/A'),
                    "url": meta.get('url', '#'),
                    "image": meta.get('image_url', '') 
                }
                found_products.append(product_data)
                
                # Context az AI-nak (zárójelben a nyelvvel, ha vegyes a lista)
                lang_tag = meta.get('lang', '?')
                context_text += f"- {title} ({lang_tag})\n"
                
                if len(found_products) >= 6: break
            
            if not found_products:
                msg = "Nu am găsit nimic relevant." if detected_lang == 'ro' else "Sajnos nem találtam releváns könyvet."
                tip = footer_ro if detected_lang == 'ro' else footer_hu
                return {"reply": msg + tip, "products": []}

        else:
            context_text = "HASZNÁLD A TUDÁSBÁZIST!"

        # 3. Válasz generálás
        if detected_lang == 'ro':
            lang_instruction = "Reply in ROMANIAN only."
        else:
            lang_instruction = "Reply in HUNGARIAN only."

        response = self.client_ai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Te Booksy vagy. {self.store_policy}"},
                {"role": "system", "content": lang_instruction},
                {"role": "user", "content": f"User: {user_input}\nFound Books:\n{context_text}"}
            ],
            temperature=0.3
        )
        
        final_reply = response.choices[0].message.content
        
        # 4. Footer hozzáadása (Ha NEM kért kifejezetten mindent)
        if scope != 'ALL':
            if detected_lang == 'ro':
                final_reply += footer_ro
            else:
                final_reply += footer_hu
        
        return {
            "reply": final_reply,
            "products": found_products
        }

bot = BooksyBrain()

@app.get("/")
def home(): return {"status": "Booksy V5 (All Languages Support)"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    return bot.process_message(request.message)