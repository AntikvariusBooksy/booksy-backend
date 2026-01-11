import os
import difflib
import unicodedata
import re
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

class HookRequest(BaseModel):
    url: str
    page_title: str
    visitor_type: str 
    cart_status: str 
    lang: str

# --- SEGÉDFÜGGVÉNYEK ---
def normalize_text(text):
    if not text: return ""
    text = str(text).lower()
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

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

    def generate_sales_hook(self, ctx: HookRequest):
        try:
            response = self.client_ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Context: Lang {ctx.lang}, Page {ctx.page_title}, Cart {ctx.cart_status}. Generate short sales hook (max 6 words)."},
                    {"role": "user", "content": "Hook me."}
                ],
                temperature=0.7, max_tokens=30
            )
            return response.choices[0].message.content.strip()
        except:
            return "Bună! Te pot ajuta?" if ctx.lang == 'ro' else "Szia! Segíthetek?"

    def generate_search_params(self, user_input):
        try:
            response = self.client_ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """
                     Analyze user query.
                     1. Language (hu/ro).
                     2. Intent (SEARCH/INFO).
                     3. Scope (ALL/SPECIFIC).
                     4. KEYWORDS: Keep names intact!
                     Output: LANG | SCOPE | INTENT | KEYWORDS
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
            
            # 1. KÖTELEZŐ KÉSZLET SZŰRÉS (Hogy ne legyen outofstock)
            filter_criteria = {"stock": "instock"}
            if scope != 'ALL' and lang_filter in ['hu', 'ro']:
                filter_criteria["lang"] = lang_filter
            
            raw_results = self.index.query(
                vector=response.data[0].embedding,
                top_k=100, 
                include_metadata=True, 
                filter=filter_criteria
            )

            matches = raw_results.get('matches', [])
            if not matches: return {"matches": []}

            # --- 2. V17 INTELLIGENS VÁLOGATÁS ---
            
            stop_words = ['konyv', 'konyvek', 'konyvet', 'carte', 'carti', 'keresek', 'kiado', 'szerzo', 'cim']
            normalized_query = normalize_text(query_text)
            # Kulcsszavak (pl. "berente", "agi")
            search_keywords = [w for w in normalized_query.split() if w not in stop_words and len(w) > 2]

            final_results = []
            seen_titles = set()

            for match in matches:
                meta = match['metadata']
                score = match['score']
                title = str(meta.get('title', ''))
                
                # Duplikáció szűrés (Hogy ne legyen 2x ugyanaz)
                if title in seen_titles: continue
                seen_titles.add(title)

                # Keresés a látható mezőkben (Cím, Szerző, Kategória)
                full_text_search = normalize_text(title) + " " + \
                                   normalize_text(str(meta.get('author', ''))) + " " + \
                                   normalize_text(str(meta.get('category', '')))
                
                # SZABÁLYRENDSZER:
                keep_it = False
                
                # A. Ha nincs kulcsszó (általános keresés), akkor a Score dönt
                if not search_keywords:
                    if score > 0.3: keep_it = True
                
                else:
                    # B. Ha VAN kulcsszó (pl. név)
                    
                    # 1. Szöveges egyezés (Legalább egy kulcsszó benne van a címben/szerzőben)
                    # Szigorítottuk: Ha több szó van, többnek kell egyeznie
                    match_count = 0
                    for kw in search_keywords:
                        if kw in full_text_search:
                            match_count += 1
                    
                    if match_count >= 1: # Ha legalább egy erős szó megvan (pl. Berente)
                        keep_it = True
                    
                    # 2. BIZALMI ELV (High Score Override)
                    # Ha a szövegben NINCS benne (pl. rejtett szerző), de az AI nagyon biztos benne
                    # (Score > 0.72 - ez nagyon magas egyezés), akkor elhisszük neki.
                    elif score > 0.72:
                        # print(f"Trusting Vector for: {title} (Score: {score})") 
                        keep_it = True

                if keep_it:
                    final_results.append(match)

            # Rendezés pontszám szerint
            final_results.sort(key=lambda x: x['score'], reverse=True)
            
            return {"matches": final_results[:25]}

        except Exception as e:
            print(f"Keresési hiba: {e}")
            return {"matches": []}

    def process_message(self, user_input):
        detected_lang, scope, intent, keywords = self.generate_search_params(user_input)
        context_text = ""
        found_products = [] 
        
        footer_hu = "\n\n💡 *Tipp: Jelenleg a nyelvednek megfelelő könyveket keresem. Ha mindent látni szeretnél, írd hozzá: „minden nyelven”!*"
        footer_ro = "\n\n💡 *Sfat: Caut cărți în limba ta. Dacă vrei să vezi toate limbile, adaugă: „toate limbile”!*"

        if intent == "SEARCH":
            results = self.search_books(keywords, detected_lang, scope)
            
            if not results.get('matches'):
                msg = "Nu am găsit rezultate în stoc." if detected_lang == 'ro' else "Sajnos nem találtam készleten lévő könyvet."
                return {"reply": msg + (footer_ro if detected_lang == 'ro' else footer_hu), "products": []}

            for match in results['matches']:
                meta = match['metadata']
                title = str(meta.get('title', 'N/A'))
                
                product_data = {
                    "title": title,
                    "price": meta.get('price', 'N/A'), 
                    "url": meta.get('url', '#'),
                    "image": meta.get('image_url', '') 
                }
                found_products.append(product_data)
                
                author = meta.get('author', 'N/A')
                cat_tag = meta.get('category', 'N/A')
                context_text += f"- {title} (Szerző: {author}, Ár: {meta.get('price')} RON, Kategória: {cat_tag})\n"
                
                if len(found_products) >= 8: break 
            
            if not found_products:
                msg = "Nu am găsit nimic relevant." if detected_lang == 'ro' else "Sajnos nem találtam releváns könyvet."
                return {"reply": msg + (footer_ro if detected_lang == 'ro' else footer_hu), "products": []}

        else:
            context_text = "HASZNÁLD A TUDÁSBÁZIST!"

        lang_instruction = "Reply in ROMANIAN only." if detected_lang == 'ro' else "Reply in HUNGARIAN only."
        response = self.client_ai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Te Booksy vagy. {self.store_policy}"},
                {"role": "system", "content": lang_instruction},
                {"role": "user", "content": f"User: {user_input}\nFound Instock Books:\n{context_text}"}
            ],
            temperature=0.3
        )
        
        final_reply = response.choices[0].message.content
        if scope != 'ALL': final_reply += footer_ro if detected_lang == 'ro' else footer_hu
        return {"reply": final_reply, "products": found_products}

bot = BooksyBrain()

@app.get("/")
def home(): return {"status": "Booksy V17 (Instock + Smart Trust Logic)"}

@app.post("/hook")
def hook_endpoint(request: HookRequest):
    return {"hook": bot.generate_sales_hook(request)}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    return bot.process_message(request.message)