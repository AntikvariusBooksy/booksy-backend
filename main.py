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
            # 1. Pinecone keresés (Nagy merítés)
            response = self.client_ai.embeddings.create(input=query_text, model="text-embedding-3-small")
            
            # --- SZŰRŐ VISSZATÉVE (CSAK RAKTÁRON LÉVŐK) ---
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

            # --- 2. V15 OKOS SZŰRÉS (Smart Filter) ---
            # Nem dobjuk el, ha nem "Tökéletes", hanem megtartjuk, ha "Erős a gyanú" (magas Score)
            
            stop_words = ['konyv', 'konyvek', 'konyvet', 'carte', 'carti', 'keresek', 'kiado', 'szerzo', 'cim']
            normalized_query = normalize_text(query_text)
            search_keywords = [w for w in normalized_query.split() if w not in stop_words and len(w) > 2]

            final_results = []
            seen_ids = set()

            for match in matches:
                meta = match['metadata']
                score = match['score']
                
                # Cím + Szerző + Kategória szöveges vizsgálata
                full_text_search = normalize_text(str(meta.get('title', ''))) + " " + \
                                   normalize_text(str(meta.get('author', ''))) + " " + \
                                   normalize_text(str(meta.get('category', '')))
                
                # A) TÖKÉLETES TALÁLAT (Szöveges egyezés)
                # Pl: "Berente" benne van a címben/szerzőben
                is_text_match = False
                if search_keywords:
                    match_count = 0
                    for kw in search_keywords:
                        if kw in full_text_search:
                            match_count += 1
                    # Ha a kulcsszavak fele benne van (vagy legalább 1)
                    if match_count >= 1: 
                        is_text_match = True

                # B) SZEMANTIKUS TALÁLAT (Vector Score)
                # Ha a szövegben nincs benne (pl. "A tenyérelemzés..." cím), de a leírásban igen,
                # akkor a Vector Score magas lesz. Ezt is megtartjuk!
                is_high_score = score > 0.55  # Ha elég erős a gyanú

                # DÖNTÉS: Megtartjuk, ha szöveges VAGY erős szemantikus találat
                if is_text_match or is_high_score:
                    if match['id'] not in seen_ids:
                        final_results.append(match)
                        seen_ids.add(match['id'])

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
            seen_titles = []
            
            if not results.get('matches'):
                msg = "Nu am găsit rezultate (în stoc)." if detected_lang == 'ro' else "Sajnos nem találtam készleten lévő könyvet."
                return {"reply": msg + (footer_ro if detected_lang == 'ro' else footer_hu), "products": []}

            for match in results['matches']:
                meta = match['metadata']
                title = str(meta.get('title', 'N/A'))
                
                is_dup = False
                for seen in seen_titles:
                    if difflib.SequenceMatcher(None, title.lower(), seen.lower()).ratio() > 0.85:
                        is_dup = True; break
                if is_dup: continue
                seen_titles.append(title)
                
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
        system_prompt = f"Te Booksy vagy. {self.store_policy} Csak a felsorolt könyvekről beszélj, amik raktáron vannak."

        response = self.client_ai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
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
def home(): return {"status": "Booksy V15 (Instock ONLY + Smart Hybrid Search)"}

@app.post("/hook")
def hook_endpoint(request: HookRequest):
    return {"hook": bot.generate_sales_hook(request)}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    return bot.process_message(request.message)