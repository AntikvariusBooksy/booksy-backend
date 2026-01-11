import os
import difflib
import unicodedata
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

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
        
        # ITT A TUDÁSBÁZIS A KÉRDÉSEKHEZ
        self.store_policy = """
        [SZEREPEK: Te Booksy vagy, az Antikvarius.ro segítőkész AI asszisztense.]
        [SZÁLLÍTÁS ROMÁNIA: 24-48 óra, GLS futár. Ár: 25 RON (utánvét), 20 RON (kártya). Ingyenes 250 RON felett.]
        [SZÁLLÍTÁS MAGYARORSZÁG: 2-4 munkanap, GLS futár. Ár: 2990 HUF. Ingyenes 25.000 HUF felett.]
        [NÉMETORSZÁG/EU: Szállítás megoldható, egyedi díjszabás alapján. Kérlek vedd fel a kapcsolatot az info@antikvarius.ro címen.]
        [FIZETÉS: Utánvét (csak RO/HU), Bankkártya (Stripe/Barion).]
        [KAPCSOLAT: info@antikvarius.ro, Tel: +40 755 583 310]
        """

    def generate_sales_hook(self, ctx: HookRequest):
        try:
            response = self.client_ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "Short friendly sales hook."}, {"role": "user", "content": "Hook."}],
                temperature=0.7, max_tokens=30
            )
            return response.choices[0].message.content.strip()
        except: return "Szia! Segíthetek?"

    # --- 1. AZ "OKOS PORTÁS" (INTENT DETECTION) ---
    def detect_intent_and_language(self, user_input):
        try:
            # Ez a lépés dönti el, hogy könyvet keresünk VAGY beszélgetünk
            response = self.client_ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """
                     Classify the user input.
                     
                     1. LANGUAGE: 'hu' (Hungarian) or 'ro' (Romanian).
                     2. INTENT: 
                        - 'SEARCH': User is looking for a specific book, author, category, or topic (e.g., "Berente Ági", "krimi", "könyvek").
                        - 'INFO': User is asking about shipping, payment, contact, "hello", "help", or general questions (e.g., "mennyibe kerül a szállítás?", "buna ziua").
                     
                     OUTPUT FORMAT: LANG | INTENT
                     Examples:
                     "Berente Ági könyvek" -> hu | SEARCH
                     "Cat costa transportul?" -> ro | INFO
                     "Szia" -> hu | INFO
                     "Harry Potter" -> hu | SEARCH
                     """},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.1
            )
            parts = response.choices[0].message.content.split('|')
            lang = parts[0].strip().lower()
            intent = parts[1].strip().upper()
            return lang if lang in ['hu', 'ro'] else 'hu', intent
        except:
            return 'hu', 'SEARCH' # Fallback

    # --- 2. KERESŐMOTOR (Csak akkor fut le, ha SEARCH az intent) ---
    def search_engine_logic(self, query_text, lang_filter):
        try:
            stop_words = ['a', 'az', 'egy', 'es', 'konyv', 'konyvek', 'keresek', 'kiado', 'szerzo', 'cim', 'regeny', 'iro', 'vennek', 'carte', 'carti', 'caut']
            normalized_query = normalize_text(query_text)
            keywords = [w for w in normalized_query.split() if w not in stop_words and len(w) > 2]
            
            clean_query = " ".join(keywords) if keywords else query_text

            response = self.client_ai.embeddings.create(input=clean_query, model="text-embedding-3-small")
            
            filter_criteria = {"stock": "instock"}
            if lang_filter: filter_criteria["lang"] = lang_filter
            
            raw_results = self.index.query(
                vector=response.data[0].embedding,
                top_k=200, 
                include_metadata=True, 
                filter=filter_criteria
            )

            matches = raw_results.get('matches', [])
            if not matches: return []

            scored_results = []
            seen_titles = set()

            for match in matches:
                meta = match['metadata']
                title = str(meta.get('title', ''))
                if title in seen_titles: continue
                seen_titles.add(title)

                title_norm = normalize_text(title)
                author_norm = normalize_text(str(meta.get('author', '')))
                cat_norm = normalize_text(str(meta.get('category', '')))
                full_text_norm = normalize_text(str(meta.get('full_search_text', '')))

                relevance_score = 0
                match_count = 0

                if not keywords:
                    relevance_score = match['score'] * 100
                else:
                    for kw in keywords:
                        kw_score = 0
                        found = False
                        
                        if kw in title_norm:
                            kw_score += 100
                            found = True
                        elif kw in author_norm:
                            kw_score += 80
                            found = True
                        elif kw in full_text_norm: # Leírásban keresés
                            kw_score += 20
                            found = True
                        
                        if found:
                            match_count += 1
                            relevance_score += kw_score

                    if match_count == len(keywords) and len(keywords) > 1:
                        relevance_score += 200 

                if keywords and relevance_score < 10:
                    continue

                match['final_relevance'] = relevance_score
                scored_results.append(match)

            scored_results.sort(key=lambda x: x['final_relevance'], reverse=True)
            return scored_results[:20]

        except Exception as e:
            print(f"Hiba: {e}")
            return []

    def process_message(self, user_input):
        # 1. LÉPÉS: Mit akar a user?
        lang, intent = self.detect_intent_and_language(user_input)
        
        # 2. LÉPÉS: ÚTVÁLASZTÁS
        
        # --- A) HA NEM KÖNYVET KERES (PL. SZÁLLÍTÁS, ÜDVÖZLÉS) ---
        if intent == 'INFO':
            lang_instruction = "Reply in ROMANIAN." if lang == 'ro' else "Reply in HUNGARIAN."
            system_prompt = f"Te Booksy vagy. {self.store_policy} Válaszolj a felhasználó kérdésére kedvesen és röviden."
            
            response = self.client_ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "system", "content": lang_instruction},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.3
            )
            return {"reply": response.choices[0].message.content, "products": []}

        # --- B) HA KÖNYVET KERES (SEARCH) ---
        matches = self.search_engine_logic(user_input, lang)
        
        context_text = ""
        found_products = []
        
        footer_hu = "\n\n💡 *Tipp: Ha mindent látni szeretnél, írd hozzá: „minden nyelven”!*"
        footer_ro = "\n\n💡 *Sfat: Adaugă: „toate limbile”!*"

        if not matches:
            msg = "Sajnos nem találtam készleten lévő könyvet." if lang == 'hu' else "Nu am găsit cărți în stoc."
            return {"reply": msg + (footer_ro if lang == 'ro' else footer_hu), "products": []}

        for match in matches:
            meta = match['metadata']
            title = str(meta.get('title', 'N/A'))
            
            product_data = {
                "title": title,
                "price": meta.get('price', 'N/A'), 
                "url": meta.get('url', '#'),
                "image": meta.get('image_url', '') 
            }
            found_products.append(product_data)
            
            author = meta.get('author', '')
            short_desc = str(meta.get('short_desc', ''))[:100]
            
            context_text += f"- {title} (Szerző: {author}, Ár: {meta.get('price')} RON, Info: {short_desc})\n"
            
            if len(found_products) >= 8: break 

        lang_instruction = "Reply in ROMANIAN." if lang == 'ro' else "Reply in HUNGARIAN."
        system_prompt = f"Te Booksy vagy. {self.store_policy} Ajánld a megtalált könyveket."

        response = self.client_ai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": lang_instruction},
                {"role": "user", "content": f"User: {user_input}\nFound Books:\n{context_text}"}
            ],
            temperature=0.3
        )
        
        final_reply = response.choices[0].message.content
        if lang == 'hu': final_reply += footer_hu
        else: final_reply += footer_ro
        
        return {"reply": final_reply, "products": found_products}

bot = BooksyBrain()

@app.get("/")
def home(): return {"status": "Booksy V21 (Intent Router + Full Search)"}

@app.post("/hook")
def hook_endpoint(request: HookRequest):
    return {"hook": bot.generate_sales_hook(request)}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    return bot.process_message(request.message)