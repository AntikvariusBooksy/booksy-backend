import os
import difflib
import unicodedata
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

# Adatmodellek
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
    """Ékezetek eltávolítása és kisbetűsítés a pontosabb szöveges kereséshez"""
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

    # --- 1. PROAKTÍV HOOK GENERÁTOR ---
    def generate_sales_hook(self, ctx: HookRequest):
        system_prompt = f"""
        You are Booksy, an AI Sales Agent for an antique bookstore.
        Your goal: Generate a SHORT (max 6-8 words), catchy, friendly 'hook' message for the chat bubble.
        Context: Language: {ctx.lang}, Visitor: {ctx.visitor_type}, Page: {ctx.page_title}, Cart: {ctx.cart_status}.
        Rules: Keep it SUPER SHORT.
        """
        try:
            response = self.client_ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": "Generate hook."}],
                temperature=0.7, max_tokens=30
            )
            return response.choices[0].message.content.strip()
        except:
            return "Bună! Te pot ajuta?" if ctx.lang == 'ro' else "Szia! Segíthetek?"

    # --- 2. KERESŐ LOGIKA (V12 - HIBRID) ---
    def generate_search_params(self, user_input):
        try:
            response = self.client_ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """
                     Analyze the user's search query for an online bookstore.
                     Tasks:
                     1. Detect Language (hu/ro).
                     2. Detect Intent (SEARCH/INFO).
                     3. Detect Scope (ALL/SPECIFIC).
                     4. KEYWORDS: Keep original keywords like author names intact!
                     Output Format: LANG | SCOPE | INTENT | KEYWORDS
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
            # 1. Vektoros keresés (széles merítés: 100 db)
            response = self.client_ai.embeddings.create(input=query_text, model="text-embedding-3-small")
            
            filter_criteria = {"stock": "instock"}
            if scope != 'ALL' and lang_filter in ['hu', 'ro']:
                filter_criteria["lang"] = lang_filter
            
            raw_results = self.index.query(
                vector=response.data[0].embedding,
                top_k=100, # SOKAT kérünk le, hogy tudjunk válogatni
                include_metadata=True, 
                filter=filter_criteria
            )

            matches = raw_results.get('matches', [])
            if not matches: return {"matches": []}

            # --- 2. PYTHON OLDALI SZIGORÚ SZŰRÉS (RERANKING) ---
            # Megnézzük, hogy a keresett szavak (pl. "berente") szövegesen benne vannak-e a találatban.
            
            search_terms = normalize_text(query_text).split()
            # Kivesszük a töltelékszavakat
            stop_words = ['könyv', 'könyvek', 'carte', 'carti', 'keresek', 'kiado', 'kiadó']
            keywords = [w for w in search_terms if w not in stop_words and len(w) > 2]

            strict_matches = []
            fuzzy_matches = []

            for match in matches:
                meta = match['metadata']
                # Adatok normalizálása a kereséshez
                full_text_search = normalize_text(str(meta.get('title', ''))) + " " + \
                                   normalize_text(str(meta.get('author', ''))) + " " + \
                                   normalize_text(str(meta.get('category', '')))
                
                # Ha a felhasználó konkrét nevet írt (pl. Berente), és az benne van:
                # Akkor ez egy "Szigorú Találat"
                is_strict = False
                if keywords:
                    # Minden kulcsszónak (pl "berente", "agi") benne kell lennie? 
                    # Vagy legalább az egyiknek? Most legyen "legalább az egyik erős szó"
                    for kw in keywords:
                        if kw in full_text_search:
                            is_strict = True
                            break
                
                if is_strict:
                    strict_matches.append(match)
                else:
                    fuzzy_matches.append(match)

            # --- 3. DÖNTÉS ---
            # Ha van elég (min 1) szigorú találat, akkor CSAK azokat mutatjuk.
            # Így dobjuk ki a Berzsenyit, ha Berentét kerestünk.
            if len(strict_matches) > 0:
                print(f"Strict Filter: Found {len(strict_matches)} exact matches for '{query_text}'")
                return {"matches": strict_matches[:25]} # Visszaadjuk a pontosakat
            
            # Ha nincs pontos találat (pl. témára keresett: "valami izgalmas"), marad a fuzzy
            return {"matches": fuzzy_matches[:25]}

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
                msg = "Nu am găsit rezultate." if detected_lang == 'ro' else "Sajnos nem találtam ilyen könyvet."
                tip = footer_ro if detected_lang == 'ro' else footer_hu
                return {"reply": msg + tip, "products": []}

            for match in results['matches']:
                # Ha Strict Match volt, akkor a score nem számít annyira, de azért ne legyen nulla
                if match['score'] < 0.25: continue 
                
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
                
                # Context AI-nak
                author = meta.get('author', '')
                cat_tag = meta.get('category', '')
                context_text += f"- {title} (Szerző: {author}, Ár: {meta.get('price')} RON, Kategória: {cat_tag})\n"
                
                if len(found_products) >= 6: break
            
            if not found_products:
                msg = "Nu am găsit nimic relevant." if detected_lang == 'ro' else "Sajnos nem találtam releváns könyvet."
                tip = footer_ro if detected_lang == 'ro' else footer_hu
                return {"reply": msg + tip, "products": []}

        else:
            context_text = "HASZNÁLD A TUDÁSBÁZIST!"

        if detected_lang == 'ro': lang_instruction = "Reply in ROMANIAN only."
        else: lang_instruction = "Reply in HUNGARIAN only."

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
        if scope != 'ALL':
            final_reply += footer_ro if detected_lang == 'ro' else footer_hu
        
        return {"reply": final_reply, "products": found_products}

bot = BooksyBrain()

@app.get("/")
def home(): return {"status": "Booksy V12 (Hybrid Search - Strict Filter)"}

@app.post("/hook")
def hook_endpoint(request: HookRequest):
    return {"hook": bot.generate_sales_hook(request)}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    return bot.process_message(request.message)