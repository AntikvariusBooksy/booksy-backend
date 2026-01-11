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

# Adatmodellek
class ChatRequest(BaseModel):
    message: str

class HookRequest(BaseModel):
    url: str
    page_title: str
    visitor_type: str # 'new' vagy 'returning'
    cart_status: str  # 'empty', 'active', 'just_added'
    lang: str

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

    # --- 1. PROAKTÍV HOOK GENERÁTOR (A buborék szövege) ---
    def generate_sales_hook(self, ctx: HookRequest):
        system_prompt = f"""
        You are Booksy, an AI Sales Agent for an antique bookstore.
        Your goal: Generate a SHORT (max 6-8 words), catchy, friendly 'hook' message for the chat bubble.
        
        Context:
        - User Language: {ctx.lang} (Reply in this language!)
        - Visitor Type: {ctx.visitor_type} (If 'returning', be warm like "Welcome back!")
        - Page: {ctx.page_title}
        - Cart Action: {ctx.cart_status}
        
        Rules:
        1. If 'cart_status' is 'just_added': Congratulate or suggest matching book.
        2. If 'page_title' suggests a specific book/category: Refer to it subtly.
        3. Keep it SUPER SHORT.
        """
        try:
            response = self.client_ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate hook."}
                ],
                temperature=0.7,
                max_tokens=30
            )
            return response.choices[0].message.content.strip()
        except:
            return "Bună! Te pot ajuta?" if ctx.lang == 'ro' else "Szia! Segíthetek?"

    # --- 2. QUERY EXPANSION (A Kereső Agya - V11) ---
    def generate_search_params(self, user_input):
        """
        Itt történik a varázslat: A 'Krimi' szót kibővítjük 'Bűnügyi, Thriller' szavakra,
        hogy megtalálja a kategóriákat is.
        """
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
                     4. EXPAND KEYWORDS: If the user asks for a genre/topic (e.g., 'krimi', 'scifi'), 
                        list related category names used in bookstores.
                        Example: User 'krimi' -> Keywords: 'krimi bűnügyi detektív thriller mystery'
                        Example: User 'gyerekkönyv' -> Keywords: 'gyerekkönyv mese ifjúsági képeskönyv'
                     
                     Output Format: LANG | SCOPE | INTENT | EXPANDED_KEYWORDS
                     """},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.3
            )
            parts = response.choices[0].message.content.split('|')
            return parts[0].strip().lower(), parts[1].strip(), parts[2].strip(), parts[3].strip()
        except:
            return "hu", "SPECIFIC", "SEARCH", user_input

    def search_books(self, query_text, lang_filter, scope):
        try:
            # Az OpenAI "Expanded Keywords"-jét vektorizáljuk
            response = self.client_ai.embeddings.create(input=query_text, model="text-embedding-3-small")
            
            filter_criteria = {"stock": "instock"}
            if scope != 'ALL' and lang_filter in ['hu', 'ro']:
                filter_criteria["lang"] = lang_filter
            
            # Súlyozott keresés (Mivel az adatbázisban a Kategória van elöl, ez nagyon pontos lesz)
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
                if match['score'] < 0.35: continue # Kicsit engedékenyebb score, mert a kategória egyezés erős
                meta = match['metadata']
                title = str(meta.get('title', 'N/A'))
                
                # Duplikáció szűrés
                is_dup = False
                for seen in seen_titles:
                    if difflib.SequenceMatcher(None, title.lower(), seen.lower()).ratio() > 0.85:
                        is_dup = True; break
                if is_dup: continue
                seen_titles.append(title)
                
                # Termék adat (ÁR az adatbázisból jön, ami már kezeli az akciót)
                product_data = {
                    "title": title,
                    "price": meta.get('price', 'N/A'), 
                    "url": meta.get('url', '#'),
                    "image": meta.get('image_url', '') 
                }
                found_products.append(product_data)
                
                # Context AI-nak (Category-t is beleírjuk!)
                cat_tag = meta.get('category', '')
                context_text += f"- {title} (Ár: {meta.get('price')} RON, Kategória: {cat_tag})\n"
                
                if len(found_products) >= 6: break
            
            if not found_products:
                msg = "Nu am găsit nimic relevant." if detected_lang == 'ro' else "Sajnos nem találtam releváns könyvet."
                tip = footer_ro if detected_lang == 'ro' else footer_hu
                return {"reply": msg + tip, "products": []}

        else:
            context_text = "HASZNÁLD A TUDÁSBÁZIST!"

        # Válasz generálás
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
def home(): return {"status": "Booksy V11 (Smart Search + Categories + Hooks)"}

@app.post("/hook")
def hook_endpoint(request: HookRequest):
    return {"hook": bot.generate_sales_hook(request)}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    return bot.process_message(request.message)