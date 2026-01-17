# BOOKSY BRAIN - V91 (V82 LOGIC RESTORED + SAFE MATH PATCH)
# --- SQLITE FIX ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- IMPORTOK ---
import os
import time
import requests
import hashlib
import re
import json
import unicodedata
import html
import xml.etree.ElementTree as ET
import gc
import chromadb
from chromadb.config import Settings
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from typing import List, Optional, Dict, Any

# --- KONFIGURÁCIÓ ---
load_dotenv()
XML_FEED_URL = os.getenv("XML_FEED_URL", "https://www.antikvarius.ro/wp-content/uploads/woo-feed/google/xml/booksyfullfeed.xml")
TEMP_FILE = "temp_feed.xml"

# --- URL TUDÁSBÁZIS ---
POLICY_PAGES = [
    {"url": "https://www.antikvarius.ro/termeni-si-conditii-de-utilizare/", "lang": "ro", "name": "Termeni și condiții"},
    {"url": "https://www.antikvarius.ro/informatii-despre-plata/", "lang": "ro", "name": "Informații despre plată"},
    {"url": "https://www.antikvarius.ro/informatii-despre-livrare/", "lang": "ro", "name": "Informații despre livrare"},
    {"url": "https://www.antikvarius.ro/contact/", "lang": "ro", "name": "Contact"},
]

# --- ADATMODELLEK ---
class ChatRequest(BaseModel):
    message: str
    context_url: Optional[str] = "" 
    session_id: Optional[str] = ""

# --- HELPEREK ---
def normalize_text(text):
    if not text: return ""
    text = str(text).lower()
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def safe_str(val):
    if val is None: return ""
    return html.unescape(str(val).strip())

def clean_html_smart(raw_html):
    if not raw_html: return ""
    s = safe_str(raw_html)
    s = re.sub(r'<script.*?>.*?</script>', '', s, flags=re.DOTALL|re.IGNORECASE)
    s = re.sub(r'<style.*?>.*?</style>', '', s, flags=re.DOTALL|re.IGNORECASE)
    s = re.sub(r'</(td|th)>', ' | ', s, flags=re.IGNORECASE) 
    s = re.sub(r'</(tr|p|div|h1|h2|h3|h4|h5|h6|li)>', '\n', s, flags=re.IGNORECASE) 
    s = re.sub(r'<br\s*/?>', '\n', s, flags=re.IGNORECASE)
    cleanr = re.compile('<.*?>')
    s = re.sub(cleanr, ' ', s)
    s = re.sub(r'\n\s*\n', '\n', s)
    s = re.sub(r'[ \t]+', ' ', s)
    return s.strip()

def extract_all_data(elem) -> Dict[str, Any]:
    data = {}
    for child in elem:
        tag = child.tag.split('}')[-1].lower()
        val = safe_str(child.text)
        if val: data[tag] = val
    return data

def generate_content_hash(data_string):
    return hashlib.md5(data_string.encode('utf-8')).hexdigest()

def clean_price_raw(raw_price):
    if not raw_price: return "0 RON"
    s = str(raw_price).strip()
    cleaned_num = re.sub(r"[^\d.,]", "", s)
    if not cleaned_num: return s 
    return f"{cleaned_num} RON"

# V91 ÚJÍTÁS (A V86-ból áthozva): Biztonságos Ár konverzió
def parse_price_to_float(price_input):
    try:
        if price_input is None: return None
        # Bármi jön (int, float, string), stringgé alakítjuk, tisztítjuk, majd float
        s = str(price_input).lower()
        s = s.replace("ron", "").replace("lei", "").replace(" ", "").strip()
        s = s.replace(",", ".") 
        if not s: return None
        return float(s)
    except:
        return None

# V91: NINCS 'detect_hungarian_intent'. AZ AI DÖNT! (Ahogy V82-ben)

# --- ADATBÁZIS ---
class DBHandler:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./booksy_db")
        self.collection = self.client.get_or_create_collection(name="booksy_collection")

# --- AUTO UPDATER (V82 Logic + Regex Fixes) ---
class AutoUpdater:
    def __init__(self, db: DBHandler):
        self.api_key_openai = os.getenv("OPENAI_API_KEY")
        self.client_ai = OpenAI(api_key=self.api_key_openai)
        self.db = db

    def download_feed(self):
        headers = {'User-Agent': 'BooksyBot/1.0'}
        for attempt in range(3):
            try:
                print(f"⬇️ [DOWNLOAD] XML Feed Letöltés (Kísérlet {attempt+1}/3)...")
                with requests.get(XML_FEED_URL, headers=headers, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with open(TEMP_FILE, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                file_size = os.path.getsize(TEMP_FILE)
                if file_size < 10000: raise Exception("Túl kicsi fájl.")
                print(f"✅ [DOWNLOAD] Siker! Méret: {file_size / 1024 / 1024:.2f} MB")
                return True
            except Exception as e:
                print(f"⚠️ Hiba: {e}")
                time.sleep(5)
        return False

    def update_policies(self, current_ts):
        print("ℹ️ [POLICY] Információs oldalak intelligens beolvasása...")
        headers = {'User-Agent': 'BooksyBot/1.0'}
        for page in POLICY_PAGES:
            try:
                url = page['url']
                r = requests.get(url, headers=headers, timeout=30)
                if r.status_code == 200:
                    raw_html = r.text
                    clean_text = clean_html_smart(raw_html)
                    if len(clean_text) > 25000: clean_text = clean_text[:25000]
                    d_hash = generate_content_hash(clean_text)
                    page_id = f"policy_{generate_content_hash(url)}"
                    try:
                        existing = self.db.collection.get(ids=[page_id], include=['metadatas'])
                        if existing['ids'] and existing['metadatas'][0].get('content_hash') == d_hash:
                            print(f"   ⏩ [SKIP] Policy változatlan: {page['name']}")
                            continue
                    except: pass
                    
                    emb_text = f"Típus: Szabályzat (ro). Cím: {page['name']}. Tartalom: {clean_text[:8000]}"
                    emb = self.client_ai.embeddings.create(input=emb_text, model="text-embedding-3-small").data[0].embedding
                    
                    meta = {
                        "title": page['name'], "url": url, "text": clean_text,
                        "lang": "ro", "type": "policy", "content_hash": d_hash, "last_seen": current_ts
                    }
                    self.db.collection.upsert(ids=[page_id], embeddings=[emb], metadatas=[meta])
                    print(f"   ✅ [POLICY] Frissítve (Strukturált): {page['name']}")
                else: print(f"   ⚠️ Hiba: {r.status_code} - {url}")
            except Exception as e: print(f"   ❌ Hiba: {e}")

    def run_daily_update(self):
        print(f"🔄 [AUTO] Napi Frissítés (V91 - V82 RESTORED)")
        current_sync_ts = int(time.time())
        self.update_policies(current_sync_ts)
        if not self.download_feed(): return
        try:
            print("🚀 [MODE] Parsing Books & Merging")
            context = ET.iterparse(TEMP_FILE, events=("end",))
            unique_books_buffer = {} 
            count_total_xml_items = 0
            for event, elem in context:
                tag_local = elem.tag.split('}')[-1].lower()
                if tag_local in ['item', 'post']:
                    count_total_xml_items += 1
                    try:
                        item_data = extract_all_data(elem)
                        bid = item_data.get('id') or item_data.get('post_id') or item_data.get('g:id')
                        if bid:
                            title = item_data.get('title') or "Nincs cím"
                            desc = item_data.get('description', '')
                            short_desc = item_data.get('shortdescription', '') or item_data.get('excerpt', '')
                            full_raw_text = f"{desc} {short_desc}"
                            clean_desc = clean_html_smart(full_raw_text)
                            category = item_data.get('product_type') or item_data.get('category') or ""
                            category = clean_html_smart(category)
                            
                            # V82 LOGIKA: Regex a Kiadóhoz + Bookman Override
                            pub = "Ismeretlen"
                            match_pub = re.search(r'(?:Kiadó|Kiadás|Publisher)\s*(?:[:|])\s*([^|\n\r]+)', full_raw_text, re.IGNORECASE)
                            if match_pub: pub = match_pub.group(1).strip()
                            if "bookman" in normalize_text(category): pub = "Bookman Kiadó"

                            match_auth = re.search(r'(?:Szerző|Írta|Author|Szerzők)\s*(?:[:|])\s*([^|\n\r]+)', full_raw_text, re.IGNORECASE)
                            auth = match_auth.group(1).strip() if match_auth else "Ismeretlen"

                            raw_price = item_data.get('sale_price') or item_data.get('price') or "0"
                            final_ron_price = clean_price_raw(raw_price)

                            cat_norm = normalize_text(category)
                            detected_lang = "hu"
                            if "carti in limba romana" in cat_norm: detected_lang = "ro"
                            elif "magyar nyelvu konyvek" in cat_norm: detected_lang = "hu"

                            if bid in unique_books_buffer:
                                existing_entry = unique_books_buffer[bid]
                                existing_cat = existing_entry['category']
                                if category and category not in existing_cat:
                                    merged_cat = f"{existing_cat} | {category}"
                                    unique_books_buffer[bid]['category'] = merged_cat
                                    unique_books_buffer[bid].update({"price": final_ron_price, "title": title})
                            else:
                                book_obj = {
                                    "id": bid, "title": title, "url": item_data.get('link', ''), "image_url": item_data.get('image_link', ''),
                                    "price": final_ron_price, "publisher": pub, "author": auth, "category": category,
                                    "description": clean_desc, "stock": "instock", "lang": detected_lang, "type": "book", "last_seen": current_sync_ts
                                }
                                for k, v in item_data.items():
                                    if k not in book_obj:
                                        clean_v = clean_html_smart(str(v))
                                        if len(clean_v) > 500: clean_v = clean_v[:500]
                                        book_obj[k] = clean_v
                                unique_books_buffer[bid] = book_obj
                    except Exception as e: pass
                    elem.clear()
                    if count_total_xml_items % 5000 == 0: gc.collect()
            
            print(f"✅ [MERGE] Kész! Egyedi könyvek: {len(unique_books_buffer)}")
            print("🚀 [SMART UPLOAD] Hash ellenőrzés...")
            ids_batch, embeddings_batch, metadatas_batch = [], [], []
            count_processed, count_skipped, count_uploaded = 0, 0, 0
            
            for bid, book_data in unique_books_buffer.items():
                count_processed += 1
                hash_input = f"{book_data['title']}|{book_data['price']}|{book_data['category']}|{book_data['publisher']}"
                d_hash = generate_content_hash(hash_input)
                book_data['content_hash'] = d_hash
                try:
                    existing = self.db.collection.get(ids=[bid], include=['metadatas'])
                    if existing and existing['ids'] and len(existing['ids']) > 0:
                        stored_hash = existing['metadatas'][0].get('content_hash', '')
                        if stored_hash == d_hash:
                            count_skipped += 1
                            if count_processed % 1000 == 0: print(f"⏩ [SKIP] {count_skipped} változatlan könyv...")
                            continue 
                except: pass
                emb_text = f"Nyelv: {book_data['lang']}. Cím: {book_data['title']}. Szerző: {book_data['author']}. Kategória: {book_data['category']}. Kiadó: {book_data['publisher']}. Leírás: {book_data['description'][:800]}"
                try:
                    emb = self.client_ai.embeddings.create(input=emb_text[:8000], model="text-embedding-3-small").data[0].embedding
                    clean_meta = book_data.copy()
                    del clean_meta['description'] 
                    clean_meta['text_preview'] = book_data['description'][:100]
                    ids_batch.append(bid)
                    embeddings_batch.append(emb)
                    metadatas_batch.append(clean_meta)
                    count_uploaded += 1
                    if len(ids_batch) >= 50:
                        self.db.collection.upsert(ids=ids_batch, embeddings=embeddings_batch, metadatas=metadatas_batch)
                        ids_batch, embeddings_batch, metadatas_batch = [], [], []
                        print(f"💾 [UPDATE] {count_uploaded} könyv frissítve...")
                except Exception as e: print(f"⚠️ Hiba ({bid}): {e}")
            if ids_batch: self.db.collection.upsert(ids=ids_batch, embeddings=embeddings_batch, metadatas=metadatas_batch)
            if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)
            print(f"🏁 [VÉGE] {count_processed} feldolgozva. ⏩ {count_skipped} változatlan. 💾 {count_uploaded} frissítve.")
        except Exception as e: print(f"❌ Hiba: {e}")

# --- BRAIN (V91 - V82 LOGIC + SAFE MATH) ---
class BooksyBrain:
    def __init__(self):
        self.db = DBHandler()
        self.updater = AutoUpdater(self.db)
        self.client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.user_session_cache = {}

    # V91: Tool definíció (V82 kompatibilis + ár paraméterek)
    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_database",
                    "description": "Keresés a könyvek vagy a szabályzatok között. Használd ezt, ha a felhasználó könyvet keres, árat kérdez, szállításról érdeklődik.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "A keresett kulcsszavak tisztítva."
                            },
                            "filter_lang": {
                                "type": "string",
                                "enum": ["hu", "ro", "all"],
                                "description": "Szűrés nyelvre. 'hu' ha magyar, 'ro' ha román, 'all' ha mindent kér."
                            },
                            "search_type": {
                                "type": "string",
                                "enum": ["book", "policy"],
                                "description": "'policy' ha szállítás/fizetés/kapcsolat. 'book' ha könyv."
                            },
                            "min_price": { "type": "number", "description": "Minimális ár (ha van)." },
                            "max_price": { "type": "number", "description": "Maximális ár (ha van)." }
                        },
                        "required": ["query", "filter_lang", "search_type"]
                    }
                }
            }
        ]

    # V91: Kereső Motor (V82 logika + SAFE MATH PATCH)
    def execute_search(self, query, filter_lang, search_type, min_price=None, max_price=None):
        try:
            q_norm = normalize_text(query)
            
            # 1. POLICY
            if search_type == "policy":
                vec = self.client_ai.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
                res = self.db.collection.query(query_embeddings=[vec], n_results=3, where={"type": "policy"})
                return self.format_chroma_results(res)

            # 2. BOOK
            vec = self.client_ai.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
            where_clause = {"$and": [{"stock": "instock"}, {"type": "book"}]}
            if filter_lang != 'all' and "bookman" not in q_norm:
                where_clause = {"$and": [{"stock": "instock"}, {"type": "book"}, {"lang": filter_lang}]}
            
            matches_raw = self.db.collection.query(query_embeddings=[vec], n_results=100, where=where_clause)
            matches = self.format_chroma_results(matches_raw)
            
            results = []
            seen_items = set()

            # V91 PATCH: Biztonságos Type Casting
            safe_min = parse_price_to_float(min_price)
            safe_max = parse_price_to_float(max_price)

            for m in matches:
                meta = m['metadata']
                raw_db_price = meta.get('price')
                final_price_str = clean_price_raw(raw_db_price)
                
                # V91 PATCH: Matematikai szűrés (Hibatűréssel)
                price_val = parse_price_to_float(final_price_str)
                if price_val is not None:
                    if safe_min is not None and price_val < safe_min: continue
                    if safe_max is not None and price_val > safe_max: continue
                
                # Dedup
                unique_key = f"{meta.get('title')}|{final_price_str}"
                if unique_key in seen_items: continue
                seen_items.add(unique_key)
                
                base_score = (2.0 - m['score']) * 100 
                title_norm = normalize_text(meta.get('title', ''))
                auth_norm = normalize_text(meta.get('author', ''))
                pub_norm = normalize_text(meta.get('publisher', ''))
                
                score = base_score
                if q_norm in title_norm: score += 50
                if q_norm in auth_norm: score += 40
                if q_norm in pub_norm: score += 40
                if "bookman" in q_norm: 
                     if "bookman" in pub_norm or "bookman" in normalize_text(meta.get('category', '')): score += 500
                
                m['custom_score'] = score
                results.append(m)
            
            results.sort(key=lambda x: x['custom_score'], reverse=True)
            return results[:10]
        except Exception as e:
            print(f"Search Error: {e}")
            return []

    def format_chroma_results(self, res):
        formatted = []
        if not res['ids']: return []
        for i in range(len(res['ids'][0])):
            formatted.append({
                "id": res['ids'][0][i],
                "score": res['distances'][0][i] if 'distances' in res else 0,
                "metadata": res['metadatas'][0][i]
            })
        return formatted

    # V91: DECOUPLED AGENT PROCESS (Router -> Python -> Writer)
    # Ez a V82 logikáját futtatja, de a technikai hibák (400) nélkül.
    def process(self, msg, context_url, session_id):
        try:
            last_search = self.user_session_cache.get(session_id, "")
            site_lang = 'ro'
            if context_url and '/hu/' in str(context_url).lower(): site_lang = 'hu'

            # --- 1. ROUTER (V82 stílusú Prompt) ---
            # Az AI-ra bízzuk a nyelvfelismerést és a szándékot!
            router_system_prompt = f"""
            You are the Brain of Booksy, an antique bookstore AI assistant.
            Current Site Language: {site_lang}
            Last User Search Topic: "{last_search}"
            
            YOUR TASKS:
            1. Detect the user's INTENT and LANGUAGE from the message.
            2. If searching for books/policy/price -> CALL 'search_database'.
            3. 'filter_lang': 
               - 'hu' if user speaks Hungarian.
               - 'ro' if user speaks Romanian.
               - 'all' ONLY if explicit.
            4. 'query': Extract keywords.
            5. 'min_price'/'max_price': Extract numbers if price range is mentioned.
            
            If user says "Hello" or chitchat, DO NOT call the tool, just reply normally.
            """

            router_messages = [
                {"role": "system", "content": router_system_prompt},
                {"role": "user", "content": msg}
            ]

            # Döntés
            router_response = self.client_ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=router_messages,
                tools=self.get_tools(),
                tool_choice="auto", 
                temperature=0.0
            )
            
            response_msg = router_response.choices[0].message
            tool_calls = response_msg.tool_calls
            
            final_products = []
            final_reply = ""
            used_lang_filter = site_lang

            # --- 2. VÉGREHAJTÁS ---
            if tool_calls:
                tool_call = tool_calls[0]
                if tool_call.function.name == "search_database":
                    args = json.loads(tool_call.function.arguments)
                    
                    if args.get('filter_lang') != 'all':
                        self.user_session_cache[session_id] = args.get('query')
                    
                    used_lang_filter = args.get('filter_lang', site_lang)
                    
                    search_results = self.execute_search(
                        query=args.get('query'),
                        filter_lang=args.get('filter_lang'),
                        search_type=args.get('search_type'),
                        min_price=args.get('min_price'),
                        max_price=args.get('max_price')
                    )
                    
                    ctx_text = ""
                    product_count = 0
                    if not search_results:
                        ctx_text = "No results found in database."
                    else:
                        product_count = len(search_results)
                        for m in search_results:
                            meta = m['metadata']
                            if args.get('search_type') == 'policy':
                                ctx_text += f"--- POLICY ({meta.get('lang')}) ---\n{meta.get('text')}\n"
                            else:
                                p_price = clean_price_raw(meta.get('price'))
                                details = f"Title: {meta.get('title')}, Price: {p_price}, Publisher: {meta.get('publisher')}, Cat: {meta.get('category')}"
                                ctx_text += f"--- BOOK ---\n{details}\n"
                                p = {"title": meta.get('title'), "price": p_price, "url": meta.get('url'), "image": meta.get('image_url')}
                                final_products.append(p)
                                if len(final_products) >= 8: break
                    
                    # --- 3. WRITER (V82 stílusú Prompt, Tiszta lappal) ---
                    writer_system_prompt = f"""
                    You are Booksy. Answer the user based ONLY on the DATA below.
                    
                    DATA:
                    {ctx_text}
                    
                    RULES:
                    1. Language: Answer in the SAME language as the User's original message ("{msg}").
                    2. NO HALLUCINATION: Only list books found in DATA. If 0 results, say so.
                    3. LIST FORMAT: Title and Price.
                    4. Be helpful and professional.
                    """
                    
                    writer_messages = [
                        {"role": "system", "content": writer_system_prompt},
                        {"role": "user", "content": "Write the response."}
                    ]
                    
                    final_res = self.client_ai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=writer_messages,
                        temperature=0.1
                    )
                    final_reply = final_res.choices[0].message.content

            else:
                final_reply = response_msg.content

            # Tipp (V82 Helper nélkül is működik, de a biztonság kedvéért itt hagyjuk a logikát)
            if final_products and used_lang_filter != 'all':
                # Egyszerű tipp generálás
                if used_lang_filter == 'hu':
                    final_reply += "\n\n💡 Tipp: Nem ezt kerested? Írd be: 'minden nyelven', hogy a teljes adatbázisban keressünk."
                elif used_lang_filter == 'ro':
                     final_reply += "\n\n💡 Sfat: Nu ai găsit? Scrie 'toate limbile' pentru a căuta în toată baza de date."

            return {"reply": final_reply, "products": final_products}

        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            return {"reply": "Hiba történt. Kérlek próbáld újra.", "products": []}

# --- APP ---
bot = BooksyBrain()
scheduler = BackgroundScheduler()
scheduler.add_job(bot.updater.run_daily_update, CronTrigger(hour=3, minute=0))

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.start()
    yield
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def home(): return {"status": "Booksy V91 (V82 RESTORED + SAFE MATH)"}

@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url, req.session_id)

@app.post("/force-update")
def force(bt: BackgroundTasks):
    bt.add_task(bot.updater.run_daily_update)
    return {"status": "V91 Force Update Running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)