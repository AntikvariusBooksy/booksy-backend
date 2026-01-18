# BOOKSY BRAIN - V95 (STRICT SKU SEPARATION + AGENTIC LOGIC)
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
from html.parser import HTMLParser # V94 Örökség: A profi parser
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

# V94/V95: SAJÁT HTML STRIPPER OSZTÁLY
class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text_parts = []
     
    def handle_data(self, d):
        self.text_parts.append(d)
     
    def handle_starttag(self, tag, attrs):
        if tag in ['br', 'p', 'div', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'tr']:
            self.text_parts.append('\n')
     
    def handle_endtag(self, tag):
        if tag in ['p', 'div', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'tr']:
            self.text_parts.append('\n')
            
    def get_data(self):
        return ''.join(self.text_parts)

def clean_html_smart(raw_html):
    if not raw_html: return ""
    try:
        s = safe_str(raw_html)
        stripper = MLStripper()
        stripper.feed(s)
        text = stripper.get_data()
        
        text = text.replace('\xa0', ' ')
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()
    except:
        return safe_str(raw_html)

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

def parse_price_to_float(price_input):
    try:
        if price_input is None: return None
        s = str(price_input).lower().replace("ron", "").replace("lei", "").replace(" ", "").strip()
        s = s.replace(",", ".") 
        if not s: return None
        return float(s)
    except: return None

# --- ADATBÁZIS ---
class DBHandler:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./booksy_db")
        self.collection = self.client.get_or_create_collection(name="booksy_collection")

# --- AUTO UPDATER (V95 - STRICT SKU ENGINE) ---
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
        print(f"🔄 [AUTO] Napi Frissítés (V95 - STRICT SKU MODE)")
        current_sync_ts = int(time.time())
        self.update_policies(current_sync_ts)
        if not self.download_feed(): return
        try:
            print("🚀 [MODE] Parsing Books (No Merging - Pure SKU)")
            context = ET.iterparse(TEMP_FILE, events=("end",))
            unique_books_buffer = {} 
            count_total_xml_items = 0
            
            for event, elem in context:
                tag_local = elem.tag.split('}')[-1].lower()
                if tag_local in ['item', 'post']:
                    count_total_xml_items += 1
                    try:
                        item_data = extract_all_data(elem)
                        # SKU/ID az elsődleges kulcs.
                        bid = item_data.get('id') or item_data.get('post_id') or item_data.get('g:id')
                        
                        if bid:
                            title = item_data.get('title') or "Nincs cím"
                            
                            # Adattisztítás
                            raw_desc = f"{item_data.get('description', '')} {item_data.get('shortdescription', '')}"
                            clean_desc = clean_html_smart(raw_desc) 
                            
                            category = clean_html_smart(item_data.get('product_type') or item_data.get('category') or "")
                            
                            # Kiadó/Szerző felismerés
                            pub = "Ismeretlen"
                            match_pub = re.search(r'(?:Kiadó|Kiadás|Publisher)\s*[:|]\s*(.*?)(?:\n|$)', clean_desc, re.IGNORECASE)
                            if match_pub: pub = match_pub.group(1).strip()
                            if "bookman" in normalize_text(category): pub = "Bookman Kiadó"

                            auth = "Ismeretlen"
                            match_auth = re.search(r'(?:Szerző|Írta|Author|Szerzők)\s*[:|]\s*(.*?)(?:\n|$)', clean_desc, re.IGNORECASE)
                            if match_auth: auth = match_auth.group(1).strip()
                            else: auth = item_data.get('author') or "Ismeretlen"

                            raw_price = item_data.get('sale_price') or item_data.get('price') or "0"
                            final_ron_price = clean_price_raw(raw_price)

                            cat_norm = normalize_text(category)
                            detected_lang = "hu"
                            if "carti in limba romana" in cat_norm: detected_lang = "ro"
                            elif "magyar nyelvu konyvek" in cat_norm: detected_lang = "hu"

                            # --- V95 VÁLTOZÁS: SZIGORÚ SKU ---
                            # Itt vettük ki a "merge" logikát.
                            # Minden egyes elem, amit az XML-ből olvasunk, önálló bejegyzés lesz.
                            # Ha a BID már létezett a bufferben (extrém ritka duplikáció az XML-en belül),
                            # akkor felülírjuk az utolsóval, de NEM vonunk össze kategóriákat.
                            
                            book_obj = {
                                "id": bid, 
                                "title": title, 
                                "url": item_data.get('link', ''), 
                                "image_url": item_data.get('image_link', ''),
                                "price": final_ron_price, 
                                "publisher": pub, 
                                "author": auth, 
                                "category": category,
                                "description": clean_desc, 
                                "stock": "instock", 
                                "lang": detected_lang, 
                                "type": "book", 
                                "last_seen": current_sync_ts
                            }
                            
                            # Extra mezők mentése
                            for k, v in item_data.items():
                                if k not in book_obj:
                                    book_obj[k] = clean_html_smart(str(v))[:500] 
                            
                            unique_books_buffer[bid] = book_obj

                    except Exception as e: pass
                    elem.clear()
                    if count_total_xml_items % 5000 == 0: gc.collect()
            
            print(f"✅ [PARSE] Kész! Egyedi SKU-k száma: {len(unique_books_buffer)}")
            print("🚀 [SMART UPLOAD] Hash ellenőrzés és feltöltés...")
            ids_batch, embeddings_batch, metadatas_batch = [], [], []
            count_processed, count_skipped, count_uploaded = 0, 0, 0
            
            for bid, book_data in unique_books_buffer.items():
                count_processed += 1
                
                # A hash most már szigorúan az adott példányhoz kötődik.
                # A "V95" prefix biztosítja, hogy mindenki frissüljön az új logikára.
                hash_input = f"V95|{bid}|{book_data['title']}|{book_data['price']}|{book_data['condition'] if 'condition' in book_data else ''}"
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

                # Agentic Embedding előkészítés
                # Itt adjuk át az AI-nak a kereséshez legfontosabb adatokat
                emb_text = f"SKU: {bid}. Nyelv: {book_data['lang']}. Cím: {book_data['title']}. Szerző: {book_data['author']}. Ár: {book_data['price']}. Kategória: {book_data['category']}. Kiadó: {book_data['publisher']}. Leírás: {book_data['description'][:800]}"
                
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

# --- BRAIN (V94 Logika + Agentic) ---
class BooksyBrain:
    def __init__(self):
        self.db = DBHandler()
        self.updater = AutoUpdater(self.db)
        self.client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.user_session_cache = {}

    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_database",
                    "description": "Keresés a könyvek vagy a szabályzatok között.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": { "type": "string" },
                            "filter_lang": { "type": "string", "enum": ["hu", "ro", "all"] },
                            "search_type": { "type": "string", "enum": ["book", "policy"] },
                            "min_price": { "type": "number" },
                            "max_price": { "type": "number" }
                        },
                        "required": ["query", "filter_lang", "search_type"]
                    }
                }
            }
        ]

    def execute_search(self, query, filter_lang, search_type, min_price=None, max_price=None):
        try:
            q_norm = normalize_text(query)
            
            if search_type == "policy":
                vec = self.client_ai.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
                res = self.db.collection.query(query_embeddings=[vec], n_results=3, where={"type": "policy"})
                return self.format_chroma_results(res)

            vec = self.client_ai.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
            where_clause = {"$and": [{"stock": "instock"}, {"type": "book"}]}
            if filter_lang != 'all' and "bookman" not in q_norm:
                where_clause = {"$and": [{"stock": "instock"}, {"type": "book"}, {"lang": filter_lang}]}
            
            matches_raw = self.db.collection.query(query_embeddings=[vec], n_results=100, where=where_clause)
            matches = self.format_chroma_results(matches_raw)
            
            results = []
            seen_items = set()
            safe_min = parse_price_to_float(min_price)
            safe_max = parse_price_to_float(max_price)

            for m in matches:
                meta = m['metadata']
                raw_db_price = meta.get('price')
                final_price_str = clean_price_raw(raw_db_price)
                
                price_val = parse_price_to_float(final_price_str)
                if price_val is not None:
                    if safe_min is not None and price_val < safe_min: continue
                    if safe_max is not None and price_val > safe_max: continue
                
                # V95 Módosítás: A szűrésnél megengedjük az azonos címeket, ha az ár/állapot más,
                # de a találati listában (seen_items) még mindig érdemes lehet szűrni a tökéletes duplikátumokat a megjelenítés miatt.
                # Itt most SKU-t (id) is belevesszük a kulcsba, hogy biztosan látszódjon minden példány.
                unique_key = f"{m['id']}" # Szigorúan ID alapú megjelenítés
                if unique_key in seen_items: continue
                seen_items.add(unique_key)
                
                base_score = (2.0 - m['score']) * 100 
                
                title_norm = normalize_text(meta.get('title', ''))
                auth_norm = normalize_text(meta.get('author', ''))
                pub_norm = normalize_text(meta.get('publisher', ''))
                cat_norm = normalize_text(meta.get('category', ''))
                
                # RELEVANCIA
                score = base_score
                keywords = q_norm.split()
                matches_keyword = False
                
                for k in keywords:
                    if len(k) > 2 and (k in title_norm or k in auth_norm or k in pub_norm or k in cat_norm):
                        score += 40
                        matches_keyword = True
                
                if "bookman" in q_norm: 
                     if "bookman" in pub_norm or "bookman" in cat_norm: score += 500
                     matches_keyword = True

                has_keywords_in_query = len([k for k in keywords if len(k)>2]) > 0
                if has_keywords_in_query and not matches_keyword and "bookman" not in q_norm:
                     continue 

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

    def process(self, msg, context_url, session_id):
        try:
            last_search = self.user_session_cache.get(session_id, "")
            site_lang = 'ro'
            if context_url and '/hu/' in str(context_url).lower(): site_lang = 'hu'

            # 1. TRIGGER
            msg_norm = normalize_text(msg)
            is_all_lang_trigger = "minden nyelven" in msg_norm or "toate limbile" in msg_norm
            
            forced_msg = msg
            if is_all_lang_trigger and last_search:
                forced_msg = f"{last_search} (search in all languages)"

            # 2. ROUTER (Agentic)
            router_system_prompt = f"""
            You are Booksy Brain.
            Last Topic: "{last_search}"
            Site Lang: {site_lang}
            
            TASKS:
            1. Detect INTENT & LANGUAGE.
            2. Call 'search_database'.
            3. 'filter_lang': 'hu' / 'ro' / 'all'. (If user says "all languages", use 'all').
            4. 'min_price'/'max_price': numbers.
            """

            messages = [
                {"role": "system", "content": router_system_prompt},
                {"role": "user", "content": forced_msg}
            ]

            response = self.client_ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.get_tools(),
                tool_choice="auto", 
                temperature=0.0
            )
            
            response_msg = response.choices[0].message
            tool_calls = response_msg.tool_calls
            
            final_products = []
            final_reply = ""
            used_lang_filter = site_lang

            if tool_calls:
                tool_call = tool_calls[0]
                if tool_call.function.name == "search_database":
                    args = json.loads(tool_call.function.arguments)
                    
                    if args.get('query'):
                        self.user_session_cache[session_id] = args.get('query')
                    
                    used_lang_filter = args.get('filter_lang', site_lang)
                    
                    # 3. ELSŐ KERESÉS
                    search_results = self.execute_search(
                        query=args.get('query'),
                        filter_lang=args.get('filter_lang'),
                        search_type=args.get('search_type'),
                        min_price=args.get('min_price'),
                        max_price=args.get('max_price')
                    )
                    
                    # 4. AUTO-RETRY
                    if not search_results and args.get('filter_lang') != 'all' and args.get('search_type') == 'book':
                        print(f"⚠️ [AUTO-RETRY] 0 találat. Kiterjesztés 'all' nyelven...")
                        search_results = self.execute_search(
                            query=args.get('query'),
                            filter_lang='all', 
                            search_type='book',
                            min_price=args.get('min_price'),
                            max_price=args.get('max_price')
                        )
                        used_lang_filter = 'all'

                    ctx_text = ""
                    if not search_results:
                        ctx_text = "No results found in database."
                    else:
                        for m in search_results:
                            meta = m['metadata']
                            if args.get('search_type') == 'policy':
                                ctx_text += f"--- POLICY ({meta.get('lang')}) ---\n{meta.get('text')}\n"
                            else:
                                p_price = clean_price_raw(meta.get('price'))
                                # Itt megjelenítjük a SKU-t is a kontextusban, ha az AI-nak szüksége lenne rá
                                details = f"SKU: {m.get('id')}, Title: {meta.get('title')}, Price: {p_price}, Publisher: {meta.get('publisher')}, Cat: {meta.get('category')}"
                                ctx_text += f"--- BOOK ---\n{details}\n"
                                p = {"title": meta.get('title'), "price": p_price, "url": meta.get('url'), "image": meta.get('image_url')}
                                final_products.append(p)
                                if len(final_products) >= 8: break
                    
                    # 5. WRITER (Agentic)
                    writer_system_prompt = f"""
                    You are Booksy. Answer based on DATA below.
                    RULES:
                    1. LANGUAGE PRIORITY: Use the User's language! If User mixes languages (Hunglish), answer in HUNGARIAN.
                    2. List books if found.
                    3. PUBLISHER CHECK: If the User asks for a specific Publisher (e.g. Kriterion) and the metadata says 'Ismeretlen', BUT the Title contains 'Kriterion', TREAT IT AS A MATCH!
                    4. If no books, say so politely.
                    
                    DATA:
                    {ctx_text}
                    """
                    
                    writer_messages = [
                        {"role": "system", "content": writer_system_prompt},
                        {"role": "user", "content": "Write response."}
                    ]
                    
                    final_res = self.client_ai.chat.completions.create(
                        model="gpt-4o-mini", messages=writer_messages, temperature=0.1
                    )
                    final_reply = final_res.choices[0].message.content

            else:
                final_reply = response_msg.content

            if final_products and used_lang_filter != 'all':
                if used_lang_filter == 'hu':
                    final_reply += "\n\n💡 Tipp: Nem ezt kerested? Írd be: 'minden nyelven', hogy a teljes adatbázisban keressünk."
                elif used_lang_filter == 'ro':
                     final_reply += "\n\n💡 Sfat: Nu ai găsit? Scrie 'toate limbile' pentru a căuta în toată baza de date."

            return {"reply": final_reply, "products": final_products}

        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            return {"reply": "Hiba történt.", "products": []}

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
def home(): return {"status": "Booksy V95 (STRICT SKU MODE)"}

@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url, req.session_id)

@app.post("/force-update")
def force(bt: BackgroundTasks):
    bt.add_task(bot.updater.run_daily_update)
    return {"status": "V95 Force Update Running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)