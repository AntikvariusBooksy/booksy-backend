# BOOKSY BRAIN - V79 (PUBLISHER REGEX FIX & CLEAN CHAT PROMPT)
# --- SQLITE FIX (CHROMADB-HEZ KÖTELEZŐ RAILWAY-EN) ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- IMPORTOK ---
import os
import time
import requests
import hashlib
import re
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

# OKOS TISZTÍTÓ (V77-től)
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

def detect_hungarian_intent(msg):
    hu_words = [
        "szia", "sziasztok", "helló", "hello", 
        "könyv", "konyv", "könyvek", "konyvek", "könyvet", 
        "keres", "keresek", "keresem", "szeretnék", "szeretnek", "vásárolni",
        "hogy", "miért", "mennyi", "mennyibe", "ár", "ara", 
        "szállítás", "szallitas", "fizetés", "fizetes", "futár",
        "van", "nincs", "mikor", "hol", 
        "kiadó", "kiado", "szerző", "szerzo", "cím", "cim", 
        "magyar", "magyarul", "minden nyelven"
    ]
    msg_norm = normalize_text(msg)
    if any(w in msg_norm for w in hu_words): return True
    return False

def clean_price_raw(raw_price):
    if not raw_price: return "0 RON"
    s = str(raw_price).strip()
    cleaned_num = re.sub(r"[^\d.,]", "", s)
    if not cleaned_num: return s 
    return f"{cleaned_num} RON"

# --- ADATBÁZIS ---
class DBHandler:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./booksy_db")
        self.collection = self.client.get_or_create_collection(name="booksy_collection")

# --- AUTO UPDATER ---
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
                    
                    # ECO MODE
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
        print(f"🔄 [AUTO] Napi Frissítés (V79 - PUBLISHER FIX)")
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

                            # V79: JAVÍTOTT REGEX - Kezeli a | jelet is, nem csak a kettőspontot
                            pub = "Ismeretlen"
                            match_pub = re.search(r'(?:Kiadó|Kiadás|Publisher)\s*(?:[:|])\s*([^|\n\r]+)', full_raw_text, re.IGNORECASE)
                            if match_pub: 
                                pub = match_pub.group(1).strip()
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
                                    unique_books_buffer[bid].update({
                                        "price": final_ron_price,
                                        "title": title 
                                    })
                            else:
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

            # 2. FÁZIS: OKOS FELTÖLTÉS
            print("🚀 [SMART UPLOAD] Hash ellenőrzés...")
            
            ids_batch = []
            embeddings_batch = []
            metadatas_batch = []
            count_processed = 0
            count_skipped = 0
            count_uploaded = 0
            
            for bid, book_data in unique_books_buffer.items():
                count_processed += 1
                
                hash_input = f"{book_data['title']}|{book_data['price']}|{book_data['category']}|{book_data['publisher']}"
                d_hash = generate_content_hash(hash_input)
                book_data['content_hash'] = d_hash
                
                # ECO CHECK
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
                        ids_batch = []
                        embeddings_batch = []
                        metadatas_batch = []
                        print(f"💾 [UPDATE] {count_uploaded} könyv frissítve...")
                        
                except Exception as e:
                    print(f"⚠️ Hiba ({bid}): {e}")

            if ids_batch:
                self.db.collection.upsert(ids=ids_batch, embeddings=embeddings_batch, metadatas=metadatas_batch)

            if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)
            print(f"🏁 [VÉGE] {count_processed} feldolgozva. ⏩ {count_skipped} változatlan (ingyen). 💾 {count_uploaded} frissítve.")

        except Exception as e: print(f"❌ Hiba: {e}")

# --- BRAIN (V79 - CLEAN OUTPUT) ---
class BooksyBrain:
    def __init__(self):
        self.db = DBHandler()
        self.updater = AutoUpdater(self.db)
        self.client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.user_session_cache = {}

    def search(self, q, search_lang_filter):
        try:
            q_norm = normalize_text(q)
            results = []
            
            # 1. POLICY
            policy_keywords = ["szallitas", "fizetes", "visszakuldes", "garancia", "kapcsolat", "bolt", "cim", "telefon", "email", "nyitva", "livrare", "plata", "contact", "cost", "cat costa"]
            if any(k in q_norm for k in policy_keywords):
                vec = self.client_ai.embeddings.create(input=q, model="text-embedding-3-small").data[0].embedding
                res = self.db.collection.query(query_embeddings=[vec], n_results=3, where={"type": "policy"})
                return self.format_chroma_results(res)

            # 2. KERESÉS
            vec = self.client_ai.embeddings.create(input=q, model="text-embedding-3-small").data[0].embedding
            
            where_clause = {"$and": [{"stock": "instock"}, {"type": "book"}]}
            if "bookman" not in q_norm and search_lang_filter != 'all':
                where_clause = {"$and": [{"stock": "instock"}, {"type": "book"}, {"lang": search_lang_filter}]}
            
            matches_raw = self.db.collection.query(query_embeddings=[vec], n_results=80, where=where_clause)
            matches = self.format_chroma_results(matches_raw)
            
            for m in matches:
                if any(r['id'] == m['id'] for r in results): continue
                meta = m['metadata']
                base_score = (2.0 - m['score']) * 100 
                
                title_norm = normalize_text(meta.get('title', ''))
                auth_norm = normalize_text(meta.get('author', ''))
                pub_norm = normalize_text(meta.get('publisher', ''))
                cat_norm = normalize_text(meta.get('category', ''))
                
                score = base_score
                if q_norm in title_norm: score += 50
                if q_norm in auth_norm: score += 30
                if q_norm in cat_norm: score += 80
                if q_norm in pub_norm: score += 40

                if "bookman" in q_norm:
                    if "bookman" in cat_norm or "bookman" in pub_norm or "bookman" in normalize_text(meta.get('description', '')):
                         score += 500
                
                m['custom_score'] = score
                results.append(m)
            
            results.sort(key=lambda x: x['custom_score'], reverse=True)
            return results[:10]
        except: return []

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
        # 1. Nyelv detektálás
        site_lang = 'ro'
        if context_url and '/hu/' in str(context_url).lower(): site_lang = 'hu'
        if detect_hungarian_intent(msg): site_lang = 'hu'
        
        # 2. Trigger Ellenőrzés
        triggers_hu = ["minden nyelven", "összes nyelven", "más nyelven"]
        triggers_ro = ["toate limbile", "alte limbi", "orice limba"]
        
        normalized_msg = normalize_text(msg).strip()
        search_query = msg
        filter_mode = site_lang 
        
        is_trigger_hu = any(t in normalized_msg for t in triggers_hu)
        is_trigger_ro = any(t in normalized_msg for t in triggers_ro)
        
        if (is_trigger_hu or is_trigger_ro) and session_id in self.user_session_cache:
            search_query = self.user_session_cache[session_id]
            filter_mode = 'all' 
            if is_trigger_hu: site_lang = 'hu'
            if is_trigger_ro: site_lang = 'ro'
        else:
            if session_id:
                self.user_session_cache[session_id] = msg

        # 3. Keresés futtatása
        matches = self.search(search_query, filter_mode)
        
        prods = []
        ctx_text = ""
        is_policy = matches and matches[0]['metadata'].get('type') == 'policy'
        is_book_search = False

        lbl_title = "Cím" if site_lang == "hu" else "Titlu"
        lbl_price = "Ár" if site_lang == "hu" else "Pret"
        lbl_pub = "Kiadó" if site_lang == "hu" else "Editura"
        lbl_cat = "Kategória" if site_lang == "hu" else "Categorie"
        
        if not matches:
             err_msg = "Sajnos nem találtam könyvet." if site_lang == 'hu' else "Nu am găsit nimic."
             return {"reply": err_msg, "products": []}

        for m in matches:
            meta = m['metadata']
            raw_db_price = meta.get('price')
            final_price = clean_price_raw(raw_db_price) 
            
            if is_policy:
                ctx_text += f"--- POLICY (Nyelv: {meta.get('lang')}) ---\n{meta.get('text', '')}\n"
            else:
                is_book_search = True
                # V79: A kategória info itt megmarad, hogy az AI tudjon róla (ha kérdezik),
                # de a promptban tiltjuk le a megjelenítését listázáskor.
                details = f"{lbl_title}: {meta.get('title')}, {lbl_price}: {final_price}, {lbl_pub}: {meta.get('publisher')}, {lbl_cat}: {meta.get('category')}"
                ctx_text += f"--- BOOK/CARTE ---\n{details}\n"
                p = {"title": meta.get('title'), "price": final_price, "url": meta.get('url'), "image": meta.get('image_url')}
                prods.append(p)
                if len(prods)>=8: break
        
        # 4. Prompt Generálás (V79 - TISZTA LISTÁZÁS)
        if site_lang == 'hu':
            sys_prompt = f"""Te a Booksy vagy, az Antikvarius.ro asszisztense. 
            KÉRDÉS: "{search_query}"
            ADATOK: {ctx_text}
            SZIGORÚ SZABÁLYOK:
            1. KIZÁRÓLAG a fenti ADATOK alapján válaszolj. 
            2. LISTÁZÁSNÁL: Csak a Címet és az Árat írd ki! NE írd ki a Kategóriát vagy a Kiadót, kivéve, ha a felhasználó kifejezetten azt kérdezi.
            3. Ha nincs adat, ne találgass.
            4. Válaszolj magyarul.
            5. Árakat mindig írd ki pontosan (pl. "20 RON")."""
        else:
            sys_prompt = f"""Ești Booksy, asistent Antikvarius.ro.
            ÎNTREBARE: "{search_query}"
            DATE: {ctx_text}
            REGULI STRICTE:
            1. Răspunde EXCLUSIV pe baza datelor de mai sus.
            2. LA LISTARE: Scrie DOAR Titlul și Prețul! NU scrie Categoria sau Editura, decât dacă utilizatorul întreabă specific.
            3. Dacă informația lipsește, nu inventa.
            4. Răspunde în română.
            5. Scrie prețurile exact (ex "20 RON")."""

        try:
            ans = self.client_ai.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role":"user", "content":sys_prompt}], temperature=0.1
            ).choices[0].message.content
        except: ans = "Hiba."

        # V78/79: TIPPEK (Marad)
        if is_book_search and filter_mode != 'all':
            if site_lang == 'hu':
                ans += "\n\n_(Tipp: Nem ezt kerested? Írd be: **'minden nyelven'**, hogy a teljes adatbázisban keressünk.)_"
            else:
                ans += "\n\n_(Sfat: Nu ai găsit? Scrie **'toate limbile'** pentru a căuta în toată baza de date.)_"

        return {"reply": ans, "products": prods}

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
def home(): return {"status": "Booksy V79 (CLEAN OUTPUT & REGEX FIX)"}

@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url, req.session_id)

@app.post("/force-update")
def force(bt: BackgroundTasks):
    bt.add_task(bot.updater.run_daily_update)
    return {"status": "V79 Force Update Running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)