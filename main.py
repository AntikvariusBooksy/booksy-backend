# BOOKSY BRAIN - V72 (RAW RON - TEXT CLEANING ONLY)
# --- SQLITE FIX (CHROMADB-HEZ KÖTELEZŐ RAILWAY-EN) ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- NORMÁL IMPORTOK ---
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

# --- TUDÁSBÁZIS (RO LINKS) ---
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

# --- HELPEREK ---
def normalize_text(text):
    if not text: return ""
    text = str(text).lower()
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def safe_str(val):
    if val is None: return ""
    return html.unescape(str(val).strip())

def clean_html_structural(raw_html):
    if not raw_html: return ""
    s = safe_str(raw_html)
    s = s.replace('</div>', '\n').replace('</p>', '\n').replace('<br>', '\n').replace('<br/>', '\n')
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', s)
    cleantext = cleantext.replace("<![CDATA[", "").replace("]]>", "")
    cleantext = re.sub(r'<script.*?>.*?</script>', '', cleantext, flags=re.DOTALL)
    cleantext = re.sub(r'<style.*?>.*?</style>', '', cleantext, flags=re.DOTALL)
    return "\n".join([line.strip() for line in cleantext.split('\n') if line.strip()])

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
        "magyar", "magyarul"
    ]
    msg_norm = normalize_text(msg)
    if any(w in msg_norm for w in hu_words): return True
    return False

# --- ÁR TISZTÍTÓ (V72 - RON ONLY) ---
def clean_price_raw(raw_price):
    """
    Ez a függvény eltávolítja a szöveget (pl. 'RON') a számból,
    majd visszaadja a számot + ' RON' stringet.
    Így nem számít, hogy a feedben "24 RON" vagy "24,00" van.
    """
    if not raw_price: return "0 RON"
    
    s = str(raw_price).strip()
    
    # Csak számjegyeket, pontot és vesszőt tartunk meg
    cleaned_num = re.sub(r"[^\d.,]", "", s)
    
    if not cleaned_num: return s 
    
    return f"{cleaned_num} RON"

# --- ADATBÁZIS KEZELŐ (CHROMADB) ---
class DBHandler:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./booksy_db")
        self.collection = self.client.get_or_create_collection(name="booksy_collection")

# --- OPTIMALIZÁLT FRISSÍTŐ MOTOR (V72) ---
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
        print("ℹ️ [POLICY] Román információs oldalak frissítése...")
        headers = {'User-Agent': 'BooksyBot/1.0'}
        for page in POLICY_PAGES:
            try:
                url = page['url']
                r = requests.get(url, headers=headers, timeout=30)
                if r.status_code == 200:
                    raw_html = r.text
                    clean_text = clean_html_structural(raw_html)
                    if len(clean_text) > 20000: clean_text = clean_text[:20000]

                    d_hash = generate_content_hash(clean_text)
                    page_id = f"policy_{generate_content_hash(url)}"
                    
                    emb_text = f"Típus: Szabályzat (ro). Cím: {page['name']}. Tartalom: {clean_text[:8000]}"
                    emb = self.client_ai.embeddings.create(input=emb_text, model="text-embedding-3-small").data[0].embedding
                    
                    meta = {
                        "title": page['name'], "url": url, "text": clean_text[:25000],
                        "lang": "ro", "type": "policy", "content_hash": d_hash, "last_seen": current_ts
                    }
                    self.db.collection.upsert(
                        ids=[page_id],
                        embeddings=[emb],
                        metadatas=[meta]
                    )
                    print(f"   ✅ [POLICY] OK: {page['name']}")
                else: print(f"   ⚠️ Hiba: {r.status_code} - {url}")
            except Exception as e: print(f"   ❌ Hiba: {e}")

    def run_daily_update(self):
        print(f"🔄 [AUTO] Napi Frissítés Indítása (V72 - Raw RON Clean)")
        current_sync_ts = int(time.time())
        
        self.update_policies(current_sync_ts)
        
        if not self.download_feed(): return

        try:
            print("🚀 [MODE] Parsing Books from Disk")
            context = ET.iterparse(TEMP_FILE, events=("end",))
            
            ids_batch = []
            embeddings_batch = []
            metadatas_batch = []
            
            count_total = 0
            count_updated = 0
            count_skipped = 0
            
            for event, elem in context:
                tag_local = elem.tag.split('}')[-1].lower()
                if tag_local in ['item', 'post']:
                    count_total += 1
                    try:
                        item_data = extract_all_data(elem)
                        bid = item_data.get('id') or item_data.get('post_id') or item_data.get('g:id')
                        
                        if bid:
                            title = item_data.get('title') or "Nincs cím"
                            desc = item_data.get('description', '')
                            short_desc = item_data.get('shortdescription', '') or item_data.get('excerpt', '')
                            full_raw_text = f"{desc} {short_desc}"
                            clean_desc = clean_html_structural(full_raw_text)
                            
                            category = item_data.get('product_type') or item_data.get('category') or ""
                            cat_norm = normalize_text(category)
                            
                            pub = "Ismeretlen"
                            if "bookman" in cat_norm:
                                pub = "Bookman Kiadó"
                            else:
                                match_pub = re.search(r'(Kiadó|Kiadás|Publisher)(?:\s|<[^>]+>)*:?(?:\s|<[^>]+>)+([^<\n\r]+)', full_raw_text, re.IGNORECASE)
                                if match_pub:
                                    extracted = match_pub.group(2).strip()
                                    if "bookman" in extracted.lower(): pub = "Bookman Kiadó"
                                    else: pub = extracted
                            
                            match_auth = re.search(r'(Szerző|Írta|Author|Szerzők)(?:\s|<[^>]+>)*:?(?:\s|<[^>]+>)+([^<\n\r]+)', full_raw_text, re.IGNORECASE)
                            auth = match_auth.group(2).strip() if match_auth else "Ismeretlen"

                            detected_lang = "hu"
                            if "carti in limba romana" in cat_norm: detected_lang = "ro"
                            elif "magyar nyelvu konyvek" in cat_norm: detected_lang = "hu"
                            
                            # --- ÁR KEZELÉS (V72: TISZTÍTÁS CSAK) ---
                            raw_price = item_data.get('sale_price') or item_data.get('price') or "0"
                            final_ron_price = clean_price_raw(raw_price)
                            
                            hash_input = "".join([f"{k}:{v}" for k, v in sorted(item_data.items())])
                            hash_input += f"|{detected_lang}|{pub}|{auth}"
                            d_hash = generate_content_hash(hash_input)
                            
                            need_emb = True
                            try:
                                existing = self.db.collection.get(ids=[bid], include=['metadatas'])
                                if existing and existing['ids']:
                                    existing_meta = existing['metadatas'][0]
                                    if existing_meta.get('content_hash') == d_hash:
                                        need_emb = False
                                        count_skipped += 1
                            except: pass
                            
                            if need_emb:
                                if count_total % 500 == 0: print(f"⏳ [PROG] {count_total}... (Upd: {count_updated})")
                                
                                emb_text = f"Nyelv: {detected_lang}. Cím: {title}. Szerző: {auth}. Kategória: {category}. Kiadó: {pub}. Leírás: {clean_desc[:800]}"
                                emb = self.client_ai.embeddings.create(input=emb_text[:8000], model="text-embedding-3-small").data[0].embedding
                                
                                meta = {
                                    "title": title, "url": item_data.get('link', ''), "image_url": item_data.get('image_link', ''),
                                    "price": final_ron_price,
                                    "publisher": pub, "author": auth, "category": category,
                                    "stock": "instock", "lang": detected_lang, "content_hash": d_hash,
                                    "last_seen": current_sync_ts, "type": "book"
                                }
                                for k, v in item_data.items():
                                    if k not in meta:
                                        clean_v = clean_html_structural(str(v))
                                        if len(clean_v) > 1000: clean_v = clean_v[:1000]
                                        meta[k] = clean_v
                                
                                ids_batch.append(bid)
                                embeddings_batch.append(emb)
                                metadatas_batch.append(meta)
                                count_updated += 1

                    except Exception as e: pass
                    elem.clear()
                    
                    if count_total % 500 == 0: gc.collect()
                    
                    if len(ids_batch) >= 50:
                        self.db.collection.upsert(ids=ids_batch, embeddings=embeddings_batch, metadatas=metadatas_batch)
                        ids_batch = []
                        embeddings_batch = []
                        metadatas_batch = []

            if ids_batch:
                self.db.collection.upsert(ids=ids_batch, embeddings=embeddings_batch, metadatas=metadatas_batch)

            print("🧹 [AUTO] Takarítás...")
            if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)
            print(f"🏁 [VÉGE] Összes: {count_total}, Frissítve: {count_updated}, Skip: {count_skipped}")

        except Exception as e: print(f"❌ Hiba: {e}")

# --- BRAIN (V72) ---
class BooksyBrain:
    def __init__(self):
        self.db = DBHandler()
        self.updater = AutoUpdater(self.db)
        self.client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def search(self, q, search_lang_filter):
        try:
            q_norm = normalize_text(q)
            results = []
            
            # 1. POLICY
            policy_keywords = ["szallitas", "fizetes", "visszakuldes", "garancia", "kapcsolat", "bolt", "cim", "telefon", "email", "nyitva", "livrare", "plata", "contact"]
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

    def process(self, msg, context_url=""):
        site_lang = 'ro'
        if context_url and '/hu/' in str(context_url).lower(): site_lang = 'hu'
        if detect_hungarian_intent(msg): site_lang = 'hu'
        
        matches = self.search(msg, site_lang)
        prods = []
        ctx_text = ""
        is_policy = matches and matches[0]['metadata'].get('type') == 'policy'
        
        lbl_title = "Cím" if site_lang == "hu" else "Titlu"
        lbl_price = "Ár" if site_lang == "hu" else "Pret"
        lbl_pub = "Kiadó" if site_lang == "hu" else "Editura"
        lbl_cat = "Kategória" if site_lang == "hu" else "Categorie"
        
        if not matches:
             err_msg = "Sajnos nem találtam könyvet." if site_lang == 'hu' else "Nu am găsit nimic."
             return {"reply": err_msg, "products": []}

        for m in matches:
            meta = m['metadata']
            
            # Röptében is tisztítunk (ha esetleg régi adat jönne)
            raw_db_price = meta.get('price')
            final_price = clean_price_raw(raw_db_price) 
            
            if is_policy:
                ctx_text += f"--- POLICY (Nyelv: {meta.get('lang')}) ---\n{meta.get('text', '')}\n"
            else:
                details = f"{lbl_title}: {meta.get('title')}, {lbl_price}: {final_price}, {lbl_pub}: {meta.get('publisher')}, {lbl_cat}: {meta.get('category')}"
                ctx_text += f"--- BOOK/CARTE ---\n{details}\n"
                p = {"title": meta.get('title'), "price": final_price, "url": meta.get('url'), "image": meta.get('image_url')}
                prods.append(p)
                if len(prods)>=8: break
            
        if site_lang == 'hu':
            sys_prompt = f"""Te a Booksy vagy, az Antikvarius.ro asszisztense. Kérdés: "{msg}" ADATOK: {ctx_text}
            UTASÍTÁS: 
            1. Válaszolj magyarul, kedvesen, röviden. 
            2. NE HASZNÁLJ KÉPET/LINKET. 
            3. Policy: Fordítsd magyarra.
            4. ÁRAK: Az adatokban lévő ár (pl. "24,00 RON") a helyes. Írd ki pontosan így! Ne számolj át semmit!"""
        else:
            sys_prompt = f"""Ești Booksy. Date: {ctx_text}
            Instructiuni: 
            1. Răspunde în română, scurt. 
            2. NU include imagini/link-uri.
            3. PRETURI: Datele conțin prețul corect (de ex "24,00 RON"). Scrie-l exact așa!"""

        try:
            ans = self.client_ai.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role":"user", "content":sys_prompt}], temperature=0.3
            ).choices[0].message.content
        except: ans = "Hiba."
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
def home(): return {"status": "Booksy V72 (RAW RON)"}

@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url)

@app.post("/force-update")
def force(bt: BackgroundTasks):
    bt.add_task(bot.updater.run_daily_update)
    return {"status": "V72 Force Update Running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)