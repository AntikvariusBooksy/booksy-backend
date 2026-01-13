import os
import time
import requests
import hashlib
import re
import unicodedata
import html
import xml.etree.ElementTree as ET
import gc
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from typing import List, Optional, Dict, Any

# --- KONFIGURÁCIÓ ---
load_dotenv()
INDEX_NAME = "booksy-index"
XML_FEED_URL = os.getenv("XML_FEED_URL", "https://www.antikvarius.ro/wp-content/uploads/woo-feed/google/xml/booksyfullfeed.xml")
TEMP_FILE = "temp_feed.xml"

# --- TUDÁSBÁZIS URL LISTA (KIZÁRÓLAG A MEGADOTT 4 ROMÁN LINK) ---
# Az AI feladata lesz ezt magyarra fordítani, ha szükséges.
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

# --- AUTOMATIZÁLT FRISSÍTŐ MOTOR (V62) ---
class AutoUpdater:
    def __init__(self):
        self.api_key_openai = os.getenv("OPENAI_API_KEY")
        self.api_key_pinecone = os.getenv("PINECONE_API_KEY")
        self.client_ai = OpenAI(api_key=self.api_key_openai)
        self.pc = Pinecone(api_key=self.api_key_pinecone)
        self.index = self.pc.Index(INDEX_NAME)

    def download_feed(self):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
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
        """Csak a megadott román oldalak letöltése."""
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
                    
                    # Embedding: Jelezzük, hogy ez román nyelvű
                    emb_text = f"Típus: Szabályzat (ro). Cím: {page['name']}. Tartalom: {clean_text[:8000]}"
                    emb = self.client_ai.embeddings.create(input=emb_text, model="text-embedding-3-small").data[0].embedding
                    
                    meta = {
                        "title": page['name'],
                        "url": url,
                        "text": clean_text[:25000],
                        "lang": "ro", # Mindig román, mert csak ezeket töltjük le
                        "type": "policy",
                        "content_hash": d_hash,
                        "last_seen": current_ts
                    }
                    
                    self.index.upsert(vectors=[(page_id, emb, meta)])
                    print(f"   ✅ [POLICY] OK: {page['name']}")
                else:
                    print(f"   ⚠️ Hiba: {r.status_code} - {url}")
            except Exception as e:
                print(f"   ❌ Hiba: {e}")

    def run_daily_update(self):
        print(f"🔄 [AUTO] Napi Frissítés Indítása (V62)")
        current_sync_ts = int(time.time())
        
        # 1. POLICY
        self.update_policies(current_sync_ts)
        
        # 2. KÖNYVEK
        if not self.download_feed(): return

        try:
            print("🚀 [MODE] Parsing Books from Disk")
            context = ET.iterparse(TEMP_FILE, events=("end",))
            batch = []
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
                            
                            pub = "Ismeretlen"
                            match_pub = re.search(r'(Kiadó|Kiadás|Publisher)(?:\s|<[^>]+>)*:?(?:\s|<[^>]+>)+([^<\n\r]+)', full_raw_text, re.IGNORECASE)
                            if match_pub:
                                pub = match_pub.group(2).strip()
                                if "bookman" in pub.lower(): pub = "Bookman Kiadó"
                            
                            match_auth = re.search(r'(Szerző|Írta|Author|Szerzők)(?:\s|<[^>]+>)*:?(?:\s|<[^>]+>)+([^<\n\r]+)', full_raw_text, re.IGNORECASE)
                            auth = match_auth.group(2).strip() if match_auth else "Ismeretlen"

                            # Kategória és Nyelv (Szigorú)
                            category = item_data.get('product_type') or item_data.get('category') or ""
                            cat_norm = normalize_text(category)
                            detected_lang = "hu"
                            # Mivel a gyökérkategóriák fixek:
                            if "carti in limba romana" in cat_norm: detected_lang = "ro"
                            elif "magyar nyelvu konyvek" in cat_norm: detected_lang = "hu"
                            
                            price = item_data.get('sale_price') or item_data.get('price') or "0"
                            
                            hash_input = "".join([f"{k}:{v}" for k, v in sorted(item_data.items())])
                            hash_input += f"|{detected_lang}|{pub}|{auth}"
                            d_hash = generate_content_hash(hash_input)
                            
                            need_emb = True
                            try:
                                fetch_res = self.index.fetch(ids=[bid])
                                if fetch_res and 'vectors' in fetch_res and bid in fetch_res['vectors']:
                                    existing_meta = fetch_res['vectors'][bid]['metadata']
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
                                    "price": price, "publisher": pub, "author": auth, "category": category,
                                    "stock": "instock", "lang": detected_lang, "content_hash": d_hash,
                                    "last_seen": current_sync_ts,
                                    "type": "book"
                                }
                                for k, v in item_data.items():
                                    if k not in meta:
                                        clean_v = clean_html_structural(str(v))
                                        if len(clean_v) > 1000: clean_v = clean_v[:1000]
                                        meta[k] = clean_v

                                batch.append((bid, emb, meta))
                                count_updated += 1

                    except Exception as e: pass
                    elem.clear()
                    if count_total % 500 == 0: gc.collect()
                    if len(batch) >= 50:
                        self.index.upsert(vectors=batch)
                        batch = []

            if batch: self.index.upsert(vectors=batch)
            
            # Takarítás
            one_week_ago = current_sync_ts - (7 * 24 * 60 * 60)
            try: self.index.delete(filter={"last_seen": {"$lt": one_week_ago}, "type": "book"})
            except: pass
            
            if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)
            print(f"🏁 [VÉGE] Összes: {count_total}, Frissítve: {count_updated}, Skip: {count_skipped}")

        except Exception as e: print(f"❌ Hiba: {e}")

# --- BRAIN (V62 - TRANSLATOR LOGIC) ---
class BooksyBrain:
    def __init__(self):
        self.updater = AutoUpdater()
        self.client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.index = Pinecone(api_key=os.getenv("PINECONE_API_KEY")).Index(INDEX_NAME)

    def search(self, q, search_lang_filter):
        try:
            q_norm = normalize_text(q)
            results = []
            
            # 1. POLICY DETECT
            policy_keywords = ["szallitas", "fizetes", "visszakuldes", "garancia", "kapcsolat", "bolt", "cim", "telefon", "email", "nyitva", "livrare", "plata", "contact"]
            if any(k in q_norm for k in policy_keywords):
                vec = self.client_ai.embeddings.create(input=q, model="text-embedding-3-small").data[0].embedding
                # Policy esetén nem szűrünk nyelvre, mert csak ROMÁN van! 
                # Az AI majd fordít.
                return self.index.query(vector=vec, top_k=3, include_metadata=True, filter={"type": "policy"})['matches']

            # 2. BOOKMAN FILTER
            if "bookman" in q_norm:
                vec = self.client_ai.embeddings.create(input=q, model="text-embedding-3-small").data[0].embedding
                direct_res = self.index.query(vector=vec, top_k=20, include_metadata=True, filter={"publisher": "Bookman Kiadó", "stock": "instock"})
                for m in direct_res['matches']:
                    m['custom_score'] = 10000 
                    results.append(m)
            
            # 3. NORMÁL KERESÉS
            vec = self.client_ai.embeddings.create(input=q, model="text-embedding-3-small").data[0].embedding
            filt = {"stock": "instock", "type": "book"}
            if search_lang_filter != 'all': filt["lang"] = search_lang_filter
            normal_res = self.index.query(vector=vec, top_k=60, include_metadata=True, filter=filt)
            
            for m in normal_res['matches']:
                if any(r['id'] == m['id'] for r in results): continue
                meta = m['metadata']
                score = m['score'] * 100 
                
                title_norm = normalize_text(meta.get('title', ''))
                auth_norm = normalize_text(meta.get('author', ''))
                pub_norm = normalize_text(meta.get('publisher', ''))
                cat_norm = normalize_text(meta.get('category', ''))
                desc_norm = normalize_text(meta.get('description', ''))
                
                if q_norm in title_norm: score += 1000
                if q_norm in auth_norm: score += 600
                if q_norm in cat_norm: score += 400
                if q_norm in pub_norm: score += 300
                if q_norm in desc_norm: score += 100
                
                m['custom_score'] = score
                results.append(m)
            
            results.sort(key=lambda x: x['custom_score'], reverse=True)
            return results[:10]
        except: return []

    def process(self, msg, context_url=""):
        # NYELV DETEKTÁLÁS (URL ALAPJÁN)
        # Alapértelmezett: Román (Mert https://www.antikvarius.ro/ a főoldal)
        site_lang = 'ro'
        # Ha az URL-ben benne van, hogy /hu/, akkor váltunk magyarra
        if context_url and '/hu/' in str(context_url).lower(): 
            site_lang = 'hu'
        
        matches = self.search(msg, site_lang)
        
        prods = []
        ctx_text = ""
        is_policy = matches and matches[0]['metadata'].get('type') == 'policy'
        
        if not matches: return {"reply": "Nincs találat." if site_lang=='hu' else "Nu am găsit nimic.", "products": []}

        for m in matches:
            meta = m['metadata']
            if is_policy:
                # Jelöljük a promptban, hogy ez román szöveg
                ctx_text += f"--- POLICY (Nyelv: {meta.get('lang')}) ---\n{meta.get('text', '')}\n"
            else:
                details = ", ".join([f"{k}: {v}" for k, v in meta.items() if k not in ['full_search_text', 'content_hash', 'last_seen', 'description', 'text']])
                ctx_text += f"--- KÖNYV ---\n{details}\n"
                p = {"title": meta.get('title'), "price": meta.get('price'), "url": meta.get('url'), "image": meta.get('image_url')}
                prods.append(p)
                if len(prods)>=8: break
            
        if site_lang == 'hu':
            # MAGYAR PROMPT (Fordítási utasítással)
            sys_prompt = f"""Te a Booksy vagy, az Antikvarius.ro asszisztense.
            Kérdés: "{msg}"
            
            ADATOK (TUDÁSBÁZIS):
            {ctx_text}
            
            UTASÍTÁS:
            1. Válaszolj pontosan az adatok alapján.
            2. FIGYELEM: A szabályzatok (Policy) szövege fennebb ROMÁNUL van. Ha a felhasználó magyarul kérdezett (és most magyarul kérdez), akkor FORDÍTSD LE az információt és válaszolj MAGYARUL.
            3. Ha könyv, és 'Bookman Kiadó', emeld ki.
            Válasz nyelve: MAGYAR."""
        else:
            # ROMÁN PROMPT
            sys_prompt = f"""Ești Booksy. Date: {ctx_text}
            Răspunde exact pe baza datelor. 
            Răspunde în română."""

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
def home(): return {"status": "Booksy V62 (CORRECT RO LINKS)"}

@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url)

@app.post("/force-update")
def force(bt: BackgroundTasks):
    bt.add_task(bot.updater.run_daily_update)
    return {"status": "V62 Force Update Started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)