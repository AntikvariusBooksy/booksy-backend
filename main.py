import os
import time
import urllib.request
import urllib.error
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
from typing import List, Optional

# --- KONFIGURÁCIÓ ---
load_dotenv()
INDEX_NAME = "booksy-index"
XML_FEED_URL = os.getenv("XML_FEED_URL", "https://www.antikvarius.ro/wp-content/uploads/woo-feed/google/xml/booksyfullfeed.xml")
TEMP_FILE = "temp_feed.xml"

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
    return "\n".join([line.strip() for line in cleantext.split('\n') if line.strip()])

def extract_author(text_content):
    if not text_content: return ""
    match = re.search(r'(Szerző|Írta|Author|Szerzők)[:\s]+([^\n|<]+)', text_content, re.IGNORECASE)
    return match.group(2).strip() if match else ""

def extract_publisher(text_content):
    if not text_content: return ""
    if "Bookman" in text_content: return "Bookman Kiadó"
    match = re.search(r'(Kiadó|Kiadás|Publisher)[:\s]+([^\n|<]+)', text_content, re.IGNORECASE)
    if match:
        pub = match.group(2).strip()
        if len(pub) > 60: return pub[:60]
        return pub
    return ""

def fuzzy_find(item, tag_suffixes):
    if isinstance(tag_suffixes, str): tag_suffixes = [tag_suffixes]
    for child in item:
        tag_name = child.tag.split('}')[-1].lower()
        for suffix in tag_suffixes:
            if tag_name == suffix.lower():
                return safe_str(child.text)
    return ""

def generate_content_hash(data_string):
    return hashlib.md5(data_string.encode('utf-8')).hexdigest()

# --- AUTOMATIZÁLT FRISSÍTŐ MOTOR (MARAD A V49/V50 STABIL) ---
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
                print(f"⬇️ [DOWNLOAD] Letöltés (Kísérlet {attempt+1}/3)...")
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

    def run_daily_update(self):
        print(f"🔄 [AUTO] Frissítés indítása (V51)")
        current_sync_ts = int(time.time())
        if not self.download_feed(): return

        try:
            print("🚀 [MODE] Parsing from Disk")
            context = ET.iterparse(TEMP_FILE, events=("end",))
            batch = []
            count_total = 0
            
            for event, elem in context:
                tag_local = elem.tag.split('}')[-1].lower()
                if tag_local in ['item', 'post']:
                    count_total += 1
                    try:
                        bid = fuzzy_find(elem, ['id', 'g:id', 'post_id'])
                        if bid:
                            title = fuzzy_find(elem, ['title', 'g:title']) or "Nincs cím"
                            desc = fuzzy_find(elem, ['description', 'g:description'])
                            short_desc = fuzzy_find(elem, ['shortdescription', 'excerpt'])
                            full_raw_text = f"{desc} {short_desc}"
                            
                            structured_text = clean_html_structural(full_raw_text)
                            auth = extract_author(structured_text)
                            pub = extract_publisher(structured_text)
                            if not pub and ("Bookman" in full_raw_text or "bookman" in full_raw_text):
                                pub = "Bookman Kiadó"

                            price = fuzzy_find(elem, ['price', 'g:price']) or "0"
                            sale = fuzzy_find(elem, ['sale_price', 'g:sale_price']) or ""
                            
                            # Hash ellenőrzés
                            d_hash = generate_content_hash(f"{bid}{title}{pub}{price}{sale}")
                            need_emb = True
                            
                            # Itt most már bekapcsolhatjuk az ellenőrzést, mert az adatbázis friss!
                            try:
                                ex = self.index.fetch(ids=[bid])
                                if ex and 'vectors' in ex and bid in ex['vectors']:
                                    if ex['vectors'][bid]['metadata'].get('content_hash') == d_hash:
                                        need_emb = False
                            except: pass

                            if need_emb:
                                if count_total % 500 == 0: print(f"⏳ [PROG] {count_total}...")
                                emb_text = f"Cím: {title}. Szerző: {auth}. Kiadó: {pub}. Leírás: {structured_text[:600]}"
                                emb = self.client_ai.embeddings.create(input=emb_text[:8000], model="text-embedding-3-small").data[0].embedding
                                
                                meta = {
                                    "title": title, "url": fuzzy_find(elem, ['link', 'g:link']), 
                                    "image_url": fuzzy_find(elem, ['image_link', 'g:image_link']),
                                    "price": price, "lang": "hu", "stock": "instock", 
                                    "author": auth, "publisher": pub, 
                                    "full_search_text": f"{title} {auth} {pub}".lower(),
                                    "content_hash": d_hash, "last_seen": current_sync_ts
                                }
                                batch.append((bid, emb, meta))

                    except Exception as e: pass
                    elem.clear()
                    if count_total % 500 == 0: gc.collect()
                    if len(batch) >= 50:
                        self.index.upsert(vectors=batch)
                        batch = []

            if batch: self.index.upsert(vectors=batch)
            print("🧹 [AUTO] Takarítás...")
            try: self.index.delete(filter={"last_seen": {"$lt": current_sync_ts}, "type": {"$ne": "policy"}})
            except: pass
            if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)
            print(f"🏁 [VÉGE] Kész! ({count_total} könyv)")

        except Exception as e: print(f"❌ Hiba: {e}")

# --- BRAIN (V51: HYBRID SEARCH) ---
class BooksyBrain:
    def __init__(self):
        self.updater = AutoUpdater()
        self.client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.index = Pinecone(api_key=os.getenv("PINECONE_API_KEY")).Index(INDEX_NAME)

    def search(self, q, search_lang_filter):
        try:
            q_norm = normalize_text(q)
            results = []
            
            # --- 1. SPECIÁLIS "BOOKMAN" SZŰRŐ (HARD FILTER) ---
            if "bookman" in q_norm:
                print("🔎 [SEARCH] Bookman detected -> Using Hard Filter")
                # Létrehozunk egy dummy vektort a kereséshez, de a FILTER a lényeg!
                vec = self.client_ai.embeddings.create(input=q, model="text-embedding-3-small").data[0].embedding
                
                # CSAK azokat kérjük le, ahol a publisher pontosan "Bookman Kiadó"
                direct_res = self.index.query(
                    vector=vec, 
                    top_k=20, 
                    include_metadata=True, 
                    filter={"publisher": "Bookman Kiadó", "stock": "instock"}
                )
                
                # Ezeket azonnal betesszük az eredmények közé, maximális prioritással
                for m in direct_res['matches']:
                    m['custom_score'] = 10000 # Hatalmas pontszám, hogy ez legyen elöl
                    results.append(m)
            
            # --- 2. NORMÁL SZEMANTIKUS KERESÉS ---
            # Ez minden másra fut, vagy kiegészítésként
            vec = self.client_ai.embeddings.create(input=q, model="text-embedding-3-small").data[0].embedding
            filt = {"stock": "instock"}
            if search_lang_filter != 'all': filt["lang"] = search_lang_filter
            
            normal_res = self.index.query(vector=vec, top_k=60, include_metadata=True, filter=filt)
            
            for m in normal_res['matches']:
                # Ha már megtaláltuk a Bookman szűrővel, ne rakjuk be duplán
                if any(r['id'] == m['id'] for r in results):
                    continue
                    
                meta = m['metadata']
                score = m['score'] * 100 
                
                # Súlyozás
                pub_norm = normalize_text(meta.get('publisher', ''))
                auth_norm = normalize_text(meta.get('author', ''))
                title_norm = normalize_text(meta.get('title', ''))
                
                if q_norm in pub_norm: score += 500
                if q_norm in auth_norm: score += 300
                if q_norm in title_norm: score += 200
                
                m['custom_score'] = score
                results.append(m)
            
            # Rendezés
            results.sort(key=lambda x: x['custom_score'], reverse=True)
            return results[:10]
            
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def process(self, msg, context_url=""):
        site_lang = 'hu'
        if context_url and '/ro/' in str(context_url).lower(): site_lang = 'ro'
        
        matches = self.search(msg, site_lang)
        
        prods = []
        ctx_text = ""
        
        if not matches:
            if site_lang == 'hu':
                return {"reply": "Sajnos nem találtam a keresésednek megfelelő könyvet.", "products": []}
            else:
                return {"reply": "Nu am găsit nicio carte.", "products": []}

        for m in matches:
            meta = m['metadata']
            display_title = meta.get('title')
            publisher = meta.get('publisher', '')
            
            info_line = f"- {display_title} (Szerző: {meta.get('author')}, Kiadó: {publisher}, Ár: {meta.get('price')})\n"
            ctx_text += info_line
            
            p = {"title": display_title, "price": meta.get('price'), "url": meta.get('url'), "image": meta.get('image_url')}
            prods.append(p)
            if len(prods)>=8: break
            
        if site_lang == 'hu':
            sys_prompt = f"""
            Te a Booksy vagy, az Antikvarius.ro segítőkész asszisztense.
            Kérdés: "{msg}"
            Talált könyvek:
            {ctx_text}
            
            Utasítások:
            1. Csak a fenti könyveket ajánld!
            2. Ha a találatok között van 'Bookman Kiadó', emeld ki, hogy ezek a saját kiadásaitok!
            3. Magyarul válaszolj.
            """
        else:
            sys_prompt = f"""
            Ești Booksy. Întrebare: "{msg}"
            Cărți găsite:
            {ctx_text}
            
            Instrucțiuni:
            1. Recomandă doar cărțile din listă.
            2. Dacă vezi 'Bookman Kiadó', menționează că sunt ediții proprii.
            3. Răspunde în română.
            """

        try:
            ans = self.client_ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user", "content":sys_prompt}],
                temperature=0.3
            ).choices[0].message.content
        except:
            ans = "Hiba történt."

        return {"reply": ans, "products": prods}

# --- APP ---
bot = BooksyBrain()
scheduler = BackgroundScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.start()
    yield
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def home(): return {"status": "Booksy V51 (Hybrid Hard-Filter)"}

@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url)

@app.post("/force-update")
def force(bt: BackgroundTasks):
    bt.add_task(bot.updater.run_daily_update)
    return {"status": "Daily Update Started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)