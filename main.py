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

# --- AUTOMATIZÁLT FRISSÍTŐ MOTOR (STEALTH MODE) ---
class AutoUpdater:
    def __init__(self):
        self.api_key_openai = os.getenv("OPENAI_API_KEY")
        self.api_key_pinecone = os.getenv("PINECONE_API_KEY")
        self.client_ai = OpenAI(api_key=self.api_key_openai)
        self.pc = Pinecone(api_key=self.api_key_pinecone)
        self.index = self.pc.Index(INDEX_NAME)

    def run_daily_update(self):
        print(f"🔄 [AUTO] Frissítés (Stealth Mode): {XML_FEED_URL}")
        current_sync_ts = int(time.time())
        
        # ÁLCÁZÁS: Böngészőnek hazudjuk magunkat
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/xml,text/xml,application/xhtml+xml',
            'Connection': 'keep-alive'
        }

        # ÚJRAPRÓBÁLKOZÁS LOGIKA
        max_retries = 3
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(XML_FEED_URL, headers=headers)
                
                with urllib.request.urlopen(req, timeout=300) as response:
                    events = ET.iterparse(response, events=("start", "end"))
                    context = iter(events)
                    event, root = next(context)

                    print(f"🚀 [MODE] V48 Stealth Stream (Kísérlet: {attempt+1})")
                    
                    batch = []
                    count_total = 0
                    count_processed = 0
                    
                    for event, elem in context:
                        if event == "end":
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

                                        # Hash ellenőrzés
                                        price = fuzzy_find(elem, ['price', 'g:price']) or "0"
                                        sale = fuzzy_find(elem, ['sale_price', 'g:sale_price']) or ""
                                        # Force Update a Bookman miatt
                                        # d_hash = generate_content_hash(f"{bid}{title}{pub}{price}{sale}")
                                        d_hash = f"v48_force_{current_sync_ts}"
                                        need_emb = True

                                        if need_emb:
                                            # Csak Bookman vagy minden 200. könyv logolása
                                            if "Bookman" in pub and count_processed % 5 == 0:
                                                print(f"✅ [FOUND] {title} ({pub})")
                                            elif count_total % 500 == 0: 
                                                print(f"⏳ [PROG] {count_total} feldolgozva...")

                                            emb_text = f"Cím: {title}. Szerző: {auth}. Kiadó: {pub}. Leírás: {structured_text[:600]}"
                                            emb = self.client_ai.embeddings.create(input=emb_text[:8000], model="text-embedding-3-small").data[0].embedding
                                            
                                            meta = {
                                                "title": title,
                                                "url": fuzzy_find(elem, ['link', 'g:link']), 
                                                "image_url": fuzzy_find(elem, ['image_link', 'g:image_link']),
                                                "price": price, "lang": "hu", "stock": "instock", 
                                                "author": auth, "publisher": pub, 
                                                "full_search_text": f"{title} {auth} {pub}".lower(),
                                                "content_hash": d_hash, "last_seen": current_sync_ts
                                            }
                                            batch.append((bid, emb, meta))
                                            count_processed += 1

                                except Exception as e: pass
                                
                                elem.clear()
                                root.clear()
                                
                                if count_total % 500 == 0: gc.collect()
                                
                                if len(batch) >= 50:
                                    self.index.upsert(vectors=batch)
                                    batch = []

                    if batch: self.index.upsert(vectors=batch)
                    
                    print("🧹 [AUTO] Régi elemek törlése...")
                    try: self.index.delete(filter={"last_seen": {"$lt": current_sync_ts}, "type": {"$ne": "policy"}})
                    except: pass
                    
                    print(f"🏁 [VÉGE] Kész! ({count_processed} könyv frissítve)")
                    return # Siker esetén kilépünk a retry loopból

            except (urllib.error.URLError, ConnectionResetError) as e:
                print(f"⚠️ Hiba a letöltésnél (Kísérlet {attempt+1}/3): {e}")
                time.sleep(5) # Vár 5 mp-et újrapróbálkozás előtt
            except Exception as e:
                print(f"❌ Váratlan hiba: {e}")
                return

# --- BRAIN ---
class BooksyBrain:
    def __init__(self):
        self.updater = AutoUpdater()
        self.client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.index = Pinecone(api_key=os.getenv("PINECONE_API_KEY")).Index(INDEX_NAME)

    def search(self, q, search_lang_filter):
        try:
            vec = self.client_ai.embeddings.create(input=q, model="text-embedding-3-small").data[0].embedding
            filt = {"stock": "instock"}
            if search_lang_filter != 'all': filt["lang"] = search_lang_filter
            res = self.index.query(vector=vec, top_k=80, include_metadata=True, filter=filt)
            
            q_norm = normalize_text(q)
            results = []
            for m in res['matches']:
                meta = m['metadata']
                score = m['score'] * 100 
                
                pub_norm = normalize_text(meta.get('publisher', ''))
                auth_norm = normalize_text(meta.get('author', ''))
                title_norm = normalize_text(meta.get('title', ''))
                
                if q_norm in pub_norm and len(q_norm) > 3: score += 1000 
                if q_norm in auth_norm: score += 300
                if q_norm in title_norm: score += 200
                
                m['custom_score'] = score
                results.append(m)
            results.sort(key=lambda x: x['custom_score'], reverse=True)
            return results[:10]
        except: return []

    def process(self, msg, context_url=""):
        site_lang = 'hu'
        if context_url and '/ro/' in str(context_url).lower(): site_lang = 'ro'
        matches = self.search(msg, site_lang)
        if not matches: return {"reply": "Sajnos nem találtam a keresésnek megfelelő könyvet.", "products": []}
        
        prods = []
        ctx_text = ""
        for m in matches:
            meta = m['metadata']
            display = meta.get('title')
            if meta.get('publisher'): display += f" ({meta.get('publisher')})"
            p = {"title": display, "price": meta.get('price'), "url": meta.get('url'), "image": meta.get('image_url')}
            prods.append(p)
            ctx_text += f"- {display}\n"
            if len(prods)>=8: break
            
        sys_prompt = f"User: {msg}. Found:\n{ctx_text}\nTask: Recommend these books."
        ans = self.client_ai.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user", "content":sys_prompt}]).choices[0].message.content
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
def home(): return {"status": "Booksy V48 (Stealth + Retry)"}

@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url)

@app.post("/force-update")
def force(bt: BackgroundTasks):
    bt.add_task(bot.updater.run_daily_update)
    return {"status": "V48 Stealth Update Started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)