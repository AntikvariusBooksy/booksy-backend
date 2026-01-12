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
    """
    Kinyeri a szöveget és EGYBŐL dekódolja a HTML entitásokat.
    Így a '&#8211;' -> '–' lesz mindenhol (címben is).
    """
    if val is None: return ""
    # 1. Stringgé alakítjuk
    text = str(val).strip()
    # 2. Dekódoljuk a HTML jeleket (&amp;, &#8211;, &lt; stb.)
    return html.unescape(text)

def clean_html_structural(raw_html):
    if not raw_html: return ""
    # Itt a safe_str már megcsinálta az unescape-et, de a biztonság kedvéért kezeljük
    s = str(raw_html)
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
                # A safe_str hívás itt történik, tehát minden kinyert adat tiszta lesz!
                return safe_str(child.text)
    return ""

# --- AUTOMATIZÁLT FRISSÍTŐ MOTOR (FULL PRODUCTION) ---
class AutoUpdater:
    def __init__(self):
        self.api_key_openai = os.getenv("OPENAI_API_KEY")
        self.api_key_pinecone = os.getenv("PINECONE_API_KEY")
        self.client_ai = OpenAI(api_key=self.api_key_openai)
        self.pc = Pinecone(api_key=self.api_key_pinecone)
        self.index = self.pc.Index(INDEX_NAME)

    def run_daily_update(self):
        print(f"🔄 [AUTO] FULL UPDATE Indítása: {XML_FEED_URL}")
        current_sync_ts = int(time.time())
        
        try:
            session = requests.Session()
            response = session.get(XML_FEED_URL, stream=True, timeout=120)
            
            if response.status_code != 200: 
                print("❌ Hiba: Nem érhető el az XML.")
                return

            response.raw.decode_content = True
            events = ET.iterparse(response.raw, events=("start", "end"))
            context = iter(events)
            event, root = next(context)

            print("🚀 [MODE] V45 Full Database Update (Clean Titles + Streaming)")
            
            batch = []
            count_total = 0
            count_uploaded = 0
            
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
                                
                                # --- FELDOLGOZÁS ---
                                structured_text = clean_html_structural(full_raw_text)
                                auth = extract_author(structured_text)
                                pub = extract_publisher(structured_text)
                                
                                # Fallback: Ha a szövegben benne van a Bookman, de nem ismerte fel a regex
                                if not pub and ("Bookman" in full_raw_text or "bookman" in full_raw_text):
                                    pub = "Bookman Kiadó"

                                # Hash ellenőrzés (hogy ne generáljunk feleslegesen, ha nem változott)
                                price = fuzzy_find(elem, ['price', 'g:price']) or "0"
                                sale = fuzzy_find(elem, ['sale_price', 'g:sale_price']) or ""
                                d_hash = generate_content_hash(f"{bid}{title}{pub}{price}{sale}")
                                
                                # FORCE UPDATE a javítás miatt
                                need_emb = True 

                                if need_emb:
                                    if count_total % 200 == 0: 
                                        print(f"⏳ [PROG] {count_total} feldolgozva... Utolsó: {title}")

                                    emb_text = f"Cím: {title}. Szerző: {auth}. Kiadó: {pub}. Leírás: {structured_text[:600]}"
                                    emb = self.client_ai.embeddings.create(input=emb_text[:8000], model="text-embedding-3-small").data[0].embedding
                                    
                                    meta = {
                                        "title": title, # Most már tiszta lesz!
                                        "url": fuzzy_find(elem, ['link', 'g:link']), 
                                        "image_url": fuzzy_find(elem, ['image_link', 'g:image_link']),
                                        "price": price, "lang": "hu", "stock": "instock", 
                                        "author": auth, "publisher": pub, 
                                        "full_search_text": f"{title} {auth} {pub}".lower(),
                                        "content_hash": d_hash, "last_seen": current_sync_ts
                                    }
                                    batch.append((bid, emb, meta))
                                    count_uploaded += 1

                        except Exception as e: pass
                        
                        elem.clear()
                        root.clear()
                        
                        if count_total % 200 == 0: gc.collect()
                        
                        if len(batch) >= 50:
                            self.index.upsert(vectors=batch)
                            batch = []

            if batch: self.index.upsert(vectors=batch)
            
            print("🧹 [AUTO] Takarítás...")
            try: self.index.delete(filter={"last_seen": {"$lt": current_sync_ts}, "type": {"$ne": "policy"}})
            except: pass
            
            print(f"🏁 [VÉGE] Teljes adatbázis frissítve! ({count_uploaded} elem)")

        except Exception as e:
            print(f"❌ Hiba: {e}")

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
def home(): return {"status": "Booksy V45 (Clean Titles)"}

@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url)

@app.post("/force-update")
def force(bt: BackgroundTasks):
    bt.add_task(bot.updater.run_daily_update)
    return {"status": "Full Clean Update Started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)