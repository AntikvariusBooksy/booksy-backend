import os
import time
import requests
import hashlib
import re
import unicodedata
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from bs4 import BeautifulSoup
from apscheduler.schedulers.background import BackgroundScheduler
from typing import List, Optional

# --- KONFIGURÁCIÓ ---
load_dotenv()
INDEX_NAME = "booksy-index"
XML_FEED_URL = os.getenv("XML_FEED_URL", "https://www.antikvarius.ro/wp-content/uploads/woo-feed/google/xml/booksyfullfeed.xml")

POLICY_URLS = {
    "KAPCSOLAT": "https://www.antikvarius.ro/contact/",
    "FIZETÉS": "https://www.antikvarius.ro/hu/fizetesi-informaciok/",
    "SZÁLLÍTÁS": "https://www.antikvarius.ro/hu/szallitasi-informaciok/",
    "ÁSZF": "https://www.antikvarius.ro/hu/altalanos-szerzodesi-es-felhasznalasi-feltetelek/"
}

# --- ADATMODELLEK ---
class ChatRequest(BaseModel):
    message: str
    context_url: Optional[str] = "" 

class VisitEvent(BaseModel):
    url: str
    title: str
    time_spent: int 

class SmartHookRequest(BaseModel):
    current_url: str
    current_title: str
    visitor_type: str 
    cart_item_count: int
    history: List[VisitEvent] = [] 
    lang: str

# --- SEGÉDFÜGGVÉNYEK ---
def normalize_text(text):
    if not text: return ""
    text = str(text).lower()
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def generate_content_hash(data_string):
    return hashlib.md5(data_string.encode('utf-8')).hexdigest()

def clean_html_structural(raw_html):
    if not raw_html: return ""
    s = str(raw_html)
    s = s.replace('</div>', '\n').replace('</p>', '\n').replace('<br>', '\n').replace('<br/>', '\n')
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', s)
    cleantext = cleantext.replace("<![CDATA[", "").replace("]]>", "")
    return "\n".join([line.strip() for line in cleantext.split('\n') if line.strip()])

def safe_str(val):
    return str(val).strip() if val is not None else ""

def extract_author(text_content):
    if not text_content: return ""
    match = re.search(r'(Szerző|Írta|Author|Szerzők)[:\s]+([^\n]+)', text_content, re.IGNORECASE)
    return match.group(2).strip() if match else ""

def extract_publisher(text_content):
    if not text_content: return ""
    if "Bookman" in text_content:
        return "Bookman Kiadó"
    match = re.search(r'(Kiadó|Kiadás|Publisher)[:\s]+([^\n]+)', text_content, re.IGNORECASE)
    if match:
        pub = match.group(2).strip()
        if len(pub) > 60: return pub[:60]
        return pub
    return ""

# --- AUTOMATIZÁLT FRISSÍTŐ MOTOR ---
class AutoUpdater:
    def __init__(self):
        self.api_key_openai = os.getenv("OPENAI_API_KEY")
        self.api_key_pinecone = os.getenv("PINECONE_API_KEY")
        self.client_ai = OpenAI(api_key=self.api_key_openai)
        self.pc = Pinecone(api_key=self.api_key_pinecone)
        self.index = self.pc.Index(INDEX_NAME)

    def scrape_policy(self):
        pass 

    def update_books_from_feed(self):
        print(f"🔄 [AUTO] Könyv szinkronizáció: {XML_FEED_URL}")
        current_sync_ts = int(time.time())
        try:
            response = requests.get(XML_FEED_URL, stream=True, timeout=120)
            if response.status_code != 200: return
            try: tree = ET.fromstring(response.content)
            except: tree = ET.fromstring(response.content.decode('utf-8', 'ignore'))

            items = tree.findall('.//item')
            if not items: items = tree.findall('.//post')
            total_items = len(items)
            print(f"📚 [AUTO] Elemek száma: {total_items}")
            
            batch = []
            ns = {'g': 'http://base.google.com/ns/1.0'}
            processed_count = 0 # SZÁMLÁLÓ
            
            for item in items:
                processed_count += 1
                try:
                    id_node = item.find('g:id', ns) or item.find('ID')
                    if not id_node: continue
                    bid = safe_str(id_node.text)
                    
                    title_node = item.find('g:title', ns) or item.find('Title')
                    title = safe_str(title_node.text) if title_node else "Nincs cím"

                    desc_node = item.find('g:description', ns) or item.find('Content')
                    raw_desc = safe_str(desc_node.text) if desc_node else ""
                    
                    short_desc_node = item.find('ShortDescription')
                    if short_desc_node and short_desc_node.text:
                        raw_desc = safe_str(short_desc_node.text)

                    structured_text = clean_html_structural(raw_desc)
                    auth = extract_author(structured_text)
                    pub = extract_publisher(structured_text)

                    cat_node = item.find('g:product_type', ns) or item.find('Productcategories')
                    cat = safe_str(cat_node.text) if cat_node else ""
                    
                    url = safe_str((item.find('g:link', ns) or item.find('Link') or item.find('Permalink')).text)
                    img = safe_str((item.find('g:image_link', ns) or item.find('ImageURL')).text)

                    price_node = item.find('g:price', ns) or item.find('Price')
                    sale_node = item.find('g:sale_price', ns) or item.find('SalePrice')
                    reg = safe_str(price_node.text) if price_node else "0"
                    sale = safe_str(sale_node.text) if sale_node else ""
                    
                    d_hash = generate_content_hash(f"{bid}{title}{pub}{reg}{sale}")

                    need_emb = True
                    try:
                        ex = self.index.fetch(ids=[bid])
                        if ex and 'vectors' in ex and bid in ex['vectors']:
                            if ex['vectors'][bid]['metadata'].get('content_hash') == d_hash:
                                need_emb = False
                    except: pass

                    if need_emb:
                        # LOGOLÁS: Minden embedding generálás előtt jelezzen, ha sok van
                        # Csak minden 50.-nél írjunk, hogy ne follyon a log, de lássuk a haladást
                        if processed_count % 50 == 0:
                            print(f"⏳ [PROGRESS] Feldolgozás... {processed_count} / {total_items} (Embedding generálás)")

                        emb_text = f"Könyv címe: {title}. Szerző: {auth}. Kiadó: {pub}. Kategória: {cat}. Leírás: {structured_text[:500]}"
                        emb = self.client_ai.embeddings.create(input=emb_text[:8000], model="text-embedding-3-small").data[0].embedding
                        
                        meta = {
                            "title": title, "price": reg, "sale_price": sale, "url": url, "image_url": img, 
                            "lang": "hu", "stock": "instock", 
                            "author": auth, "publisher": pub, 
                            "category": cat,
                            "short_desc": structured_text[:300], 
                            "full_search_text": f"{title} {auth} {pub} {cat}".lower(),
                            "content_hash": d_hash, "last_seen": current_sync_ts
                        }
                        batch.append((bid, emb, meta))
                        
                    if len(batch) >= 50: 
                        print(f"🚀 [UPLOAD] 50 db könyv feltöltése Pinecone-ba... (Eddig összesen: {processed_count})")
                        self.index.upsert(vectors=batch)
                        batch = []
                        
                except Exception as e: 
                    print(f"⚠️ Hiba a(z) {processed_count}. elemnél: {e}")
                    continue

            if batch: 
                print(f"🚀 [UPLOAD] Maradék {len(batch)} db feltöltése...")
                self.index.upsert(vectors=batch)
            
            print("🧹 [AUTO] Takarítás (Mirror Sync)...")
            try: self.index.delete(filter={"last_seen": {"$lt": current_sync_ts}, "type": {"$ne": "policy"}})
            except: pass
            
            print("✅ [DONE] Teljes szinkronizáció kész!")

        except Exception as e: print(f"❌ Sync Error: {e}")

    def run_daily_update(self):
        self.update_books_from_feed()

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
            
            res = self.index.query(vector=vec, top_k=60, include_metadata=True, filter=filt)
            
            q_norm = normalize_text(q)
            results = []
            
            for m in res['matches']:
                meta = m['metadata']
                score = m['score'] * 100 
                
                pub_norm = normalize_text(meta.get('publisher', ''))
                auth_norm = normalize_text(meta.get('author', ''))
                title_norm = normalize_text(meta.get('title', ''))
                full_norm = normalize_text(meta.get('full_search_text', ''))

                if q_norm in pub_norm and len(q_norm) > 3:
                    score += 500 
                if q_norm in auth_norm:
                    score += 300
                if q_norm in title_norm:
                    score += 200
                if q_norm in full_norm:
                    score += 50

                m['custom_score'] = score
                results.append(m)
            
            results.sort(key=lambda x: x['custom_score'], reverse=True)
            return results[:10]
            
        except: return []

    def process(self, msg, context_url=""):
        site_lang = 'hu'
        if context_url and '/ro/' in str(context_url).lower(): site_lang = 'ro'
        
        matches = self.search(msg, site_lang)
        
        if not matches:
            return {"reply": "Sajnos nem találtam a keresésednek megfelelő könyvet.", "products": []}
            
        prods = []
        ctx_text = ""
        for m in matches:
            meta = m['metadata']
            display = meta.get('title')
            if meta.get('publisher'):
                display += f" ({meta.get('publisher')})"
                
            p = {"title": display, "price": meta.get('price'), "url": meta.get('url'), "image": meta.get('image_url')}
            prods.append(p)
            ctx_text += f"- {display} (Szerző: {meta.get('author')}, Ár: {meta.get('price')})\n"
            if len(prods)>=8: break
            
        sys_prompt = f"User searched for: {msg}. Found these books:\n{ctx_text}\n\nTask: Briefly recommend them. Mention 'Bookman' if user searched for it."
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
def home(): return {"status": "Booksy V38 (Verbose Logs)"}

@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url)

@app.post("/force-update")
def force(bt: BackgroundTasks):
    bt.add_task(bot.updater.run_daily_update)
    return {"status": "Update Started. Watch logs for progress!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)