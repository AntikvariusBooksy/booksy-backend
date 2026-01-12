import os
import time
import requests
import hashlib
import re
import unicodedata
import html  # FONTOS: Ez kell a &lt; jelek dekódolásához!
import xml.etree.ElementTree as ET
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

class SmartHookRequest(BaseModel):
    current_url: str
    current_title: str
    visitor_type: str 
    cart_item_count: int
    history: List[dict] = [] 
    lang: str

# --- HELPEREK ---
def normalize_text(text):
    if not text: return ""
    text = str(text).lower()
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def generate_content_hash(data_string):
    return hashlib.md5(data_string.encode('utf-8')).hexdigest()

def clean_html_structural(raw_html):
    if not raw_html: return ""
    
    # 1. LÉPÉS: HTML Entity dekódolás (&lt; -> <)
    # Ez a legfontosabb javítás a te XML feededhez!
    s = html.unescape(str(raw_html))
    
    # 2. LÉPÉS: Struktúra megőrzése (blokk elemek -> sortörés)
    s = s.replace('</div>', '\n').replace('</p>', '\n').replace('<br>', '\n').replace('<br/>', '\n')
    s = s.replace('</h1>', '\n').replace('</h2>', '\n').replace('</h3>', '\n').replace('</h4>', '\n')
    s = s.replace('</li>', '\n')
    
    # 3. LÉPÉS: Tag-ek törlése
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', s)
    
    # 4. LÉPÉS: CDATA és felesleges szóközök takarítása
    cleantext = cleantext.replace("<![CDATA[", "").replace("]]>", "")
    
    # Soronkénti tisztítás, hogy szép tiszta szöveget kapjunk
    return "\n".join([line.strip() for line in cleantext.split('\n') if line.strip()])

def safe_str(val):
    return str(val).strip() if val is not None else ""

def extract_author(text_content):
    if not text_content: return ""
    match = re.search(r'(Szerző|Írta|Author|Szerzők)[:\s]+([^\n|<]+)', text_content, re.IGNORECASE)
    return match.group(2).strip() if match else ""

def extract_publisher(text_content):
    if not text_content: return ""
    
    # 1. Bookman prioritás (bárhol a szövegben)
    if "Bookman" in text_content: 
        return "Bookman Kiadó"
        
    # 2. Normál keresés
    match = re.search(r'(Kiadó|Kiadás|Publisher)[:\s]+([^\n|<]+)', text_content, re.IGNORECASE)
    if match:
        pub = match.group(2).strip()
        if len(pub) > 60: return pub[:60]
        return pub
    return ""

# --- NAMESPACE-FÜGGETLEN KERESŐ (V41 logika) ---
def fuzzy_find(item, tag_suffixes):
    """Megtalálja a g:id-t és az id-t is."""
    if isinstance(tag_suffixes, str): tag_suffixes = [tag_suffixes]
    
    for child in item:
        tag_name = child.tag.split('}')[-1].lower() if '}' in child.tag else child.tag.lower()
        for suffix in tag_suffixes:
            if tag_name == suffix.lower():
                return safe_str(child.text)
    return ""

# --- AUTOMATIZÁLT FRISSÍTŐ MOTOR ---
class AutoUpdater:
    def __init__(self):
        self.api_key_openai = os.getenv("OPENAI_API_KEY")
        self.api_key_pinecone = os.getenv("PINECONE_API_KEY")
        self.client_ai = OpenAI(api_key=self.api_key_openai)
        self.pc = Pinecone(api_key=self.api_key_pinecone)
        self.index = self.pc.Index(INDEX_NAME)

    def run_daily_update(self):
        print(f"🔄 [AUTO] Indítás: {XML_FEED_URL}")
        current_sync_ts = int(time.time())
        try:
            response = requests.get(XML_FEED_URL, stream=True, timeout=120)
            if response.status_code != 200: 
                print("❌ Hiba: Nem sikerült letölteni az XML-t.")
                return
            
            try: tree = ET.fromstring(response.content)
            except: tree = ET.fromstring(response.content.decode('utf-8', 'ignore'))

            all_items = tree.findall('.//item')
            if not all_items: all_items = tree.findall('.//post')
            
            print(f"📚 [AUTO] XML letöltve. Összes elem: {len(all_items)}")
            print("🎯 [MODE] V42: Namespace Fix + HTML Entity Decode")

            batch = []
            processed_bookman = 0
            processed_other = 0
            
            # --- TESZT: Minden 200. könyvet dolgozzuk fel + az összes Bookmant ---
            # Így gyorsan végigér, de reprezentatív lesz
            
            for index, item in enumerate(all_items):
                try:
                    # 1. ID Keresés (Fuzzy)
                    bid = fuzzy_find(item, ['id', 'g:id', 'post_id'])
                    if not bid: continue

                    # 2. Adatok kinyerése
                    title = fuzzy_find(item, ['title', 'g:title']) or "Nincs cím"
                    
                    # Leírás összerakása
                    desc = fuzzy_find(item, ['description', 'content', 'g:description'])
                    short_desc = fuzzy_find(item, ['shortdescription', 'excerpt'])
                    full_raw_text = f"{desc} {short_desc}"
                    
                    # 3. TISZTÍTÁS ÉS ADATKINYERÉS (V42 MAGIC)
                    # Először dekódoljuk a &lt; jeleket, aztán takarítunk
                    structured_text = clean_html_structural(full_raw_text)
                    
                    auth = extract_author(structured_text)
                    pub = extract_publisher(structured_text)
                    
                    # Ha üres a kiadó, és a szövegben van Bookman, akkor az!
                    if not pub and ("Bookman" in full_raw_text or "bookman" in full_raw_text):
                        pub = "Bookman Kiadó"

                    # SZŰRÉS: Csak Bookman VAGY minden 50. könyv (tesztnek)
                    is_bookman = "Bookman" in pub
                    should_process = is_bookman or (index % 50 == 0)
                    
                    if not should_process: continue

                    # További mezők
                    cat = fuzzy_find(item, ['product_type', 'category', 'g:product_type'])
                    url = fuzzy_find(item, ['link', 'g:link', 'permalink'])
                    img = fuzzy_find(item, ['image_link', 'g:image_link', 'imageurl'])
                    price = fuzzy_find(item, ['price', 'g:price'])
                    sale_price = fuzzy_find(item, ['sale_price', 'g:sale_price'])
                    reg = price if price else "0"
                    sale = sale_price if sale_price else ""

                    # Logolás
                    if is_bookman:
                        processed_bookman += 1
                        print(f"✅ [FOUND] Bookman: {title} (Kiadó: {pub})")
                    else:
                        processed_other += 1

                    # 4. EMBEDDING (Force Update)
                    d_hash = f"forced_v42_{current_sync_ts}"
                    
                    emb_text = f"Cím: {title}. Szerző: {auth}. Kiadó: {pub}. Kat: {cat}. Leírás: {structured_text[:600]}"
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
                        
                    if len(batch) >= 20: 
                        print(f"🚀 [UPLOAD] Batch feltöltése...")
                        self.index.upsert(vectors=batch)
                        batch = []
                        
                except Exception as e: 
                    print(f"⚠️ Hiba az elemnél: {e}")
                    continue

            if batch: 
                print(f"🚀 [UPLOAD] Utolsó batch feltöltése...")
                self.index.upsert(vectors=batch)
            
            print(f"🏁 [VÉGE] Bookman: {processed_bookman} | Egyéb (mintavétel): {processed_other}")

        except Exception as e: print(f"❌ Fatális Hiba: {e}")

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
                
                # Súlyozás
                pub_norm = normalize_text(meta.get('publisher', ''))
                auth_norm = normalize_text(meta.get('author', ''))
                title_norm = normalize_text(meta.get('title', ''))
                full_norm = normalize_text(meta.get('full_search_text', ''))

                if q_norm in pub_norm and len(q_norm) > 3: score += 1000 # Brutális boost
                if q_norm in auth_norm: score += 300
                if q_norm in title_norm: score += 200
                if q_norm in full_norm: score += 50

                m['custom_score'] = score
                results.append(m)
            
            results.sort(key=lambda x: x['custom_score'], reverse=True)
            return results[:10]
        except: return []

    def process(self, msg, context_url=""):
        site_lang = 'hu'
        if context_url and '/ro/' in str(context_url).lower(): site_lang = 'ro'
        matches = self.search(msg, site_lang)
        
        if not matches: return {"reply": "Sajnos nem találtam a keresésnek megfelelőt.", "products": []}
        
        prods = []
        ctx_text = ""
        for m in matches:
            meta = m['metadata']
            display = meta.get('title')
            if meta.get('publisher'): display += f" ({meta.get('publisher')})"
            p = {"title": display, "price": meta.get('price'), "url": meta.get('url'), "image": meta.get('image_url')}
            prods.append(p)
            ctx_text += f"- {display} (Szerző: {meta.get('author')}, Kiadó: {meta.get('publisher')})\n"
            if len(prods)>=8: break
            
        sys_prompt = f"User query: {msg}. Found:\n{ctx_text}\n\nTask: Recommend these. If Bookman was searched, highlight that these are from Bookman."
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
def home(): return {"status": "Booksy V42 (HTML Entity + Namespace Fix)"}

@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url)

@app.post("/force-update")
def force(bt: BackgroundTasks):
    bt.add_task(bot.updater.run_daily_update)
    return {"status": "V42 Update Started - Check logs!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)