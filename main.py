# BOOKSY BRAIN - V165 (ULTIMATE STABLE EDITION - FULL CONSOLIDATED CODE)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os, time, requests, hashlib, re, json, random, unicodedata, html, urllib.parse, gc, chromadb, pytz, smtplib
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google import genai
from google.genai import types
import anthropic 
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from typing import List, Optional, Dict, Any
from datetime import datetime
from bs4 import BeautifulSoup
import markdownify
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- KÖRNYEZETI BEÁLLÍTÁSOK ---
load_dotenv()
LOCAL_TZ = pytz.timezone('Europe/Bucharest')
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
CLAUDE_MODEL = "claude-sonnet-4-6"
XML_FEED_URL = "https://www.antikvarius.ro/wp-content/uploads/woo-feed/google/xml/booksyfullfeed.xml"
TEMP_FILE = "temp_feed.xml"

# --- MOVIEPY COMPAT ---
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
try:
    from moviepy.editor import ImageClip, concatenate_videoclips
    import moviepy.video.fx.all as vfx
    MOVIEPY_AVAILABLE = True
except:
    MOVIEPY_AVAILABLE = False

# --- SEGÉDFÜGGVÉNYEK ---
def normalize_text(text):
    if not text: return ""
    return ''.join(c for c in unicodedata.normalize('NFD', str(text).lower()) if unicodedata.category(c) != 'Mn')

def clean_price_raw(raw_price):
    if not raw_price: return "0 RON"
    cleaned_num = re.sub(r"[^\d.,]", "", str(raw_price).strip())
    return f"{cleaned_num} RON" if cleaned_num else str(raw_price).strip()

def html_to_markdown_clean(raw_html):
    if not raw_html: return ""
    try: return markdownify.markdownify(raw_html, heading_style="ATX", strip=['script', 'style']).strip()
    except: return str(raw_html)

def safe_json_parse(text):
    """Kitisztítja a Claude által esetleg bennhagyott Markdown jeleket a JSON-ból."""
    try:
        clean_text = re.sub(r'```json\s*|\s*```', '', text).strip()
        return json.loads(clean_text)
    except:
        return {}

def extract_metadata_from_html(raw_html):
    meta = {"publisher": None, "author": None}
    if not raw_html: return meta
    try:
        soup = BeautifulSoup(raw_html, 'lxml')
        for label, key in [('(?:Kiadó|Editura)', 'publisher'), ('(?:Szerző|Autor)', 'author')]:
            target = soup.find(string=re.compile(label + r'\s*:', re.IGNORECASE))
            if target and target.find_parent('td'):
                next_td = target.find_parent('td').find_next_sibling('td')
                if next_td: meta[key] = next_td.get_text(strip=True)
    except: pass
    return meta

# --- ADATBÁZIS KEZELÉS ---
class DBHandler:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./booksy_db")
        self.collection = self.client.get_or_create_collection(name="booksy_collection_gemini_v2")

db_handler = DBHandler()

# --- AUTOMATA FRISSÍTŐ ---
class AutoUpdater:
    def __init__(self, db: DBHandler):
        self.db = db

    def download_feed(self):
        try:
            r = requests.get(XML_FEED_URL, stream=True, timeout=300)
            r.raise_for_status()
            with open(TEMP_FILE, 'wb') as f:
                for chunk in r.iter_content(8192): f.write(chunk)
            return True
        except: return False

    def run_daily_update(self):
        print("🚀 [SZINKRON] Indítás...")
        if not self.download_feed(): return
        unique_books = {}
        try:
            for _, elem in ET.iterparse(TEMP_FILE, events=("end",)):
                if elem.tag.split('}')[-1].lower() in ['item', 'post']:
                    d = {c.tag.split('}')[-1].lower(): (c.text or "") for c in elem}
                    bid = d.get('id') or d.get('post_id')
                    if bid:
                        raw_desc = f"{d.get('description', '')} {d.get('shortdescription', '')}"
                        ext_meta = extract_metadata_from_html(raw_desc)
                        unique_books[bid] = {
                            "id": bid, "title": d.get('title', 'Nincs cím'), "url": d.get('link', ''),
                            "image_url": d.get('image_link', ''), "price": clean_price_raw(d.get('sale_price') or d.get('price')),
                            "publisher": ext_meta['publisher'] or "Ismeretlen",
                            "author": ext_meta['author'] or d.get('author') or "Ismeretlen",
                            "description": html_to_markdown_clean(raw_desc),
                            "stock": "instock", "lang": "hu", "type": "book"
                        }
                    elem.clear()
            
            total = len(unique_books)
            ids, texts, metas = [], [], []
            for i, (bid, b) in enumerate(unique_books.items()):
                ids.append(bid)
                texts.append(f"Cím: {b['title']}. Szerző: {b['author']}. Leírás: {b['description'][:600]}")
                m = b.copy(); del m['description']; m['text_preview'] = b['description'][:150]
                metas.append(m)
                if len(ids) >= 100:
                    res = gemini_client.models.embed_content(model="gemini-embedding-001", contents=texts, config=types.EmbedContentConfig(output_dimensionality=768))
                    self.db.collection.upsert(ids=ids, embeddings=[e.values for e in res.embeddings], metadatas=metas)
                    ids, texts, metas = [], [], []
                    print(f"⏳ {i+1}/{total} kész...")
                    time.sleep(2.5)
            if ids:
                res = gemini_client.models.embed_content(model="gemini-embedding-001", contents=texts, config=types.EmbedContentConfig(output_dimensionality=768))
                self.db.collection.upsert(ids=ids, embeddings=[e.values for e in res.embeddings], metadatas=metas)
            print("✅ [SZINKRON] Sikeresen befejeződött.")
        except Exception as e: print(f"❌ SZINKRON HIBA: {e}")

# --- CHAT AGY ---
class BooksyBrain:
    def __init__(self, db: DBHandler):
        self.db = db
        self.claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def process(self, msg, context_url, session_id):
        try:
            vec = gemini_client.models.embed_content(model="gemini-embedding-001", contents=msg, config=types.EmbedContentConfig(output_dimensionality=768)).embeddings[0].values
            res = self.db.collection.query(query_embeddings=[vec], n_results=5, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
            ctx_text = ""
            products = []
            if res['ids'] and res['ids'][0]:
                for m in res['metadatas'][0]:
                    products.append({"title": m['title'], "price": m['price'], "url": m['url'], "image": m['image_url']})
                    ctx_text += f"- {m['title']} by {m.get('author', 'Ismeretlen')} ({m['price']}).\n"
            
            r = self.claude.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1000,
                system="You are Booksy, the elegant Hungarian antique bookstore assistant. Use CopySEO style. Recommend books warmly.",
                messages=[{"role": "user", "content": f"Found books:\n{ctx_text}\nUser asks: {msg}"}]
            )
            return {"reply": r.content[0].text, "products": products}
        except Exception as e:
            print(f"❌ CHAT HIBA: {e}")
            return {"reply": "Elnézést, technikai hiba történt a keresésnél.", "products": []}

    def negotiate_handshake(self, ui_lang):
        return {"ui_lang": ui_lang, "bubble_text": "Szia! Miben segíthetek?", "placeholder": "Keresel valamit?"}

# --- SOCIAL AGENT ---
class BooksySocialAgent:
    def __init__(self, db: DBHandler):
        self.db = db
        self.claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _fetch_wikipedia_births(self):
        today = datetime.now(LOCAL_TZ).strftime('%m/%d')
        url = f"https://en.wikipedia.org/api/rest_v1/feed/onthisday/births/{today}"
        try:
            r = requests.get(url, headers={'User-Agent': 'BooksyBot/1.0'}, timeout=15)
            if r.status_code == 200:
                verified = []
                for p in r.json().get('births', []):
                    comb = (p.get('text', '') + " " + (p.get('pages', [{}])[0].get('extract', '') if p.get('pages') else "")).lower()
                    if any(kw in comb for kw in ['writer', 'author', 'poet', 'novelist']):
                        verified.append({"name": p.get('text', '').split(',')[0], "bio": p.get('pages', [{}])[0].get('extract', '') if p.get('pages') else p.get('text')})
                return verified
            return []
        except: return []

    def run_night_generation(self):
        print(f"🕒 [SOCIAL] Agent indul ({CLAUDE_MODEL})...")
        try:
            writers = self._fetch_wikipedia_births()
            prompt = f"Today is {datetime.now(LOCAL_TZ).strftime('%B %d')}. Select 1 writer from: {json.dumps(writers[:15])}. Output ONLY JSON: {{\"name\": \"...\", \"bio\": \"...\"}}"
            r_wiki = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=300, messages=[{"role": "user", "content": prompt}])
            writer_data = safe_json_parse(r_wiki.content[0].text)
            
            search_name = writer_data.get('name', 'ritka antikvár könyv')
            vec = gemini_client.models.embed_content(model="gemini-embedding-001", contents=search_name, config=types.EmbedContentConfig(output_dimensionality=768)).embeddings[0].values
            res = self.db.collection.query(query_embeddings=[vec], n_results=1, where={"type": "book"})
            
            if not res['ids'] or not res['ids'][0]: return
            target = res['metadatas'][0][0]
            
            # Kép generálás
            p_img = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=200, messages=[{"role": "user", "content": f"Artistic image prompt for: {target['title']}. NO TEXT."}]).content[0].text
            img_url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(p_img)}?width=1024&height=1024&nologo=true"
            img_path = "social_img.jpg"
            with open(img_path, 'wb') as f: f.write(requests.get(img_url).content)
            
            # Poszt szöveg
            post_text = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=1000, system="You are Booksy CopySEO expert.", messages=[{"role": "user", "content": f"Write FB post in HU about {target['title']} by {target.get('author')}. URL: {target['url']}"}]).content[0].text
            with open("social_state.json", "w") as f: json.dump({"text": post_text}, f)
            
            # FB feltöltés (DRAFT)
            fb_id, fb_token = os.getenv("FB_PAGE_ID"), os.getenv("FB_PAGE_TOKEN")
            if fb_id and fb_token:
                requests.post(f"https://graph.facebook.com/v19.0/{fb_id}/photos", data={"message": post_text, "published": False, "unpublished_content_type": "DRAFT", "access_token": fb_token}, files={'source': open(img_path, 'rb')})
            
            if os.path.exists(img_path): os.remove(img_path)
            print("✅ [SOCIAL] Kész.")
        except Exception as e: print(f"❌ SOCIAL HIBA: {e}")

# --- API SZERVER ---
updater = AutoUpdater(db_handler)
bot = BooksyBrain(db_handler)
social_agent = BooksySocialAgent(db_handler)
scheduler = BackgroundScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(updater.run_daily_update, CronTrigger(hour=3, minute=0, timezone=LOCAL_TZ))
    scheduler.add_job(social_agent.run_night_generation, CronTrigger(hour=4, minute=0, timezone=LOCAL_TZ))
    scheduler.start()
    yield
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_headers=["*"], allow_methods=["*"])

class ChatRequest(BaseModel): message: str; context_url: Optional[str] = ""; session_id: Optional[str] = ""
class InitRequest(BaseModel): url: str; session_id: str; ui_lang: str = "ro"

@app.get("/")
def home(): return {"status": "Booksy V165 Online", "claude": CLAUDE_MODEL}

@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url, req.session_id)

@app.post("/init-chat")
def init_chat(req: InitRequest): return bot.negotiate_handshake(req.ui_lang)

@app.post("/force-update")
def force_update(bt: BackgroundTasks): bt.add_task(updater.run_daily_update); return {"status": "Update started"}

@app.post("/test-social-night")
def test_night(bt: BackgroundTasks): bt.add_task(social_agent.run_night_generation); return {"status": "Social triggered"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)