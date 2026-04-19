# BOOKSY BRAIN - V157 (FULL RESTORE - PART 1)
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
from email.utils import formatdate, make_msgid

import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
try:
    from moviepy.editor import ImageClip, concatenate_videoclips
    import moviepy.video.fx.all as vfx
    MOVIEPY_AVAILABLE = True
except:
    MOVIEPY_AVAILABLE = False

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
LOCAL_TZ = pytz.timezone('Europe/Bucharest')
XML_FEED_URL = "https://www.antikvarius.ro/wp-content/uploads/woo-feed/google/xml/booksyfullfeed.xml"
TEMP_FILE = "temp_feed.xml"

def normalize_text(text):
    if not text: return ""
    return ''.join(c for c in unicodedata.normalize('NFD', str(text).lower()) if unicodedata.category(c) != 'Mn')

def clean_price_raw(raw_price):
    if not raw_price: return "0 RON"
    cleaned_num = re.sub(r"[^\d.,]", "", str(raw_price).strip())
    return f"{cleaned_num} RON" if cleaned_num else str(raw_price).strip()

def html_to_markdown_clean(raw_html):
    if not raw_html: return ""
    return markdownify.markdownify(raw_html, heading_style="ATX", strip=['script', 'style']).strip()

class DBHandler:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./booksy_db")
        self.collection = self.client.get_or_create_collection(name="booksy_collection_gemini_v2")

db_handler = DBHandler()
class AutoUpdater:
    def __init__(self, db: DBHandler): self.db = db
    def download_feed(self):
        try:
            r = requests.get(XML_FEED_URL, stream=True, timeout=300)
            with open(TEMP_FILE, 'wb') as f:
                for chunk in r.iter_content(8192): f.write(chunk)
            return True
        except: return False

    def run_daily_update(self):
        print("🚀 Szinkronizálás indul...")
        if not self.download_feed(): return
        unique_books = {}
        for _, elem in ET.iterparse(TEMP_FILE, events=("end",)):
            if elem.tag.split('}')[-1].lower() in ['item', 'post']:
                d = {c.tag.split('}')[-1].lower(): (c.text or "") for c in elem}
                bid = d.get('id') or d.get('post_id')
                if bid:
                    unique_books[bid] = {
                        "id": bid, "title": d.get('title', 'Nincs cím'), "url": d.get('link', ''),
                        "image_url": d.get('image_link', ''), "price": clean_price_raw(d.get('sale_price') or d.get('price')),
                        "author": d.get('author', 'Ismeretlen'), "description": html_to_markdown_clean(d.get('description', '')),
                        "stock": "instock", "lang": "hu", "type": "book"
                    }
                elem.clear()
        
        ids, texts, metas = [], [], []
        for bid, b in unique_books.items():
            ids.append(bid)
            texts.append(f"Cím: {b['title']}. Szerző: {b['author']}. Leírás: {b['description'][:500]}")
            m = b.copy(); del m['description']; m['text_preview'] = b['description'][:150]
            metas.append(m)
            if len(ids) >= 100:
                res = gemini_client.models.embed_content(model="gemini-embedding-001", contents=texts, config=types.EmbedContentConfig(output_dimensionality=768))
                self.db.collection.upsert(ids=ids, embeddings=[e.values for e in res.embeddings], metadatas=metas)
                ids, texts, metas = [], [], []
                time.sleep(2.5)
        print("✅ Kész.")

class BooksyBrain:
    def __init__(self, db: DBHandler):
        self.db = db
        self.claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def process(self, msg, context_url, session_id):
        try:
            # Intent & Vector search
            vec = gemini_client.models.embed_content(model="gemini-embedding-001", contents=msg, config=types.EmbedContentConfig(output_dimensionality=768)).embeddings[0].values
            res = self.db.collection.query(query_embeddings=[vec], n_results=5, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
            
            ctx_text = ""
            products = []
            if res['ids'] and res['ids'][0]:
                for m in res['metadatas'][0]:
                    products.append({"title": m['title'], "price": m['price'], "url": m['url'], "image": m['image_url']})
                    ctx_text += f"- {m['title']} ({m['price']})\n"
            
            r = self.claude.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=1000,
                system="You are Booksy, the helpful Hungarian bookstore assistant. Recommend books warmly.",
                messages=[{"role": "user", "content": f"Books found:\n{ctx_text}\nUser: {msg}"}]
            )
            return {"reply": r.content[0].text, "products": products}
        except Exception as e:
            print(f"Chat hiba: {e}")
            return {"reply": "Sajnos hiba történt a keresésnél.", "products": []}
            class BooksySocialAgent:
    def __init__(self, db: DBHandler):
        self.db = db
        self.claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def run_night_generation(self):
        print("🕒 Social Agent indul...")
        # Fallback search for a book to post
        vec = gemini_client.models.embed_content(model="gemini-embedding-001", contents="érdekes antikvár könyv", config=types.EmbedContentConfig(output_dimensionality=768)).embeddings[0].values
        res = self.db.collection.query(query_embeddings=[vec], n_results=1, where={"type": "book"})
        if not res['ids'] or not res['ids'][0]: return
        
        target = res['metadatas'][0][0]
        post_text = self.claude.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1000,
            system="You are Booksy CopySEO expert. Write a compelling HU Facebook post.",
            messages=[{"role": "user", "content": f"Book: {target['title']} by {target.get('author')}. URL: {target['url']}"}]
        ).content[0].text
        
        with open("social_state.json", "w") as f: json.dump({"text": post_text}, f)
        print("✅ Social post vázlat kész.")

updater = AutoUpdater(db_handler)
bot = BooksyBrain(db_handler)
social_agent = BooksySocialAgent(db_handler)
scheduler = BackgroundScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.start(); yield; scheduler.shutdown()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_headers=["*"], allow_methods=["*"])

class ChatRequest(BaseModel): message: str; context_url: Optional[str] = ""; session_id: Optional[str] = ""
class InitRequest(BaseModel): url: str; session_id: str; ui_lang: str = "ro"

@app.get("/")
def home(): return {"status": "Booksy V157 Full Online"}
@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url, req.session_id)
@app.post("/init-chat")
def init_chat(req: InitRequest):
    return {"ui_lang": req.ui_lang, "bubble_text": "Szia! Miben segíthetek?", "placeholder": "Keresel valamit?"}
@app.post("/force-update")
def force_update(bt: BackgroundTasks): bt.add_task(updater.run_daily_update); return {"status": "Update Started"}
@app.post("/test-social-night")
def test_night(bt: BackgroundTasks): bt.add_task(social_agent.run_night_generation); return {"status": "Social Triggered"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)