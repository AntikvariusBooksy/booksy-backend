# BOOKSY BRAIN - V164 (FULL AGENTIC RESTORE - PART 1)
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
CLAUDE_MODEL = "claude-sonnet-4-6"

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

def extract_metadata_from_html(raw_html):
    meta = {"publisher": None, "author": None}
    if not raw_html: return meta
    try:
        soup = BeautifulSoup(raw_html, 'lxml')
        pub_label = soup.find(string=re.compile(r'(?:Kiadó|Publisher|Editura)\s*:', re.IGNORECASE))
        if pub_label and pub_label.find_parent('td'):
            next_td = pub_label.find_parent('td').find_next_sibling('td')
            if next_td: meta['publisher'] = next_td.get_text(strip=True)
        auth_label = soup.find(string=re.compile(r'(?:Szerző|Írta|Author|Autor)\s*:', re.IGNORECASE))
        if auth_label and auth_label.find_parent('td'):
            next_td = auth_label.find_parent('td').find_next_sibling('td')
            if next_td: meta['author'] = next_td.get_text(strip=True)
    except: pass
    return meta

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
        print("🚀 [FULL SYNC] Szinkronizálás indul...")
        if not self.download_feed(): return
        unique_books = {}
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
        
        ids, texts, metas = [], [], []
        for bid, b in unique_books.items():
            ids.append(bid)
            texts.append(f"Cím: {b['title']}. Szerző: {b['author']}. Leírás: {b['description'][:600]}")
            m = b.copy(); del m['description']; m['text_preview'] = b['description'][:150]
            metas.append(m)
            if len(ids) >= 100:
                try:
                    res = gemini_client.models.embed_content(model="gemini-embedding-001", contents=texts, config=types.EmbedContentConfig(output_dimensionality=768))
                    self.db.collection.upsert(ids=ids, embeddings=[e.values for e in res.embeddings], metadatas=metas)
                    time.sleep(2.5)
                except: time.sleep(10)
                ids, texts, metas = [], [], []
        print("✅ Szinkronizálás kész.")

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
                    ctx_text += f"- {m['title']} by {m.get('author', 'Unknown')} ({m['price']}). Kategória: {m.get('category', 'Antikvár')}\n"
            
            r = self.claude.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                system="You are the elegant Booksy bookstore assistant. Respond in Hungarian using CopySEO principles.",
                messages=[{"role": "user", "content": f"Context: {ctx_text}\nUser asks: {msg}"}]
            )
            return {"reply": r.content[0].text, "products": products}
        except Exception as e:
            return {"reply": f"Hiba: {e}", "products": []}
class BooksySocialAgent:
    def __init__(self, db: DBHandler):
        self.db = db
        self.claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _fetch_wikipedia_births(self):
        today = datetime.now(LOCAL_TZ)
        url = f"https://en.wikipedia.org/api/rest_v1/feed/onthisday/births/{today.strftime('%m/%d')}"
        try:
            resp = requests.get(url, headers={'User-Agent': 'BooksyBot/1.0'}, timeout=15)
            if resp.status_code == 200:
                verified = []
                for p in resp.json().get('births', []):
                    comb = (p.get('text', '') + " " + (p.get('pages', [{}])[0].get('extract', '') if p.get('pages') else "")).lower()
                    if any(kw in comb for kw in ['writer', 'author', 'poet', 'novelist']):
                        verified.append({"name": p.get('text', '').split(',')[0], "bio": p.get('pages', [{}])[0].get('extract', '') if p.get('pages') else p.get('text')})
                return verified
            return []
        except: return []

    def _create_infinite_loop_video(self, image_path, output_path):
        if not MOVIEPY_AVAILABLE: return False
        try:
            clip = ImageClip(image_path).resize(width=800)
            def zoom(t): return 1 + 0.02 * t
            zoomed = clip.resize(zoom).crop(x_center=clip.w/2, y_center=clip.h/2, width=clip.w, height=clip.h).set_duration(4)
            final = concatenate_videoclips([zoomed, zoomed.fx(vfx.time_mirror)])
            final.write_videofile(output_path, fps=15, codec="libx264", audio=False, logger=None, threads=1, preset="ultrafast")
            return True
        except: return False

    def run_night_generation(self):
        print(f"🕒 Social Agent indul ({CLAUDE_MODEL})...")
        try:
            wiki_writers = self._fetch_wikipedia_births()
            prompt = f"Today is {datetime.now(LOCAL_TZ).strftime('%B %d')}. Writers: {json.dumps(wiki_writers[:15])}. Select 1, HU name, HU bio sentence. Output ONLY JSON: {{\"name\": \"...\", \"bio\": \"...\"}}"
            r_wiki = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=200, messages=[{"role": "user", "content": prompt}])
            writer_data = json.loads(r_wiki.content[0].text)
            
            vec = gemini_client.models.embed_content(model="gemini-embedding-001", contents=writer_data['name'], config=types.EmbedContentConfig(output_dimensionality=768)).embeddings[0].values
            res = self.db.collection.query(query_embeddings=[vec], n_results=1, where={"type": "book"})
            
            if not res['ids'] or not res['ids'][0]: return
            target = res['metadatas'][0][0]
            
            # Kép generálás
            p_img = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=200, messages=[{"role": "user", "content": f"Write an artistic image prompt for: {target['title']}. NO TEXT."}]).content[0].text
            img_url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(p_img)}?width=1024&height=1024&nologo=true"
            img_path = "social_img.jpg"
            with open(img_path, 'wb') as f: f.write(requests.get(img_url).content)
            
            # Poszt megírása
            post_text = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=1000, system="You are Booksy CopySEO.", messages=[{"role": "user", "content": f"Write FB post about {target['title']} by {target.get('author')}. Link: {target['url']}"}]).content[0].text
            with open("social_state.json", "w") as f: json.dump({"text": post_text}, f)
            print("✅ Social draft kész.")
        except Exception as e: print(f"❌ Social Agent error: {e}")
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
def home(): return {"status": "Booksy V164 FULL AGENTIC ONLINE", "model": CLAUDE_MODEL}
@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url, req.session_id)
@app.post("/init-chat")
def init_chat(req: InitRequest):
    return {"ui_lang": req.ui_lang, "bubble_text": "Szia! Miben segíthetek?", "placeholder": "Keresel valamit?"}
@app.post("/force-update")
def force_update(bt: BackgroundTasks): bt.add_task(updater.run_daily_update); return {"status": "Started"}
@app.post("/test-social-night")
def test_night(bt: BackgroundTasks): bt.add_task(social_agent.run_night_generation); return {"status": "Social Triggered"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)