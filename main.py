# BOOKSY BRAIN - V168 (THE ULTIMATE CONSOLIDATED AGENTIC EDITION)
# VERZIÓ: V168 - DRAFT MODE RESTORED + CLAUDE 4.6 + FULL LOGIC

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
try:
    import PIL.Image
    if not hasattr(PIL.Image, 'ANTIALIAS'): 
        PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
    from moviepy.editor import ImageClip, concatenate_videoclips
    import moviepy.video.fx.all as vfx
    MOVIEPY_AVAILABLE = True
except Exception as e:
    print(f"MoviePy betöltési hiba (nem kritikus): {e}")
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
    try:
        return markdownify.markdownify(raw_html, heading_style="ATX", strip=['script', 'style']).strip()
    except:
        return str(raw_html)

def safe_json_parse(text):
    """Eltávolítja a Markdown jeleket és parse-olja a JSON-t."""
    try:
        clean_text = re.sub(r'```json\s*|\s*```', '', text).strip()
        return json.loads(clean_text)
    except Exception as e:
        print(f"JSON Parse hiba: {e} | Eredeti szöveg: {text[:100]}")
        return {}

def extract_metadata_from_html(raw_html):
    meta = {"publisher": "Ismeretlen", "author": "Ismeretlen"}
    if not raw_html: return meta
    try:
        soup = BeautifulSoup(raw_html, 'lxml')
        for label, key in [('(?:Kiadó|Editura|Publisher)', 'publisher'), ('(?:Szerző|Autor|Author)', 'author')]:
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

# --- SZOLGÁLTATÁSOK ---
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
        except Exception as e:
            print(f"Letöltési hiba: {e}")
            return False

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
                            "publisher": ext_meta['publisher'], "author": d.get('author') or ext_meta['author'],
                            "description": html_to_markdown_clean(raw_desc), "stock": "instock", "lang": "hu", "type": "book"
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
                    try:
                        res = gemini_client.models.embed_content(model="gemini-embedding-001", contents=texts, config=types.EmbedContentConfig(output_dimensionality=768))
                        self.db.collection.upsert(ids=ids, embeddings=[e.values for e in res.embeddings], metadatas=metas)
                        print(f"⏳ {i+1}/{total} vektorizálva...")
                        time.sleep(2.5) # Paid Tier RPM védelem
                    except Exception as e:
                        print(f"Batch hiba: {e}")
                        time.sleep(10)
                    ids, texts, metas = [], [], []
            
            if ids:
                res = gemini_client.models.embed_content(model="gemini-embedding-001", contents=texts, config=types.EmbedContentConfig(output_dimensionality=768))
                self.db.collection.upsert(ids=ids, embeddings=[e.values for e in res.embeddings], metadatas=metas)
            
            if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)
            print("✅ [SZINKRON] Kész.")
        except Exception as e: print(f"❌ SZINKRON HIBA: {e}")

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
                max_tokens=1024,
                system="You are Booksy, the elegant Hungarian antique bookstore assistant. Respond in Hungarian using CopySEO principles.",
                messages=[{"role": "user", "content": f"Context books:\n{ctx_text}\n\nUser Question: {msg}"}]
            )
            return {"reply": r.content[0].text, "products": products}
        except Exception as e:
            print(f"❌ CHAT HIBA: {e}")
            return {"reply": "Elnézést, egy kis technikai zavar támadt. Kérlek próbáld újra!", "products": []}

    def negotiate_handshake(self, ui_lang):
        return {"ui_lang": ui_lang, "bubble_text": "Szia! Miben segíthetek?", "placeholder": "Keresel valamit?"}

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

    def _create_video(self, img_path, out_path):
        if not MOVIEPY_AVAILABLE: return False
        try:
            clip = ImageClip(img_path).resize(width=1080)
            def zoom(t): return 1 + 0.03 * t
            zoomed = clip.resize(zoom).crop(x_center=clip.w/2, y_center=clip.h/2, width=clip.w, height=clip.h).set_duration(5)
            final = concatenate_videoclips([zoomed, zoomed.fx(vfx.time_mirror)])
            final.write_videofile(out_path, fps=24, codec="libx264", audio=False, logger=None, threads=2)
            return True
        except Exception as e:
            print(f"Videó hiba: {e}")
            return False

    def send_morning_email(self, post_text):
        try:
            sender, password = os.getenv("SMTP_SENDER"), os.getenv("SMTP_PASSWORD")
            admin_emails = [e.strip() for e in os.getenv("ADMIN_EMAIL", "").split(",") if e.strip()]
            if not sender or not admin_emails: return
            server = smtplib.SMTP(os.getenv("SMTP_SERVER", "mail.antikvarius.ro"), 26, timeout=20)
            server.starttls()
            server.login(sender, password)
            for admin in admin_emails:
                msg = MIMEMultipart()
                msg['Subject'] = f"✅ Booksy Social Draft ({datetime.now(LOCAL_TZ).strftime('%Y-%m-%d')})"
                msg.attach(MIMEText(f"<html><body><h2>Mai Facebook vázlat:</h2><pre>{post_text}</pre></body></html>", 'html'))
                server.send_message(msg)
            server.quit()
        except Exception as e: print(f"📧 Email küldési hiba: {e}")

    def run_night_generation(self):
        print(f"🕒 [SOCIAL] Agent indul ({CLAUDE_MODEL})...")
        try:
            writers = self._fetch_wikipedia_births()
            prompt = f"Today is {datetime.now(LOCAL_TZ).strftime('%B %d')}. Select 1 legendary writer from: {json.dumps(writers[:15])}. JSON ONLY: {{\"name\": \"...\", \"bio\": \"...\"}}"
            r_wiki = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=300, messages=[{"role": "user", "content": prompt}])
            w_data = safe_json_parse(r_wiki.content[0].text)
            
            # Keresés
            search_name = w_data.get('name', 'ritka antikvár könyv')
            vec = gemini_client.models.embed_content(model="gemini-embedding-001", contents=search_name, config=types.EmbedContentConfig(output_dimensionality=768)).embeddings[0].values
            res = self.db.collection.query(query_embeddings=[vec], n_results=1, where={"type": "book"})
            
            if not res['ids'] or not res['ids'][0]:
                print("⚠️ Fallback keresés indul...")
                vec = gemini_client.models.embed_content(model="gemini-embedding-001", contents="antikvár könyv különlegesség", config=types.EmbedContentConfig(output_dimensionality=768)).embeddings[0].values
                res = self.db.collection.query(query_embeddings=[vec], n_results=1, where={"type": "book"})
            
            target = res['metadatas'][0][0]
            print(f"📚 Kiválasztott könyv: {target['title']}")

            # Média generálás
            p_img = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=200, messages=[{"role": "user", "content": f"Artistic, moody library image prompt for: {target['title']}. High quality, NO TEXT, NO HUMAN FACES."}]).content[0].text
            img_path, vid_path = "social_img.jpg", "social_video.mp4"
            with open(img_path, 'wb') as f: f.write(requests.get(f"https://image.pollinations.ai/prompt/{urllib.parse.quote(p_img)}?width=1024&height=1024&nologo=true").content)
            
            has_video = self._create_video(img_path, vid_path)
            
            # Szövegírás
            post_text = self.claude.messages.create(
                model=CLAUDE_MODEL, 
                max_tokens=1500, 
                system="You are Booksy CopySEO expert. Write a professional, high-engagement Hungarian Facebook post.", 
                messages=[{"role": "user", "content": f"Write about {target['title']} by {target.get('author')}. Info: {target.get('text_preview')}. URL: {target['url']}"}]
            ).content[0].text
            
            # FACEBOOK DRAFT UPLOAD
            fb_id, fb_token = os.getenv("FB_PAGE_ID"), os.getenv("FB_PAGE_TOKEN")
            if fb_id and fb_token:
                if has_video:
                    fb_res = requests.post(
                        f"https://graph.facebook.com/v19.0/{fb_id}/videos", 
                        data={'access_token': fb_token, 'description': post_text, 'published': 'false'}, 
                        files={'source': open(vid_path, 'rb')}
                    )
                else:
                    fb_res = requests.post(
                        f"https://graph.facebook.com/v19.0/{fb_id}/photos", 
                        data={'access_token': fb_token, 'message': post_text, 'published': 'false', 'unpublished_content_type': 'DRAFT'}, 
                        files={'source': open(img_path, 'rb')}
                    )
                print(f"FB response: {fb_res.text}")
            
            self.send_morning_email(post_text)
            for p in [img_path, vid_path]:
                if os.path.exists(p): os.remove(p)
            print("✅ [SOCIAL] Vázlat sikeresen feltöltve.")
        except Exception as e: print(f"❌ SOCIAL ÜGYNÖK HIBA: {e}")

# --- FASTAPI SZERVER ---
updater = AutoUpdater(db_handler)
bot = BooksyBrain(db_handler)
social_agent = BooksySocialAgent(db_handler)
scheduler = BackgroundScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Időzített feladatok (Cron)
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
def home(): return {"status": "Booksy V168 Online", "model": CLAUDE_MODEL, "mode": "DRAFT"}

@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url, req.session_id)

@app.post("/init-chat")
def init_chat(req: InitRequest): return bot.negotiate_handshake(req.ui_lang)

@app.post("/force-update")
def force_update(bt: BackgroundTasks): bt.add_task(updater.run_daily_update); return {"status": "Update process started"}

@app.post("/test-social-night")
def test_night(bt: BackgroundTasks): bt.add_task(social_agent.run_night_generation); return {"status": "Social draft generation started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)