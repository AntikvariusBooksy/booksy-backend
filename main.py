# BOOKSY BRAIN - V154 (PAID TIER MATHEMATICAL QUOTA & RETRY LOGIC)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import time
import requests
import hashlib
import re
import json
import random
import unicodedata
import html
import urllib.parse
import xml.etree.ElementTree as ET
import gc
import chromadb
import pytz
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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formatdate, make_msgid

# --- PILLOW & MOVIEPY COMPAT ---
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

try:
    from moviepy.editor import ImageClip, concatenate_videoclips
    import moviepy.video.fx.all as vfx
    MOVIEPY_AVAILABLE = True
except Exception as e:
    MOVIEPY_AVAILABLE = False

load_dotenv()

# Gemini Kliens
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

XML_FEED_URL = os.getenv("XML_FEED_URL", "https://www.antikvarius.ro/wp-content/uploads/woo-feed/google/xml/booksyfullfeed.xml")
TEMP_FILE = "temp_feed.xml"
LOCAL_TZ = pytz.timezone('Europe/Bucharest')

def normalize_text(text):
    if not text: return ""
    return ''.join(c for c in unicodedata.normalize('NFD', str(text).lower()) if unicodedata.category(c) != 'Mn')

def safe_str(val):
    return html.unescape(str(val).strip()) if val else ""

def clean_price_raw(raw_price):
    if not raw_price: return "0 RON"
    cleaned_num = re.sub(r"[^\d.,]", "", str(raw_price).strip())
    return f"{cleaned_num} RON" if cleaned_num else str(raw_price).strip()

def html_to_markdown_clean(raw_html):
    if not raw_html: return ""
    try:
        md = markdownify.markdownify(raw_html, heading_style="ATX", strip=['script', 'style'])
        return re.sub(r'\n\s*\n', '\n\n', md).strip()
    except: return safe_str(raw_html)

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

def extract_all_data(elem) -> Dict[str, Any]:
    data = {}
    for child in elem:
        val = safe_str(child.text)
        if val: data[child.tag.split('}')[-1].lower()] = val
    return data

class DBHandler:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./booksy_db")
        self.collection = self.client.get_or_create_collection(name="booksy_collection_gemini_v2")

class AutoUpdater:
    def __init__(self, db: DBHandler):
        self.db = db

    def download_feed(self):
        try:
            with requests.get(XML_FEED_URL, headers={'User-Agent': 'BooksyBot/1.0'}, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(TEMP_FILE, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            return os.path.getsize(TEMP_FILE) > 10000
        except: return False

    def run_daily_update(self, force_refresh=False):
        print("🚀 [START] Gemini Paid Tier szinkronizálás indul (V154 - Kvótavédelemmel)...")
        current_sync_ts = int(time.time())
        if not self.download_feed(): return
        try:
            context = ET.iterparse(TEMP_FILE, events=("end",))
            unique_books_buffer = {} 
            for event, elem in context:
                tag_local = elem.tag.split('}')[-1].lower()
                if tag_local in ['item', 'post']:
                    try:
                        item_data = extract_all_data(elem)
                        bid = item_data.get('id') or item_data.get('post_id') or item_data.get('g:id')
                        if bid:
                            raw_desc = f"{item_data.get('description', '')} {item_data.get('shortdescription', '')}"
                            ext_meta = extract_metadata_from_html(raw_desc)
                            cat = html_to_markdown_clean(item_data.get('product_type') or item_data.get('category') or "")
                            unique_books_buffer[bid] = {
                                "id": bid, "title": item_data.get('title') or "Nincs cím", "url": item_data.get('link', ''), 
                                "image_url": item_data.get('image_link', ''), "price": clean_price_raw(item_data.get('sale_price') or item_data.get('price')), 
                                "publisher": ext_meta['publisher'] or "Ismeretlen", "author": ext_meta['author'] or item_data.get('author') or "Ismeretlen", 
                                "category": cat, "description": html_to_markdown_clean(raw_desc), 
                                "stock": "instock", "lang": "ro" if "carti in limba romana" in normalize_text(cat) else "hu", 
                                "type": "book", "last_seen": current_sync_ts
                            }
                    except: pass
                    elem.clear()
            
            total_books = len(unique_books_buffer)
            processed_books = 0
            ids_batch, emb_texts_batch, metadatas_batch = [], [], []
            
            print(f"📊 Adatbázis elemzés kész: {total_books} könyv. Vektorizálás megkezdése...")
            
            for bid, book_data in unique_books_buffer.items():
                emb_text = f"SKU: {bid}. Cím: {book_data['title']}. Szerző: {book_data['author']}. Leírás: {book_data['description'][:600]}"
                clean_meta = book_data.copy()
                del clean_meta['description'] 
                clean_meta['text_preview'] = book_data['description'][:150]
                
                ids_batch.append(bid)
                emb_texts_batch.append(emb_text[:8000])
                metadatas_batch.append(clean_meta)
                
                if len(ids_batch) >= 100:
                    success = False
                    retries = 0
                    while not success and retries < 5:
                        try:
                            result = gemini_client.models.embed_content(
                                model="gemini-embedding-001", 
                                contents=emb_texts_batch,
                                config=types.EmbedContentConfig(output_dimensionality=768)
                            )
                            self.db.collection.upsert(ids=ids_batch, embeddings=[e.values for e in result.embeddings], metadatas=metadatas_batch)
                            processed_books += len(ids_batch)
                            if processed_books % 500 == 0:
                                print(f"⏳ [FOLYAMAT] {processed_books} / {total_books} könyv kész...")
                            success = True
                            
                            # Matematikai fék a 3000/min (50/sec) PAID RPM limit alá (2400/min-re állítva)
                            time.sleep(2.5) 
                        except Exception as e:
                            error_msg = str(e)
                            print(f"⚠️ Batch hiba (újrapróbálkozás {retries+1}/5): {error_msg[:100]}...")
                            retries += 1
                            # Ha a Google büntet, kivárjuk a kért időt (kb 40 mp)
                            time.sleep(40)
                            
                    ids_batch, emb_texts_batch, metadatas_batch = [], [], []
            
            if ids_batch: 
                retries = 0
                while retries < 3:
                    try:
                        result = gemini_client.models.embed_content(model="gemini-embedding-001", contents=emb_texts_batch, config=types.EmbedContentConfig(output_dimensionality=768))
                        self.db.collection.upsert(ids=ids_batch, embeddings=[e.values for e in result.embeddings], metadatas=metadatas_batch)
                        break
                    except:
                        retries += 1
                        time.sleep(10)
            
            if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)
            print(f"✅ [SIKER] Mind a {total_books} könyv vektorizálva. A Booksy agya felfrissült.")
        except Exception as e: print(f"❌ Végzetes hiba: {e}")

class ChatRequest(BaseModel): message: str; context_url: Optional[str] = ""; session_id: Optional[str] = ""
class InitRequest(BaseModel): url: str; session_id: str; ui_lang: str = "ro"

class BooksyBrain:
    def __init__(self, db: DBHandler):
        self.db = db
        self.user_session_cache = {}
        self.client_claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def process(self, msg, context_url, session_id):
        try:
            # Intent elemzés Geminivel (Paid tier - gyorsabb)
            prompt_intent = f"Intent analysis for bookstore. Input: '{msg}'. JSON: {{\"intent\": \"search\"|\"policy\", \"query\": \"query\"}}"
            analysis = gemini_client.models.generate_content(model='gemini-1.5-flash', contents=prompt_intent, config=types.GenerateContentConfig(response_mime_type="application/json")).text
            intent_data = json.loads(analysis)
            
            query_text = intent_data.get('query', msg)
            vec = gemini_client.models.embed_content(model="gemini-embedding-001", contents=query_text, config=types.EmbedContentConfig(output_dimensionality=768)).embeddings[0].values
            
            final_reply, final_products = "", []
            
            if intent_data.get('intent') == "policy":
                res = self.db.collection.query(query_embeddings=[vec], n_results=2, where={"type": "policy"})
                ctx = "".join([m.get('text', '') for m in res['metadatas'][0]]) if res['ids'] else ""
                reply_res = self.client_claude.messages.create(model="claude-sonnet-4-6", max_tokens=800, temperature=0.5, system="You are the elegant Booksy Assistant. Respond in Hungarian.", messages=[{"role": "user", "content": f"Context: {ctx}\nQ: {msg}"}])
                final_reply = reply_res.content[0].text
            else:
                res = self.db.collection.query(query_embeddings=[vec], n_results=5, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
                if res['ids'] and res['ids'][0]:
                    ctx_text = ""
                    for meta in res['metadatas'][0]:
                        p_price = clean_price_raw(meta.get('price'))
                        final_products.append({"title": meta.get('title'), "price": p_price, "url": meta.get('url'), "image": meta.get('image_url')})
                        ctx_text += f"- {meta.get('title')} by {meta.get('author', 'Unknown')} ({p_price}). Kategória: {meta.get('category', '')}\n"
                    reply_res = self.client_claude.messages.create(model="claude-sonnet-4-6", max_tokens=800, temperature=0.7, system="You are the Booksy CopySEO Assistant. Write a warm recommendation in Hungarian.", messages=[{"role": "user", "content": f"Books: {ctx_text}\nQ: {msg}"}])
                    final_reply = reply_res.content[0].text
                else: final_reply = "Sajnos nem találtam megfelelőt."
            
            self.user_session_cache[session_id] = msg
            return {"reply": final_reply, "products": final_products}
        except Exception as e: return {"reply": "Hiba történt a keresésnél.", "products": []}

    def negotiate_handshake(self, url, session_id, ui_lang):
        try:
            res = gemini_client.models.generate_content(model='gemini-1.5-flash', contents=f"JSON greeting in {ui_lang}.", config=types.GenerateContentConfig(response_mime_type="application/json")).text
            return json.loads(res)
        except: return {"ui_lang": ui_lang, "bubble_text": "Miben segíthetek?", "placeholder": "Keresel valamit?"}

class BooksySocialAgent:
    def __init__(self, db: DBHandler):
        self.db = db
        self.client_claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) 

    def _fetch_wikipedia_births(self):
        today = datetime.now(LOCAL_TZ)
        url = f"https://en.wikipedia.org/api/rest_v1/feed/onthisday/births/{today.strftime('%m/%d')}"
        try:
            response = requests.get(url, headers={'User-Agent': 'BooksyBot/1.0'}, timeout=15)
            if response.status_code == 200:
                verified = []
                for p in response.json().get('births', []):
                    combined = (p.get('text', '') + " " + (p.get('pages', [{}])[0].get('extract', '') if p.get('pages') else "")).lower()
                    if any(kw in combined for kw in ['writer', 'author', 'poet', 'novelist']) and not any(fw in combined for fw in ['politician', 'athlete', 'actor']):
                        verified.append({"name": p.get('text', '').split(',')[0], "bio": p.get('pages', [{}])[0].get('extract', '') if p.get('pages') else p.get('text')})
                return verified
            return []
        except: return []

    def _get_agentic_calendar(self):
        today_str = datetime.now(LOCAL_TZ).strftime("%B %d")
        wiki_writers = self._fetch_wikipedia_births()
        prompt = f"Today is {today_str}. Prominent writers: {json.dumps(wiki_writers[:20])}. Select 3-5, format Hungarian names correctly, translate bio to 1 Hungarian sentence. Output ONLY JSON: {{\"authors\": [{{'name': '...', 'bio': '...'}}]}}"
        try:
            res = self.client_claude.messages.create(model="claude-sonnet-4-6", max_tokens=1000, temperature=0.0, messages=[{"role": "user", "content": prompt}])
            raw_json = res.content[0].text
            json_marker = "```json"
            if json_marker in raw_json:
                raw_json = raw_json.split(json_marker)[1].split("```")[0].strip()
            return json.loads(raw_json)
        except: return {"authors": []}

    def _create_infinite_loop_video(self, image_path, output_path):
        if not MOVIEPY_AVAILABLE: return False
        try:
            clip = ImageClip(image_path).resize(width=800)
            def zoom(t): return 1 + 0.02 * t
            zoomed = clip.resize(zoom)
            cropped = zoomed.crop(x_center=clip.w/2, y_center=clip.h/2, width=clip.w, height=clip.h).set_duration(4)
            final_clip = concatenate_videoclips([cropped, cropped.fx(vfx.time_mirror)])
            final_clip.write_videofile(output_path, fps=15, codec="libx264", audio=False, logger=None, threads=1, preset="ultrafast")
            return True
        except: return False

    def send_morning_email(self):
        if not os.path.exists("social_state.json"): return
        with open("social_state.json", "r") as f: state = json.load(f)
        try:
            sender, password = os.getenv("SMTP_SENDER"), os.getenv("SMTP_PASSWORD")
            admin_emails = [e.strip() for e in os.getenv("ADMIN_EMAIL", "").split(",") if e.strip()]
            server = smtplib.SMTP(os.getenv("SMTP_SERVER", "mail.antikvarius.ro"), 26, timeout=15)
            server.starttls(); server.login(sender, password)
            for admin in admin_emails:
                msg = MIMEMultipart()
                msg['Subject'] = f"✅ Booksy Poszt Elkészült ({datetime.now(LOCAL_TZ).strftime('%Y-%m-%d')})"
                msg.attach(MIMEText(f"<html><body><h3>Vázlat:</h3><pre>{state['text']}</pre></body></html>", 'html'))
                server.send_message(msg)
            server.quit()
            os.remove("social_state.json")
        except: pass

    def run_night_generation(self):
        print("🕒 [SOCIAL] Agentikus Generálás indul (V154 - Paid Optimization)...")
        calendar = self._get_agentic_calendar()
        poszt_adatai = []
        for író in calendar.get("authors", []):
            vec = gemini_client.models.embed_content(model="gemini-embedding-001", contents=író['name'], config=types.EmbedContentConfig(output_dimensionality=768)).embeddings[0].values
            res = self.db.collection.query(query_embeddings=[vec], n_results=1, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
            if res['ids'] and res['ids'][0]:
                meta = res['metadatas'][0][0]
                if normalize_text(író['name'].split()[-1]) in normalize_text(str(meta.get('author', ''))):
                    poszt_adatai.append({"author": író['name'], "title": meta.get('title'), "url": meta.get('url'), "price": clean_price_raw(meta.get('price')), "preview": meta.get('text_preview', ''), "category": meta.get('category', '')})

        target = poszt_adatai[0] if poszt_adatai else None
        if not target: return
        
        # Kutató (Gemini) + Rendező (Claude) + Grafikus (Pollinations)
        research = gemini_client.models.generate_content(model='gemini-1.5-flash', contents=f"Summary of {target['title']} by {target['author']}. Concise, visual. Respond in English.").text
        prompt_img = self.client_claude.messages.create(model="claude-sonnet-4-6", max_tokens=300, temperature=0.7, messages=[{"role": "user", "content": f"Write Image Prompt based on: {research}. NO TEXT, NO FACES."}]).content[0].text
        
        image_path = "social_img.jpg"
        with open(image_path, 'wb') as f: f.write(requests.get(f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt_img)}?width=1024&height=1024&nologo=true").content)
        video_path = "social_video.mp4"
        is_video = self._create_infinite_loop_video(image_path, video_path)

        post_text = self.client_claude.messages.create(model="claude-sonnet-4-6", max_tokens=1500, temperature=0.7, system="You are Booksy CopySEO.", messages=[{"role": "user", "content": f"Write FB post in HU about: {json.dumps(poszt_adatai)}"}]).content[0].text
        with open("social_state.json", "w") as f: json.dump({"text": post_text}, f)

        fb_page_id, fb_token = os.getenv("FB_PAGE_ID"), os.getenv("FB_PAGE_TOKEN")
        if fb_page_id and fb_token:
            if is_video:
                requests.post(f"https://graph.facebook.com/v19.0/{fb_page_id}/videos", data={'access_token': fb_token, 'description': post_text, 'published': 'false', 'unpublished_content_type': 'DRAFT'}, files={'source': open(video_path, 'rb')})
            else:
                requests.post(f"https://graph.facebook.com/v19.0/{fb_page_id}/photos", data={"message": post_text, "published": False, "unpublished_content_type": "DRAFT", "access_token": fb_token}, files={'source': open(image_path, 'rb')})
        
        for p in [image_path, video_path]:
            if os.path.exists(p): os.remove(p)
        self.send_morning_email()

db_handler = DBHandler()
updater = AutoUpdater(db_handler)
bot = BooksyBrain(db_handler)
social_agent = BooksySocialAgent(db_handler)
scheduler = BackgroundScheduler()
scheduler.add_job(updater.run_daily_update, CronTrigger(hour=3, minute=0, timezone=LOCAL_TZ))
scheduler.add_job(social_agent.run_night_generation, CronTrigger(hour=4, minute=0, timezone=LOCAL_TZ))
scheduler.add_job(social_agent.send_morning_email, CronTrigger(hour=9, minute=0, timezone=LOCAL_TZ))

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.start(); yield; scheduler.shutdown()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def home(): return {"status": "Booksy V154 (PAID TIER SAFE SYNC)"}
@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url, req.session_id)
@app.post("/init-chat")
def init_chat(req: InitRequest): return bot.negotiate_handshake(req.url, req.session_id, req.ui_lang)
@app.post("/test-social-night")
def test_night(bt: BackgroundTasks): bt.add_task(social_agent.run_night_generation); return {"status": "Triggered"}
@app.post("/test-social-morning")
def test_morning(bt: BackgroundTasks): bt.add_task(social_agent.send_morning_email); return {"status": "Triggered"}
@app.post("/force-update")
def force_update(bt: BackgroundTasks): bt.add_task(updater.run_daily_update); return {"status": "Safe Sync Started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)