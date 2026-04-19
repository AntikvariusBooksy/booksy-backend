# BOOKSY BRAIN - V128 (SMTP 587 STARTTLS FIX + STRICT AI PROMPT FIX)
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
import xml.etree.ElementTree as ET
import gc
import chromadb
import pytz
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from typing import List, Optional, Dict, Any
from datetime import datetime
from bs4 import BeautifulSoup
import markdownify
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- KÖTELEZŐ MONKEY PATCH A PILLOW 10+ ÉS MOVIEPY KOMPATIBILITÁSHOZ ---
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
XML_FEED_URL = os.getenv("XML_FEED_URL", "https://www.antikvarius.ro/wp-content/uploads/woo-feed/google/xml/booksyfullfeed.xml")
TEMP_FILE = "temp_feed.xml"

# --- HIVATALOS ROMÁNIAI IDŐZÓNA BEÁLLÍTÁSA ---
LOCAL_TZ = pytz.timezone('Europe/Bucharest')

# --- ALAP FUNKCIÓK ---
def normalize_text(text):
    if not text: return ""
    text = str(text).lower()
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def safe_str(val):
    if val is None: return ""
    return html.unescape(str(val).strip())

def generate_content_hash(data_string): 
    return hashlib.md5(data_string.encode('utf-8')).hexdigest()

def clean_price_raw(raw_price):
    if not raw_price: return "0 RON"
    s = str(raw_price).strip()
    cleaned_num = re.sub(r"[^\d.,]", "", s)
    if not cleaned_num: return s 
    return f"{cleaned_num} RON"

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
        self.collection = self.client.get_or_create_collection(name="booksy_collection")

class AutoUpdater:
    def __init__(self, db: DBHandler):
        self.client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.db = db

    def download_feed(self):
        headers = {'User-Agent': 'BooksyBot/1.0'}
        try:
            with requests.get(XML_FEED_URL, headers=headers, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(TEMP_FILE, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            return os.path.getsize(TEMP_FILE) > 10000
        except: return False

    def run_daily_update(self, force_refresh=False):
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
                            pub = ext_meta['publisher'] or "Ismeretlen"
                            auth = ext_meta['author'] or item_data.get('author') or "Ismeretlen"
                            lang = "ro" if "carti in limba romana" in normalize_text(cat) else "hu"
                            unique_books_buffer[bid] = {
                                "id": bid, "title": item_data.get('title') or "Nincs cím", "url": item_data.get('link', ''), 
                                "image_url": item_data.get('image_link', ''), "price": clean_price_raw(item_data.get('sale_price') or item_data.get('price')), 
                                "publisher": pub, "author": auth, "category": cat, "description": html_to_markdown_clean(raw_desc), 
                                "stock": "instock", "lang": lang, "type": "book", "last_seen": current_sync_ts
                            }
                    except: pass
                    elem.clear()
            
            ids_batch, embeddings_batch, metadatas_batch = [], [], []
            for bid, book_data in unique_books_buffer.items():
                d_hash = generate_content_hash(f"V128|{bid}|{book_data['title']}|{book_data['price']}")
                book_data['content_hash'] = d_hash
                emb_text = f"SKU: {bid}. Nyelv: {book_data['lang']}. Cím: {book_data['title']}. Szerző: {book_data['author']}. Leírás: {book_data['description'][:800]}"
                try:
                    emb = self.client_ai.embeddings.create(input=emb_text[:8000], model="text-embedding-3-small").data[0].embedding
                    clean_meta = book_data.copy()
                    del clean_meta['description'] 
                    clean_meta['text_preview'] = book_data['description'][:150]
                    ids_batch.append(bid); embeddings_batch.append(emb); metadatas_batch.append(clean_meta)
                    if len(ids_batch) >= 50:
                        self.db.collection.upsert(ids=ids_batch, embeddings=embeddings_batch, metadatas=metadatas_batch)
                        ids_batch, embeddings_batch, metadatas_batch = [], [], []
                except: pass
            if ids_batch: self.db.collection.upsert(ids=ids_batch, embeddings=embeddings_batch, metadatas=metadatas_batch)
            if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)
        except Exception as e: pass

class BooksyBrain:
    def __init__(self, db: DBHandler):
        self.db = db
        self.client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class BooksySocialAgent:
    def __init__(self, db: DBHandler):
        self.db = db
        self.client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _fetch_wikipedia_births(self):
        today = datetime.now(LOCAL_TZ)
        mm, dd = today.strftime("%m"), today.strftime("%d")
        url = f"https://en.wikipedia.org/api/rest_v1/feed/onthisday/births/{mm}/{dd}"
        headers = {'User-Agent': 'BooksyBot/1.0 (antikvarius.ro)'}
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                births = response.json().get('births', [])
                verified_writers = []
                writer_keywords = ['writer', 'author', 'novelist', 'poet', 'playwright', 'essayist', 'literature']
                forbidden_keywords = ['mathematician', 'physicist', 'economist', 'politician', 'chemist', 'biologist', 'actor', 'composer', 'singer', 'athlete', 'bishop', 'pope']
                for person in births:
                    text = person.get('text', '').lower()
                    pages = person.get('pages', [])
                    desc = (pages[0].get('extract', '').lower() + " " + pages[0].get('description', '').lower()) if pages else ""
                    combined = text + " " + desc
                    if any(kw in combined for kw in writer_keywords) and not any(fkw in combined for fkw in forbidden_keywords):
                        verified_writers.append({"name": person.get('text', '').split(',')[0], "year": person.get('year'), "bio_en": pages[0].get('extract', '') if pages else person.get('text', '')})
                return verified_writers
            return []
        except: return []

    def _get_agentic_calendar(self):
        today_local = datetime.now(LOCAL_TZ)
        today_str = today_local.strftime("%B %d")
        wiki_writers = self._fetch_wikipedia_births()
        wiki_text = json.dumps(wiki_writers[:25], ensure_ascii=False) if wiki_writers else "No valid Wikipedia data available for today."
        
        # PROMPT JAVÍTÁS (Magyar fordítás és névsorrend kötelezővé tétele)
        prompt = f"""
        Today's exact local date in Bucharest, Romania is {today_str}. Act as factual editor.
        Select ONLY prominent literary writers from this list: {wiki_text}
        
        CRITICAL RULES: 
        1. Max 3-5 people. 
        2. Format ONLY Hungarian names as Lastname Firstname (e.g. Németh László). Keep English/American names in their original order (e.g. Stanley Fish, DO NOT write Fish Stanley).
        3. You MUST translate and summarize the bio into exactly 1 SENTENCE IN HUNGARIAN. Do not leave the bio in English!
        
        Output ONLY JSON: {{"holiday": "...", "authors": [{{"name": "...", "bio": "Hungarian translation of bio..."}}]}}
        """
        try:
            res = self.client_ai.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"}, temperature=0.0).choices[0].message.content
            data = json.loads(res)
            return data
        except: return {"holiday": None, "authors": []}

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
        """E-mail küldő modul (STARTTLS / Port 587 Fix)"""
        print("📧 [EMAIL] Értesítő folyamat indul (V128)...")
        if not os.path.exists("social_state.json"):
            print("⚠️ [EMAIL] Nincs elmentett poszt (social_state.json hiányzik).")
            return
        
        with open("social_state.json", "r") as f: state = json.load(f)
        sender, password = os.getenv("SMTP_SENDER"), os.getenv("SMTP_PASSWORD")
        server_addr = os.getenv("SMTP_SERVER", "mail.antikvarius.ro")
        admin_emails = [e.strip() for e in os.getenv("ADMIN_EMAIL", "").split(",") if e.strip()]

        if not sender or not password or not admin_emails:
            print("❌ [EMAIL] SMTP adatok hiányoznak!")
            return

        try:
            print(f"🔗 [EMAIL] Csatlakozás a szerverhez: {server_addr}:587 (STARTTLS)...")
            # VÁLTOZÁS: 465 (SSL) helyett 587 (STARTTLS) a timeout hibák elkerülésére
            server = smtplib.SMTP(server_addr, 587, timeout=20)
            server.ehlo()
            server.starttls()
            server.ehlo()
            
            print("🔑 [EMAIL] Bejelentkezés...")
            server.login(sender, password)
            
            for admin in admin_emails:
                msg = MIMEMultipart()
                msg['From'] = f"Booksy Social Agent <{sender}>"; msg['To'] = admin
                msg['Subject'] = f"✅ Booksy Poszt Elkészült ({datetime.now(LOCAL_TZ).strftime('%Y-%m-%d')})"
                body = f"<html><body><h3>A mai poszt elkészült és vázlatba került:</h3><pre style='background:#f4f4f4;padding:10px;border:1px solid #ddd;'>{state['text']}</pre></body></html>"
                msg.attach(MIMEText(body, 'html'))
                server.send_message(msg)
                print(f"📧 [EMAIL] Sikeresen elküldve ide: {admin}")
            
            server.quit()
            os.remove("social_state.json")
            print("🗑️ [EMAIL] social_state.json törölve.")
        except Exception as e:
            print(f"❌ [EMAIL] Kritikus SMTP hiba (587 STARTTLS): {str(e)}")

    def run_night_generation(self):
        print("🕒 [SOCIAL] Agentikus Generálás indul (V128)...")
        calendar = self._get_agentic_calendar()
        napi_ünnep, ünnepeltek = calendar.get("holiday"), calendar.get("authors", [])
        
        poszt_adatai = []
        if ünnepeltek:
            for író in ünnepeltek:
                vec = self.client_ai.embeddings.create(input=író['name'], model="text-embedding-3-small").data[0].embedding
                res = self.db.collection.query(query_embeddings=[vec], n_results=1, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
                if res['ids'] and res['ids'][0]:
                    meta = res['metadatas'][0][0]
                    if normalize_text(író['name'].split()[-1]) in normalize_text(str(meta.get('author', ''))):
                        poszt_adatai.append({"author": író['name'], "bio": író.get('bio', ''), "title": meta.get('title'), "url": meta.get('url'), "price": clean_price_raw(meta.get('price')), "preview": meta.get('text_preview', '')})

        has_author_books = len(poszt_adatai) > 0
        fallback_adatai = []
        if not has_author_books:
            themes = ["ritka antikvár könyv", "izgalmas krimik", "klasszikus magyar szépirodalom", "történelmi szakkönyvek", "önfejlesztés és pszichológia"]
            selected_theme = random.choice(themes)
            vec = self.client_ai.embeddings.create(input=selected_theme, model="text-embedding-3-small").data[0].embedding
            fallback_res = self.db.collection.query(query_embeddings=[vec], n_results=10, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
            if fallback_res['ids'] and fallback_res['ids'][0]:
                for i in random.sample(range(len(fallback_res['ids'][0])), min(3, len(fallback_res['ids'][0]))):
                    f_meta = fallback_res['metadatas'][0][i]
                    fallback_adatai.append({"author": f_meta.get('author', 'Ismeretlen'), "title": f_meta.get('title'), "url": f_meta.get('url'), "price": clean_price_raw(f_meta.get('price')), "preview": f_meta.get('text_preview', '')})

        konyv_cim = poszt_adatai[0]['title'] if has_author_books else (fallback_adatai[0]['title'] if fallback_adatai else "Antikvár kincsek")
        konyv_tartalom = poszt_adatai[0].get('preview', '') if has_author_books else (fallback_adatai[0].get('preview', '') if fallback_adatai else "")
        img_prompt = f"Cinematic conceptual scene inspired by '{konyv_cim}'. Mood: '{konyv_tartalom[:150]}'. Realistic photography, 8k, atmospheric. No text, no faces."
        
        video_path, image_path = "social_video.mp4", "social_img.jpg"
        media_url, is_video = "", False
        try:
            img_res = self.client_ai.images.generate(model="dall-e-3", prompt=img_prompt, size="1024x1024", quality="hd", n=1)
            media_url = img_res.data[0].url
            with open(image_path, 'wb') as f: f.write(requests.get(media_url).content)
            is_video = self._create_infinite_loop_video(image_path, video_path)
        except Exception as e: print(f"❌ Kép hiba: {e}")

        marketing_prompt = f"Act as Booksy CopySEO. Write FB post in HU. Today is birthday of: {json.dumps(ünnepeltek)}. Holiday: {napi_ünnep}. Books: {json.dumps(poszt_adatai if has_author_books else fallback_adatai)}. Use provided URLs."
        post_text = self.client_ai.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": marketing_prompt}]).choices[0].message.content

        with open("social_state.json", "w") as f: json.dump({"text": post_text}, f)

        fb_page_id, fb_token = os.getenv("FB_PAGE_ID"), os.getenv("FB_PAGE_TOKEN")
        if fb_page_id and fb_token:
            try:
                if is_video:
                    res = requests.post(f"https://graph.facebook.com/v19.0/{fb_page_id}/videos", data={'access_token': fb_token, 'description': post_text, 'published': 'false', 'unpublished_content_type': 'DRAFT'}, files={'source': open(video_path, 'rb')})
                else:
                    res = requests.post(f"https://graph.facebook.com/v19.0/{fb_page_id}/photos", json={"url": media_url, "message": post_text, "published": False, "unpublished_content_type": "DRAFT", "access_token": fb_token})
            except Exception as e: pass

        if os.path.exists(image_path): os.remove(image_path)
        if os.path.exists(video_path): os.remove(video_path)
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

@app.post("/test-social-night")
def test_night(bt: BackgroundTasks): bt.add_task(social_agent.run_night_generation); return {"status": "Triggered"}
@app.post("/test-social-morning")
def test_morning(bt: BackgroundTasks): bt.add_task(social_agent.send_morning_email); return {"status": "Triggered"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)