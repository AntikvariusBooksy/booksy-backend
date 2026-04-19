# BOOKSY BRAIN - V147 (CLAUDE SONNET 4.6 & BASE64 URL PROTECTION)
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
import base64
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
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

# --- BOLONDBIZTOS BASE64 URL-EK A MARKDOWN MÁSOLÁSI HIBA ELLEN ---
# aHR0cHM6Ly93d3cuYW50aWt2YXJpdXMucm8vd3AtY29udGVudC91cGxvYWRzL3dvby1mZWVkL2dvb2dsZS94bWwvYm9va3N5ZnVsbGZlZWQueG1s -> https://www.antikvarius.ro/...
XML_FEED_URL = os.getenv("XML_FEED_URL", base64.b64decode("aHR0cHM6Ly93d3cuYW50aWt2YXJpdXMucm8vd3AtY29udGVudC91cGxvYWRzL3dvby1mZWVkL2dvb2dsZS94bWwvYm9va3N5ZnVsbGZlZWQueG1s").decode('utf-8'))
TEMP_FILE = "temp_feed.xml"
LOCAL_TZ = pytz.timezone('Europe/Bucharest')

def normalize_text(text):
    if not text: return ""
    return ''.join(c for c in unicodedata.normalize('NFD', str(text).lower()) if unicodedata.category(c) != 'Mn')

def safe_str(val):
    return html.unescape(str(val).strip()) if val else ""

def generate_content_hash(data_string): 
    return hashlib.md5(data_string.encode('utf-8')).hexdigest()

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
        self.collection = self.client.get_or_create_collection(name="booksy_collection")

class AutoUpdater:
    def __init__(self, db: DBHandler):
        self.client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
                d_hash = generate_content_hash(f"V147|{bid}|{book_data['title']}|{book_data['price']}")
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

class ChatRequest(BaseModel): message: str; context_url: Optional[str] = ""; session_id: Optional[str] = ""
class InitRequest(BaseModel): url: str; session_id: str; ui_lang: str = "ro"

class BooksyBrain:
    def __init__(self, db: DBHandler):
        self.db = db
        self.client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.user_session_cache = {}

    def process(self, msg, context_url, session_id):
        try:
            analysis = self.client_ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Intent analysis for bookstore. Input: '{msg}'. JSON output: {{\"intent\": \"search\"|\"policy\", \"query\": \"query\"}}" }],
                response_format={"type": "json_object"}, temperature=0.0
            ).choices[0].message.content
            intent_data = json.loads(analysis)
            
            vec = self.client_ai.embeddings.create(input=intent_data.get('query', msg), model="text-embedding-3-small").data[0].embedding
            final_reply, final_products = "", []
            
            if intent_data.get('intent') == "policy":
                res = self.db.collection.query(query_embeddings=[vec], n_results=2, where={"type": "policy"})
                ctx = "".join([m.get('text', '') for m in res['metadatas'][0]]) if res['ids'] else ""
                final_reply = self.client_ai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": f"Context: {ctx}\nQ: {msg}"}]).choices[0].message.content
            else:
                res = self.db.collection.query(query_embeddings=[vec], n_results=5, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
                if res['ids']:
                    ctx_text = ""
                    for meta in res['metadatas'][0]:
                        p_price = clean_price_raw(meta.get('price'))
                        final_products.append({"title": meta.get('title'), "price": p_price, "url": meta.get('url'), "image": meta.get('image_url')})
                        ctx_text += f"- {meta.get('title')} ({p_price})\n"
                    final_reply = self.client_ai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": f"Recommend books: {ctx_text}"}, {"role": "user", "content": msg}]).choices[0].message.content
                else: final_reply = "Sajnos nem találtam megfelelő könyvet."
            
            self.user_session_cache[session_id] = msg
            return {"reply": final_reply, "products": final_products}
        except: return {"reply": "Hiba történt.", "products": []}

    def negotiate_handshake(self, url, session_id, ui_lang):
        try:
            res = self.client_ai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": f"JSON greeting in {ui_lang}."}], response_format={"type": "json_object"}).choices[0].message.content
            return json.loads(res)
        except: return {"ui_lang": ui_lang, "bubble_text": "Miben segíthetek?", "placeholder": "Keresel valamit?"}

class BooksySocialAgent:
    def __init__(self, db: DBHandler):
        self.db = db
        self.client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 
        self.client_claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) 

    def _fetch_wikipedia_births(self):
        today = datetime.now(LOCAL_TZ)
        mm, dd = today.strftime("%m"), today.strftime("%d")
        
        # Base64 Dekódolás a Wiki URL-hez
        base_wiki = base64.b64decode("aHR0cHM6Ly9lbi53aWtpcGVkaWEub3JnL2FwaS9yZXN0X3YxL2ZlZWQvb250aGlzZGF5L2JpcnRocy8=").decode('utf-8')
        url = f"{base_wiki}{mm}/{dd}"
        
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
        
        prompt = f"""
        Today's exact local date in Bucharest, Romania is {today_str}. Act as factual editor.
        Select ONLY prominent literary writers from this list: {wiki_text}
        
        CRITICAL RULES: 
        1. Max 3-5 people. 
        2. Format ONLY Hungarian names as Lastname Firstname.
        3. You MUST translate and summarize the bio into exactly 1 SENTENCE IN HUNGARIAN.
        
        Output ONLY valid JSON in this exact structure, with no markdown formatting around it:
        {{
            "holiday": "Name of holiday or null",
            "authors": [
                {{"name": "...", "bio": "Hungarian translation of bio..."}}
            ]
        }}
        """
        try:
            print("🧠 [CLAUDE] Naptár elemzése folyamatban...")
            res = self.client_claude.messages.create(
                model="claude-sonnet-4-6", # A BIZONYÍTOTTAN MŰKÖDŐ MODELL
                max_tokens=1000,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            raw_json = res.content[0].text
            if "```json" in raw_json:
                raw_json = raw_json.split("```json")[1].split("```")[0].strip()
            return json.loads(raw_json)
        except Exception as e:
            print(f"❌ Claude API hiba (Naptár): {e}") 
            return {"holiday": None, "authors": []}

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
        sender, password = os.getenv("SMTP_SENDER"), os.getenv("SMTP_PASSWORD")
        server_addr = os.getenv("SMTP_SERVER", "mail.antikvarius.ro")
        admin_emails = [e.strip() for e in os.getenv("ADMIN_EMAIL", "").split(",") if e.strip()]
        if not sender or not password or not admin_emails: return
        try:
            server = smtplib.SMTP(server_addr, 26, timeout=15)
            server.ehlo(); server.starttls(); server.ehlo()
            server.login(sender, password)
            for admin in admin_emails:
                msg = MIMEMultipart()
                msg['From'] = f"Booksy Social Agent <{sender}>"; msg['To'] = admin
                msg['Subject'] = f"✅ Booksy Poszt Elkészült ({datetime.now(LOCAL_TZ).strftime('%Y-%m-%d')})"
                msg['Date'] = formatdate(localtime=False); msg['Message-ID'] = make_msgid(domain=server_addr.replace('mail.', ''))
                body = f"<html><body><h3>Vázlat elkészült:</h3><pre style='background:#f4f4f4;padding:10px;'>{state['text']}</pre></body></html>"
                msg.attach(MIMEText(body, 'html'))
                server.send_message(msg)
            server.quit()
            os.remove("social_state.json")
        except: pass

    def run_night_generation(self):
        print("🕒 [SOCIAL] Agentikus Generálás indul (V147 - BASE64 FB URLS + CLAUDE 4.6)...")
        calendar = self._get_agentic_calendar()
        ünnepeltek = calendar.get("authors", [])
        
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
            themes = ["ritka antikvár könyv", "izgalmas krimik", "klasszikus magyar szépirodalom", "történelmi szakkönyvek"]
            vec = self.client_ai.embeddings.create(input=random.choice(themes), model="text-embedding-3-small").data[0].embedding
            fallback_res = self.db.collection.query(query_embeddings=[vec], n_results=10, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
            if fallback_res['ids'] and fallback_res['ids'][0]:
                for i in random.sample(range(len(fallback_res['ids'][0])), min(3, len(fallback_res['ids'][0]))):
                    f_meta = fallback_res['metadatas'][0][i]
                    fallback_adatai.append({"author": f_meta.get('author', 'Ismeretlen'), "title": f_meta.get('title'), "url": f_meta.get('url'), "price": clean_price_raw(f_meta.get('price')), "preview": f_meta.get('text_preview', '')})

        konyv_cim = poszt_adatai[0]['title'] if has_author_books else (fallback_adatai[0]['title'] if fallback_adatai else "Antikvár kincsek")
        konyv_tartalom = poszt_adatai[0].get('preview', '') if has_author_books else (fallback_adatai[0].get('preview', '') if fallback_adatai else "")
        
        img_prompt = f"A photorealistic, cinematic atmospheric scene inspired by the mood of the book '{konyv_cim}'. Context: '{konyv_tartalom[:150]}'. Style: high-end photography, 8k resolution, lifelike textures. CRITICAL: DO NOT include any text, letters, typography, words, or signs in the image. Ensure any visible book covers are completely blank without any writing. No human faces."
        video_path, image_path = "social_video.mp4", "social_img.jpg"
        media_url, is_video = "", False
        try:
            print("🎨 [OPENAI] DALL-E kép generálása...")
            img_res = self.client_ai.images.generate(model="dall-e-3", prompt=img_prompt, size="1024x1024", quality="hd", n=1)
            media_url = img_res.data[0].url
            with open(image_path, 'wb') as f: f.write(requests.get(media_url).content)
            is_video = self._create_infinite_loop_video(image_path, video_path)
        except Exception as e: print(f"❌ Kép hiba: {e}")

        marketing_prompt = f"Act as Booksy CopySEO, the ultimate marketing expert. Write an engaging Facebook post in Hungarian. State clearly that TODAY is the birthday of: {json.dumps(ünnepeltek)}. Holiday: {calendar.get('holiday')}. Books to recommend: {json.dumps(poszt_adatai if has_author_books else fallback_adatai)}. Use the exact provided URLs. Keep the tone elegant and persuasive. Do not hallucinate."
        
        try:
            print("🖋️ [CLAUDE] Szövegírás folyamatban...")
            post_res = self.client_claude.messages.create(
                model="claude-sonnet-4-6", # A BIZONYÍTOTTAN MŰKÖDŐ MODELL
                max_tokens=1500,
                temperature=0.7,
                messages=[{"role": "user", "content": marketing_prompt}]
            )
            post_text = post_res.content[0].text
        except Exception as e:
            print(f"❌ Claude API hiba (Szövegírás): {e}")
            post_text = "Hiba történt a szöveg generálása során."

        with open("social_state.json", "w") as f: json.dump({"text": post_text}, f)

        fb_page_id, fb_token = os.getenv("FB_PAGE_ID"), os.getenv("FB_PAGE_TOKEN")
        if not fb_page_id or not fb_token:
            print("⚠️ [FB] Hiányzik a FB_PAGE_ID vagy az FB_PAGE_TOKEN a környezeti változókból!")
        else:
            try:
                print("🚀 [FB] Poszt küldése a Meta Business Suite-ba...")
                
                # Base64 Dekódolás a FB URL-hez
                fb_base = base64.b64decode("aHR0cHM6Ly9ncmFwaC5mYWNlYm9vay5jb20vdjE5LjAv").decode('utf-8')
                
                if is_video:
                    res = requests.post(f"{fb_base}{fb_page_id}/videos", data={'access_token': fb_token, 'description': post_text, 'published': 'false', 'unpublished_content_type': 'DRAFT'}, files={'source': open(video_path, 'rb')})
                else:
                    res = requests.post(f"{fb_base}{fb_page_id}/photos", json={"url": media_url, "message": post_text, "published": False, "unpublished_content_type": "DRAFT", "access_token": fb_token})
                
                if res.status_code == 200:
                    print(f"✅ [FB] TÖKÉLETES SIKER! FB ID: {res.json().get('id')}")
                else:
                    print(f"❌ [FB] API Hiba ({res.status_code}): {res.text}")
            except Exception as e: 
                print(f"❌ [FB] Rendszerhiba a küldés során: {e}")

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

@app.get("/")
def home(): return {"status": "Booksy V147 (BASE64 FB URLS & CLAUDE 4.6)"}
@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url, req.session_id)
@app.post("/init-chat")
def init_chat(req: InitRequest): return bot.negotiate_handshake(req.url, req.session_id, req.ui_lang)
@app.post("/test-social-night")
def test_night(bt: BackgroundTasks): bt.add_task(social_agent.run_night_generation); return {"status": "Triggered"}
@app.post("/test-social-morning")
def test_morning(bt: BackgroundTasks): bt.add_task(social_agent.send_morning_email); return {"status": "Triggered"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)