# BOOKSY BRAIN - V104 (CORE V102 + ISOLATED PROACTIVE SOCIAL AGENT WITH SSL EMAIL)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import time
import requests
import hashlib
import re
import json
import unicodedata
import html
import xml.etree.ElementTree as ET
import gc
import chromadb
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

# --- VIDEÓ FELDOLGOZÓ ---
try:
    from moviepy.editor import ImageClip, concatenate_videoclips
    import moviepy.video.fx.all as vfx
    MOVIEPY_AVAILABLE = True
except Exception as e:
    print(f"MoviePy import warning: {e}")
    MOVIEPY_AVAILABLE = False

load_dotenv()
XML_FEED_URL = os.getenv("XML_FEED_URL", "https://www.antikvarius.ro/wp-content/uploads/woo-feed/google/xml/booksyfullfeed.xml")
TEMP_FILE = "temp_feed.xml"

POLICY_PAGES = [
    {"url": "https://www.antikvarius.ro/termeni-si-conditii-de-utilizare/", "lang": "ro", "name": "Termeni și condiții"},
    {"url": "https://www.antikvarius.ro/informatii-despre-plata/", "lang": "ro", "name": "Informații despre plată"},
    {"url": "https://www.antikvarius.ro/informatii-despre-livrare/", "lang": "ro", "name": "Informații despre livrare"},
    {"url": "https://www.antikvarius.ro/contact/", "lang": "ro", "name": "Contact"},
]

# ==========================================
# 1. HELPEREK ÉS ALAP FUNKCIÓK (V102 Parity)
# ==========================================
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

def parse_price_to_float(price_input):
    try:
        if price_input is None: return None
        s = str(price_input).lower().replace("ron", "").replace("lei", "").replace(" ", "").strip()
        s = s.replace(",", ".") 
        if not s: return None
        return float(s)
    except: return None

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
        for attempt in range(3):
            try:
                with requests.get(XML_FEED_URL, headers=headers, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with open(TEMP_FILE, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
                if os.path.getsize(TEMP_FILE) > 10000: return True
            except: time.sleep(5)
        return False

    def update_policies(self, current_ts, force_refresh=False):
        for page in POLICY_PAGES:
            try:
                r = requests.get(page['url'], timeout=30)
                if r.status_code == 200:
                    clean_text = html_to_markdown_clean(r.text)
                    d_hash = generate_content_hash(clean_text)
                    page_id = f"policy_{generate_content_hash(page['url'])}"
                    if not force_refresh:
                        try:
                            existing = self.db.collection.get(ids=[page_id], include=['metadatas'])
                            if existing['ids'] and existing['metadatas'][0].get('content_hash') == d_hash: continue
                        except: pass
                    emb = self.client_ai.embeddings.create(input=f"Típus: Szabályzat. Cím: {page['name']}. Tartalom: {clean_text[:8000]}", model="text-embedding-3-small").data[0].embedding
                    meta = {"title": page['name'], "url": page['url'], "text": clean_text, "lang": "ro", "type": "policy", "content_hash": d_hash, "last_seen": current_ts}
                    self.db.collection.upsert(ids=[page_id], embeddings=[emb], metadatas=[meta])
            except: pass

    def run_daily_update(self, force_refresh=False):
        print(f"🔄 [AUTO] Frissítés (Force: {force_refresh})")
        current_sync_ts = int(time.time())
        self.update_policies(current_sync_ts, force_refresh)
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
                d_hash = generate_content_hash(f"V104|{bid}|{book_data['title']}|{book_data['price']}|{book_data['publisher']}")
                book_data['content_hash'] = d_hash
                if not force_refresh:
                    try:
                        existing = self.db.collection.get(ids=[bid], include=['metadatas'])
                        if existing['ids'] and existing['metadatas'][0].get('content_hash') == d_hash: continue
                    except: pass
                
                emb_text = f"SKU: {bid}. Nyelv: {book_data['lang']}. Cím: {book_data['title']}. Szerző: {book_data['author']}. Kiadó: {book_data['publisher']}. Leírás: {book_data['description'][:800]}"
                try:
                    emb = self.client_ai.embeddings.create(input=emb_text[:8000], model="text-embedding-3-small").data[0].embedding
                    clean_meta = book_data.copy()
                    del clean_meta['description'] 
                    clean_meta['text_preview'] = book_data['description'][:100]
                    ids_batch.append(bid); embeddings_batch.append(emb); metadatas_batch.append(clean_meta)
                    if len(ids_batch) >= 50:
                        self.db.collection.upsert(ids=ids_batch, embeddings=embeddings_batch, metadatas=metadatas_batch)
                        ids_batch, embeddings_batch, metadatas_batch = [], [], []
                except: pass
            if ids_batch: self.db.collection.upsert(ids=ids_batch, embeddings=embeddings_batch, metadatas=metadatas_batch)
            if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)
            print("🏁 [VÉGE] Adatbázis frissítve.")
        except Exception as e: print(f"❌ Hiba: {e}")

# ==========================================
# 2. BOOKSY CHAT ASSZISZTENS (Érintetlen)
# ==========================================
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
                messages=[{"role": "user", "content": f"Intent analysis for bookstore. Input: '{msg}'. Context: '{self.user_session_cache.get(session_id, '')}'. Output JSON: {{\"intent\": \"search_book\"|\"policy\", \"query\": \"translated query if policy in ro, else normal\"}}" }],
                response_format={"type": "json_object"}, temperature=0.0
            ).choices[0].message.content
            intent_data = json.loads(analysis)
            
            final_reply, final_products = "", []
            vec = self.client_ai.embeddings.create(input=intent_data.get('query', msg), model="text-embedding-3-small").data[0].embedding
            
            if intent_data.get('intent') == "policy":
                res = self.db.collection.query(query_embeddings=[vec], n_results=2, where={"type": "policy"})
                ctx = "".join([m.get('text', '') for m in res['metadatas'][0]]) if res['ids'] else ""
                final_reply = self.client_ai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "You are Booksy. Answer briefly using context."}, {"role": "user", "content": f"Context: {ctx}\nQ: {msg}"}]).choices[0].message.content
            else:
                res = self.db.collection.query(query_embeddings=[vec], n_results=5, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
                if res['ids']:
                    ctx_text = ""
                    for meta in res['metadatas'][0]:
                        p_price = clean_price_raw(meta.get('price'))
                        final_products.append({"title": meta.get('title'), "price": p_price, "url": meta.get('url'), "image": meta.get('image_url')})
                        ctx_text += f"- {meta.get('title')} ({p_price})\n"
                    final_reply = self.client_ai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": f"Recommend these books enthusiastically in user's language.\n{ctx_text}"}, {"role": "user", "content": msg}]).choices[0].message.content
                else: final_reply = "Sajnos nem találtam megfelelő könyvet."
            
            self.user_session_cache[session_id] = msg
            return {"reply": final_reply, "products": final_products}
        except Exception as e: return {"reply": "Hiba történt.", "products": []}

    def negotiate_handshake(self, url, session_id, ui_lang):
        try:
            res = self.client_ai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": f"Create short JSON greeting and placeholder in {ui_lang}."}], response_format={"type": "json_object"}).choices[0].message.content
            data = json.loads(res)
            data['ui_lang'] = ui_lang
            return data
        except: return {"ui_lang": ui_lang, "bubble_text": "Miben segíthetek?" if ui_lang=="hu" else "Cu ce te pot ajuta?", "placeholder": "Keresel valamit?" if ui_lang=="hu" else "Cauți o carte?"}

# ==========================================
# 3. BOOKSY SOCIAL AGENT (Teljesen Elkülönítve)
# ==========================================
class BooksySocialAgent:
    def __init__(self, db: DBHandler):
        self.db = db
        self.client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._ensure_calendars_exist()

    def _ensure_calendars_exist(self):
        if not os.path.exists("authors_calendar.json"):
            with open("authors_calendar.json", "w", encoding="utf-8") as f:
                json.dump([
                    {"name": "Jókai Mór", "birthday": "02-18"},
                    {"name": "Petőfi Sándor", "birthday": "01-01"}
                ], f, ensure_ascii=False, indent=4)
        if not os.path.exists("holidays.json"):
            with open("holidays.json", "w", encoding="utf-8") as f:
                json.dump({"01-01": "Újév / Anul Nou", "03-15": "Március 15. Ünnep"}, f, ensure_ascii=False, indent=4)

    def _create_infinite_loop_video(self, image_path, output_path):
        if not MOVIEPY_AVAILABLE: return False
        try:
            print("🎥 Videó renderelés indítása (Infinite Loop)...")
            clip = ImageClip(image_path).set_duration(5)
            def resize_func(t): return 1 + 0.02 * t 
            zoomed = clip.resize(resize_func)
            w, h = clip.size
            zoomed = zoomed.crop(x_center=zoomed.w/2, y_center=zoomed.h/2, width=w, height=h)
            reversed_clip = zoomed.fx(vfx.time_mirror)
            final_clip = concatenate_videoclips([zoomed, reversed_clip])
            final_clip.write_videofile(output_path, fps=24, codec="libx264", logger=None)
            return True
        except Exception as e:
            print(f"❌ Videó renderelési hiba: {e}")
            return False

    def run_night_generation(self):
        print("🕒 [SOCIAL] Éjszakai Generálás indul...")
        today_md = datetime.now().strftime("%m-%d")
        
        with open("authors_calendar.json", "r", encoding="utf-8") as f: authors = json.load(f)
        with open("holidays.json", "r", encoding="utf-8") as f: holidays = json.load(f)
        
        napi_ünnep = holidays.get(today_md, None)
        ünnepeltek = [a for a in authors if a['birthday'] == today_md]
        
        poszt_adatai = []
        for író in ünnepeltek[:10]:
            vec = self.client_ai.embeddings.create(input=író['name'], model="text-embedding-3-small").data[0].embedding
            results = self.db.collection.query(query_embeddings=[vec], n_results=1, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
            if results['ids']:
                meta = results['metadatas'][0][0]
                if író['name'].split()[-1].lower() in str(meta.get('author', '')).lower() or író['name'].split()[0].lower() in str(meta.get('author', '')).lower():
                    poszt_adatai.append({"author": író['name'], "title": meta.get('title'), "url": meta.get('url'), "price": meta.get('price')})

        if not poszt_adatai and not napi_ünnep:
            print("🚫 Nincs releváns író/könyv ma. Készenlét.")
            return

        konyv_cim = poszt_adatai[0]['title'] if poszt_adatai else "Antikvár ritkaságok"
        img_prompt = f"Based on the book '{konyv_cim}', create an atmospheric scene. No text, no faces. Style: Classic Dark Academia, warm vintage bookstore lighting, steaming teacup on dark wood table, cinematic 8k resolution, mysterious mood."
        
        video_path = "social_video.mp4"
        image_path = "social_img.jpg"
        media_url = ""
        is_video = False

        try:
            img_res = self.client_ai.images.generate(model="dall-e-3", prompt=img_prompt, size="1024x1024", quality="hd", n=1)
            media_url = img_res.data[0].url
            img_data = requests.get(media_url).content
            with open(image_path, 'wb') as handler: handler.write(img_data)
            if self._create_infinite_loop_video(image_path, video_path): is_video = True
        except Exception as e: print(f"⚠️ Vizuális hiba: {e}")

        prompt = f"""
        Act as Booksy, the Webdevmk marketing agent. Write a Hungarian Facebook post.
        Holiday today: {napi_ünnep}. Featured Authors & Books: {poszt_adatai}.
        Rule: Use a 'Storytelling' or 'FOMO' framework. Be elegant, moody, and engaging. Include URLs.
        Output ONLY the final post text.
        """
        post_text = self.client_ai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]).choices[0].message.content

        fb_page_id = os.getenv("FB_PAGE_ID")
        fb_token = os.getenv("FB_PAGE_TOKEN")
        
        if fb_page_id and fb_token:
            try:
                if is_video:
                    fb_url = f"https://graph.facebook.com/v19.0/{fb_page_id}/videos"
                    files = {'source': open(video_path, 'rb')}
                    data = {'access_token': fb_token, 'description': post_text, 'published': 'false'}
                    res = requests.post(fb_url, data=data, files=files)
                else:
                    fb_url = f"https://graph.facebook.com/v19.0/{fb_page_id}/photos"
                    res = requests.post(fb_url, json={"url": media_url, "message": post_text, "published": "false", "access_token": fb_token})
                if res.ok: print("✅ FB Draft Created.")
                else: print(f"❌ FB API Error: {res.text}")
            except Exception as e: print(f"❌ FB Upload Error: {e}")

        with open("social_state.json", "w") as f:
            json.dump({"text": post_text, "type": "video" if is_video else "image"}, f)
        
        if os.path.exists(image_path): os.remove(image_path)
        if os.path.exists(video_path): os.remove(video_path)

    def send_morning_email(self):
        if not os.path.exists("social_state.json"): return
        with open("social_state.json", "r") as f: state = json.load(f)
        
        sender = os.getenv("SMTP_SENDER")
        password = os.getenv("SMTP_PASSWORD")
        smtp_server = os.getenv("SMTP_SERVER", "mail.antikvarius.ro")
        smtp_port = int(os.getenv("SMTP_PORT", 465))
        
        # Több adminisztrátor kezelése (vesszővel elválasztott lista az .env-ből)
        admin_emails_raw = os.getenv("ADMIN_EMAIL", "")
        admin_emails = [email.strip() for email in admin_emails_raw.split(",") if email.strip()]

        if sender and password and admin_emails:
            try:
                # Port 465 -> SMTP_SSL csatlakozás szükséges!
                server = smtplib.SMTP_SSL(smtp_server, smtp_port)
                server.login(sender, password)
                
                for admin in admin_emails:
                    msg = MIMEMultipart()
                    msg['From'] = f"Booksy Social Agent <{sender}>"
                    msg['To'] = admin
                    msg['Subject'] = "✅ Booksy: Új Facebook Vázlat Elkészült!"
                    html_body = f"<p>A mai ({state.get('type', 'ismeretlen')}) posztod bent van a Meta Business Suite vázlatai között.</p><pre>{state.get('text', '')}</pre>"
                    msg.attach(MIMEText(html_body, 'html'))
                    
                    server.send_message(msg)
                    print(f"✅ Reggeli email elküldve neki: {admin}")
                
                server.quit()
            except Exception as e:
                print(f"❌ Email küldési hiba: {e}")
        
        try: os.remove("social_state.json")
        except: pass

# ==========================================
# 4. RENDSZER INICIALIZÁLÁS ÉS VÉGPONTOK
# ==========================================
db_handler = DBHandler()
updater = AutoUpdater(db_handler)
bot = BooksyBrain(db_handler)
social_agent = BooksySocialAgent(db_handler)

scheduler = BackgroundScheduler()
scheduler.add_job(updater.run_daily_update, CronTrigger(hour=3, minute=0))
scheduler.add_job(social_agent.run_night_generation, CronTrigger(hour=4, minute=0))
scheduler.add_job(social_agent.send_morning_email, CronTrigger(hour=9, minute=0))

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.start()
    yield
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def home(): return {"status": "Booksy V104 (DUAL AGENT: Chat + Social Active - SSL EMail)"}
@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url, req.session_id)
@app.post("/init-chat")
def init_chat(req: InitRequest): return bot.negotiate_handshake(req.url, req.session_id, req.ui_lang)
@app.post("/force-update")
def force(bt: BackgroundTasks): bt.add_task(updater.run_daily_update, force_refresh=True); return {"status": "Force Update Started"}
@app.post("/test-social-night")
def test_night(bt: BackgroundTasks): bt.add_task(social_agent.run_night_generation); return {"status": "Night Generation Triggered"}
@app.post("/test-social-morning")
def test_morning(bt: BackgroundTasks): bt.add_task(social_agent.send_morning_email); return {"status": "Morning Email Triggered"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)