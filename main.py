# BOOKSY BRAIN - V108 (FFMPEG & MOVIEPY VERSION LOCK FIX)
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
    print("✅ MoviePy (1.0.3) és videó modul betöltve.")
except Exception as e:
    print(f"⚠️ MoviePy nem elérhető (A videó generálás képként fog lefutni): {e}")
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
                d_hash = generate_content_hash(f"V108|{bid}|{book_data['title']}|{book_data['price']}")
                book_data['content_hash'] = d_hash
                emb_text = f"SKU: {bid}. Nyelv: {book_data['lang']}. Cím: {book_data['title']}. Szerző: {book_data['author']}. Leírás: {book_data['description'][:800]}"
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

# --- BOOKSY CHAT ASSZISZTENS ---
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
            
            final_reply, final_products = "", []
            vec = self.client_ai.embeddings.create(input=intent_data.get('query', msg), model="text-embedding-3-small").data[0].embedding
            
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

# --- PURE AGENTIC SOCIAL AGENT (V108) ---
class BooksySocialAgent:
    def __init__(self, db: DBHandler):
        self.db = db
        self.client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _get_agentic_calendar(self):
        today = datetime.now().strftime("%B %d")
        prompt = f"""
        Today is {today}. Act as an expert in Hungarian, Romanian and world literature.
        Identify:
        1. Any official or cultural holidays today in Romania (for both Hungarians and Romanians). If none, output null.
        2. AT LEAST 3 up to 10 famous authors (HU, RO, World) born on this day. (There is always someone born today).
        Output ONLY a JSON:
        {{
            "holiday": "Name of holiday or null",
            "authors": [
                {{"name": "Author Name", "bio": "1-sentence bio/tribute in Hungarian"}}
            ]
        }}
        """
        try:
            res = self.client_ai.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"}).choices[0].message.content
            return json.loads(res)
        except: return {"holiday": None, "authors": [{"name": "Ismeretlen klasszikus", "bio": "A világirodalom rejtett tehetségei."}]}

    def _create_infinite_loop_video(self, image_path, output_path):
        if not MOVIEPY_AVAILABLE: return False
        try:
            print("🎥 Videó renderelés indítása...")
            clip = ImageClip(image_path).set_duration(5)
            zoomed = clip.resize(lambda t: 1 + 0.02 * t)
            zoomed = zoomed.crop(x_center=zoomed.w/2, y_center=zoomed.h/2, width=clip.w, height=clip.h)
            reversed_clip = zoomed.fx(vfx.time_mirror)
            final_clip = concatenate_videoclips([zoomed, reversed_clip])
            final_clip.write_videofile(output_path, fps=24, codec="libx264", logger=None)
            print("✅ Videó renderelés sikeres!")
            return True
        except Exception as e:
            print(f"❌ Videó hiba: {e}")
            return False

    def run_night_generation(self):
        print("🕒 [SOCIAL] Agentikus Generálás indul (V108)...")
        calendar = self._get_agentic_calendar()
        
        napi_ünnep = calendar.get("holiday")
        ünnepeltek = calendar.get("authors", [])
        
        poszt_adatai = []
        if ünnepeltek:
            for író in ünnepeltek:
                vec = self.client_ai.embeddings.create(input=író['name'], model="text-embedding-3-small").data[0].embedding
                results = self.db.collection.query(query_embeddings=[vec], n_results=2, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
                if results['ids']:
                    for i in range(len(results['ids'][0])):
                        meta = results['metadatas'][0][i]
                        if normalize_text(író['name'].split()[-1]) in normalize_text(str(meta.get('author', ''))):
                            poszt_adatai.append({"author": író['name'], "bio": író.get('bio', ''), "title": meta.get('title'), "url": meta.get('url'), "price": clean_price_raw(meta.get('price'))})
                            break

        has_author_books = len(poszt_adatai) > 0
        fallback_adatai = []

        if not has_author_books:
            print("⚠️ Nincs raktáron könyv a mai ünnepeltektől. Kincskereső bekapcsolva.")
            vec = self.client_ai.embeddings.create(input="ritka antikvár irodalom és regény kincsek", model="text-embedding-3-small").data[0].embedding
            fallback_res = self.db.collection.query(query_embeddings=[vec], n_results=3, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
            if fallback_res['ids']:
                for i in range(len(fallback_res['ids'][0])):
                    f_meta = fallback_res['metadatas'][0][i]
                    fallback_adatai.append({"author": f_meta.get('author', 'Ismeretlen'), "title": f_meta.get('title'), "url": f_meta.get('url'), "price": clean_price_raw(f_meta.get('price'))})

        if not has_author_books and not fallback_adatai:
            print("❌ Raktár teljesen üres, nincs mit ajánlani. Poszt megszakítva.")
            return

        konyv_cim = poszt_adatai[0]['title'] if has_author_books else (fallback_adatai[0]['title'] if fallback_adatai else "Antikvár ritkaságok")
        img_prompt = f"Scene from book '{konyv_cim}'. Style: Classic Dark Academia, warm vintage bookstore lighting, steaming tea, dark wood, cinematic 8k, mysterious mood. No text."
        
        video_path, image_path = "social_video.mp4", "social_img.jpg"
        media_url, is_video = "", False

        try:
            print("🎨 DALL-E Kép generálása...")
            img_res = self.client_ai.images.generate(model="dall-e-3", prompt=img_prompt, size="1024x1024", quality="hd", n=1)
            media_url = img_res.data[0].url
            with open(image_path, 'wb') as f: f.write(requests.get(media_url).content)
            
            if self._create_infinite_loop_video(image_path, video_path): 
                is_video = True
            else:
                print("⚠️ Videó készítése sikertelen, statikus képet töltünk fel.")
        except Exception as e: 
            print(f"❌ Képgenerálási hiba: {e}")

        marketing_prompt = f"""
        Act as Booksy, the CopySEO marketing agent. Write a Facebook post in Hungarian.
        Today's holiday (if any): {napi_ünnep if napi_ünnep else 'None'}.
        Today's celebrated authors and their biographies: {json.dumps(ünnepeltek, ensure_ascii=False)}.
        """
        if has_author_books:
            marketing_prompt += f"\nWe HAVE books from these authors in stock. Recommend these books: {json.dumps(poszt_adatai, ensure_ascii=False)}."
        else:
            marketing_prompt += f"\nUnfortunately, we do NOT have books from today's authors in stock right now. You MUST acknowledge this naturally (e.g., 'Sajnos a mai ünnepeltektől jelenleg minden példányunk gazdára talált...'), but highly recommend these other literary treasures instead: {json.dumps(fallback_adatai, ensure_ascii=False)}."

        marketing_prompt += "\nRule: ALWAYS start the post by commemorating the authors' birthdays and holidays (if any). Use an elegant, poetic, and marketing-savvy style. Use the exact URLs provided."
        
        post_text = self.client_ai.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": marketing_prompt}]).choices[0].message.content

        fb_page_id = os.getenv("FB_PAGE_ID")
        fb_token = os.getenv("FB_PAGE_TOKEN")
        
        if fb_page_id and fb_token:
            print("🚀 Feltöltés a Facebookra...")
            try:
                if is_video:
                    fb_url = f"https://graph.facebook.com/v19.0/{fb_page_id}/videos"
                    files = {'source': open(video_path, 'rb')}
                    data = {'access_token': fb_token, 'description': post_text, 'published': 'false'}
                    res = requests.post(fb_url, data=data, files=files)
                else:
                    fb_url = f"https://graph.facebook.com/v19.0/{fb_page_id}/photos"
                    payload = {"url": media_url, "message": post_text, "published": False, "access_token": fb_token}
                    res = requests.post(fb_url, json=payload)
                
                if res.status_code == 200:
                    print("✅ FB Draft Created Successfully!")
                else:
                    print(f"❌ FB API Hiba! Válaszkód: {res.status_code}")
                    print(f"❌ FB API Részletek: {res.text}")
            except Exception as e: 
                print(f"❌ FB Feltöltési kritikus hiba: {e}")

        with open("social_state.json", "w") as f:
            json.dump({"text": post_text, "type": "video" if is_video else "image"}, f)
        
        if os.path.exists(image_path): os.remove(image_path)
        if os.path.exists(video_path): os.remove(video_path)

    def send_morning_email(self):
        if not os.path.exists("social_state.json"): return
        with open("social_state.json", "r") as f: state = json.load(f)
        
        sender, password = os.getenv("SMTP_SENDER"), os.getenv("SMTP_PASSWORD")
        admin_emails = [e.strip() for e in os.getenv("ADMIN_EMAIL", "").split(",") if e.strip()]

        if sender and password and admin_emails:
            try:
                server = smtplib.SMTP_SSL(os.getenv("SMTP_SERVER", "mail.antikvarius.ro"), 465)
                server.login(sender, password)
                for admin in admin_emails:
                    msg = MIMEMultipart()
                    msg['From'] = f"Booksy Social Agent <{sender}>"; msg['To'] = admin
                    msg['Subject'] = "✅ Booksy: Új Facebook Vázlat Elkészült!"
                    msg.attach(MIMEText(f"<html><body><p>A mai poszt elkészült:</p><pre>{state['text']}</pre></body></html>", 'html'))
                    server.send_message(msg)
                server.quit()
            except Exception as e: print(f"❌ Email hiba: {e}")
        try: os.remove("social_state.json")
        except: pass

# --- INITIALIZATION ---
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
def home(): return {"status": "Booksy V108 (FFMPEG & MOVIEPY 1.0.3 FIXED)"}
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