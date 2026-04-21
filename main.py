# BOOKSY BRAIN - V199 (THE GOLYÓÁLLÓ EDITION)
# VERZIÓ: V199 - FLEXIBLE STOCK SYNC + LIVE CHECK + ANTI-AI SLOP + 8AM SCHED

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

# --- CONFIG & CLIENTS ---
load_dotenv()
LOCAL_TZ = pytz.timezone('Europe/Bucharest')
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
CLAUDE_MODEL = "claude-sonnet-4-6"
XML_FEED_URL = "https://www.antikvarius.ro/wp-content/uploads/woo-feed/google/xml/booksyfullfeed.xml"
TEMP_FILE = "temp_feed.xml"
SOCIAL_MEMORY_FILE = "./booksy_db/social_memory.json"

try:
    import PIL.Image
    if not hasattr(PIL.Image, 'ANTIALIAS'): 
        PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
    from moviepy.editor import ImageClip, concatenate_videoclips
    import moviepy.video.fx.all as vfx
    MOVIEPY_AVAILABLE = True
except Exception as e:
    MOVIEPY_AVAILABLE = False

# --- UTILS ---
def normalize_text(text):
    if not text: return ""
    return ''.join(c for c in unicodedata.normalize('NFD', str(text).lower()) if unicodedata.category(c) != 'Mn')

def normalize_fingerprint(text):
    if not text: return ""
    return re.sub(r'\W+', '', text).lower()

def clean_price_raw(raw_price):
    if not raw_price: return "0 RON"
    cleaned = re.sub(r"[^\d.,]", "", str(raw_price).strip())
    return f"{cleaned} RON" if cleaned else str(raw_price)

def html_to_markdown_clean(raw_html):
    if not raw_html: return ""
    try: return markdownify.markdownify(raw_html, heading_style="ATX", strip=['script', 'style']).strip()
    except: return str(raw_html)

def safe_authors_parse(text):
    try:
        soup = BeautifulSoup(text, 'html.parser')
        authors = []
        for auth in soup.find_all('author'):
            name = auth.find('name').get_text(strip=True) if auth.find('name') else ""
            nat = auth.find('nationality').get_text(strip=True) if auth.find('nationality') else "Világirodalom"
            bio = auth.find('bio').get_text(strip=True) if auth.find('bio') else ""
            if name: authors.append({"name": name, "nationality": nat, "bio": bio})
        if len(authors) > 0: return authors
    except Exception as e: print(f"⚠️ XML Parse Hiba: {e}")
    return [{"name": "Klasszikus Szerzők", "nationality": "Világirodalom", "bio": "Ma a világirodalom nagyjaira emlékezünk."}]

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

# --- DB HANDLER ---
class DBHandler:
    def __init__(self):
        if not os.path.exists("./booksy_db"): os.makedirs("./booksy_db")
        self.client = chromadb.PersistentClient(path="./booksy_db")
        self.collection = self.client.get_or_create_collection(name="booksy_collection_gemini_v2")

db_handler = DBHandler()

# --- SERVICES ---
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
        print("🚀 [SYNC] Indítás (XML -> DB)...")
        if not self.download_feed(): return
        unique_books = {}
        try:
            for _, elem in ET.iterparse(TEMP_FILE, events=("end",)):
                if elem.tag.split('}')[-1].lower() in ['item', 'post']:
                    d = {c.tag.split('}')[-1].lower(): (c.text or "") for c in elem}
                    bid = d.get('id') or d.get('post_id')
                    if bid:
                        raw_desc = f"{d.get('description', '')} {d.get('shortdescription', '')}"
                        ext = extract_metadata_from_html(raw_desc)
                        
                        # --- RUGALMAS STOCK ELLENŐRZÉS (in_stock, in stock, instock javítás) ---
                        raw_avail = d.get('availability', 'instock').lower().replace('_', '').replace(' ', '')
                        stock_status = "instock" if raw_avail == "instock" else "outofstock"
                        
                        unique_books[bid] = {
                            "id": bid, "title": d.get('title', 'Nincs cím'), "url": d.get('link', ''),
                            "image_url": d.get('image_link', ''), "price": clean_price_raw(d.get('sale_price') or d.get('price')),
                            "publisher": ext['publisher'], "author": d.get('author') or ext['author'],
                            "description": html_to_markdown_clean(raw_desc), 
                            "stock": stock_status, 
                            "type": "book"
                        }
                    elem.clear()
            
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
                    time.sleep(2.5)
            if ids:
                res = gemini_client.models.embed_content(model="gemini-embedding-001", contents=texts, config=types.EmbedContentConfig(output_dimensionality=768))
                self.db.collection.upsert(ids=ids, embeddings=[e.values for e in res.embeddings], metadatas=metas)
            print("✅ [SYNC] Kész.")
        except Exception as e: print(f"❌ SZINKRON HIBA: {e}")

class BooksyBrain:
    def __init__(self, db: DBHandler):
        self.db = db
        self.claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def process(self, msg, context_url, session_id):
        if msg.startswith("/booklink"):
            parts = msg.split()
            admin_pass = os.getenv("COMMENT_PASSWORD", "admin123")
            if len(parts) >= 2 and parts[1] == admin_pass:
                force_id = parts[2] if len(parts) >= 3 else None
                return self._trigger_fb_comment(force_id)
            else: return {"reply": "🤖 Téves parancs vagy hibás jelszó.", "products": []}

        try:
            vec = gemini_client.models.embed_content(model="gemini-embedding-001", contents=msg, config=types.EmbedContentConfig(output_dimensionality=768)).embeddings[0].values
            res = self.db.collection.query(query_embeddings=[vec], n_results=5, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
            ctx_text = ""
            prods = []
            if res['ids'] and res['ids'][0]:
                for m in res['metadatas'][0]:
                    prods.append({"title": m['title'], "price": m['price'], "url": m['url'], "image": m['image_url']})
                    ctx_text += f"- {m['title']} by {m.get('author', 'Ismeretlen')} ({m['price']})\n"
            
            r = self.claude.messages.create(
                model=CLAUDE_MODEL, max_tokens=1000,
                system="You are Booksy, the elegant Hungarian bookstore assistant.",
                messages=[{"role": "user", "content": f"Context books:\n{ctx_text}\nUser asks: {msg}"}]
            )
            return {"reply": r.content[0].text, "products": prods}
        except Exception as e: return {"reply": f"Hiba: {e}", "products": []}

    def _trigger_fb_comment(self, force_post_id=None):
        try:
            print(f"🕒 [COMMENT BOT] Triggered. Force ID: {force_post_id}")
            fb_id, fb_token = os.getenv("FB_PAGE_ID"), os.getenv("FB_PAGE_TOKEN")
            if not fb_id or not fb_token: return {"reply": "❌ FB API kulcsok hiányoznak.", "products": []}
            if not os.path.exists(SOCIAL_MEMORY_FILE): return {"reply": "❌ Nincs mentett link memória.", "products": []}
                
            with open(SOCIAL_MEMORY_FILE, "r", encoding="utf-8") as f: memory = json.load(f)
            
            target_post_id = force_post_id
            if not target_post_id:
                r = requests.get(f"https://graph.facebook.com/v19.0/{fb_id}/posts?access_token={fb_token}&limit=10")
                posts = r.json().get('data', [])
                fingerprint = normalize_fingerprint(memory.get("fingerprint", ""))
                if fingerprint:
                    for p in posts:
                        if fingerprint in normalize_fingerprint(p.get("message", "")):
                            target_post_id = p["id"]; break
            
            if not target_post_id:
                return {"reply": "❌ Nem találom a posztot ujjlenyomat alapján. Használd az ID-t: /booklink admin123 POST_ID", "products": []}

            # --- LIVE CHECK (Belesimul a komment generálásába) ---
            comment_text = "📚 A mai válogatásunk kincseit itt éred el:\n\n"
            for book in memory.get("links", []):
                res = self.db.collection.get(ids=[book.get('id', 'None')])
                status_suffix = ""
                if res['metadatas'] and res['metadatas'][0].get('stock') == 'outofstock':
                    status_suffix = " ❌ (Már el is kelt!)"
                
                comment_text += f"📖 {book['author']} - {book['title']}{status_suffix}\n🔗 {book['url']}\n\n"
            
            comment_text += "Aki kapja, marja! 😉"

            c_res = requests.post(f"https://graph.facebook.com/v19.0/{target_post_id}/comments", data={'access_token': fb_token, 'message': comment_text})
            if "id" in c_res.text: return {"reply": "✅ KÜLDETÉS TELJESÍTVE! A linkek kimentek a poszt alá.", "products": []}
            else: return {"reply": f"❌ FB hiba: {c_res.text}", "products": []}
        except Exception as e: return {"reply": f"❌ Rendszerhiba: {e}", "products": []}

class BooksySocialAgent:
    def __init__(self, db: DBHandler):
        self.db = db
        self.claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _create_video(self, img_path, out_path):
        if not MOVIEPY_AVAILABLE: return False
        try:
            clip = ImageClip(img_path).resize(width=1080)
            zoomed = clip.resize(lambda t: 1 + 0.03 * t).crop(x_center=clip.w/2, y_center=clip.h/2, width=clip.w, height=clip.h).set_duration(5)
            final = concatenate_videoclips([zoomed, zoomed.fx(vfx.time_mirror)])
            final.write_videofile(out_path, fps=24, codec="libx264", audio=False, logger=None, threads=2)
            return True
        except: return False

    def send_morning_email(self, post_text, memory_links):
        try:
            sender, password = os.getenv("SMTP_SENDER"), os.getenv("SMTP_PASSWORD")
            admin_emails = [e.strip() for e in os.getenv("ADMIN_EMAIL", "").split(",") if e.strip()]
            if not sender: return
            
            # --- TISZTA FORMÁZÁS AZ EMAILBEN (MÁSOLHATÓ) ---
            links_body = ""
            for b in memory_links: links_body += f"📖 {b['author']} - {b['title']}\n🔗 {b['url']}\n\n"

            server = smtplib.SMTP(os.getenv("SMTP_SERVER", "mail.antikvarius.ro"), 26, timeout=20)
            server.starttls(); server.login(sender, password)
            for admin in admin_emails:
                msg = MIMEMultipart()
                msg['From'] = f"Booksy AI <{sender}>"; msg['To'] = admin
                msg['Subject'] = f"✅ Booksy Social Vázlat ({datetime.now(LOCAL_TZ).strftime('%Y-%m-%d')})"
                body = f"Üdv!\n\nA FB vázlat elkészült a Drafts mappába.\n\nSZÖVEG:\n{post_text}\n\nKOMMENTBE MEGY (MÁSOLHATÓ):\n{links_body}\n\nPublikálás után: /booklink admin123"
                msg.attach(MIMEText(body, 'plain', 'utf-8'))
                server.send_message(msg)
            server.quit()
        except Exception as e: print(f"📧 Email hiba: {e}")

    def run_night_generation(self):
        print(f"🕒 [SOCIAL] Agent indul (Reggel 8:00 AM)...")
        try:
            today_date = datetime.now(LOCAL_TZ).strftime('%B %d')
            # Wiki births
            r_wiki = requests.get(f"https://en.wikipedia.org/api/rest_v1/feed/onthisday/births/{datetime.now(LOCAL_TZ).strftime('%m/%d')}", headers={'User-Agent': 'BooksyBot/1.0'})
            writers = []
            if r_wiki.status_code == 200:
                for p in r_wiki.json().get('births', []):
                    if any(kw in p.get('text', '').lower() for kw in ['writer', 'author', 'poet']): writers.append(p)
            
            prompt_wiki = f"Today is {today_date}. Provide EXACTLY 6 writers born today (inc. Hungarian/Romanian) in XML format <authors><author><name>...</name><nationality>...</nationality><bio>...</bio></author></authors>."
            r_claude_wiki = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=800, messages=[{"role": "user", "content": prompt_wiki}])
            authors_list = safe_authors_parse(r_claude_wiki.content[0].text)[:6]

            selected_books, seen_ids = [], set()
            for author in authors_list:
                vec = gemini_client.models.embed_content(model="gemini-embedding-001", contents=author['name'], config=types.EmbedContentConfig(output_dimensionality=768)).embeddings[0].values
                res = self.db.collection.query(query_embeddings=[vec], n_results=3, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
                if res['ids'] and res['ids'][0]:
                    for p_target in res['metadatas'][0]:
                        if p_target['id'] not in seen_ids:
                            selected_books.append(p_target); seen_ids.add(p_target['id']); break

            # FALLBACK: Ha nincs elég könyv az íróktól, feltöltjük népszerűekkel (MOST MÁR LÁTNI FOGJA AZ IN_STOCK-OT IS)
            if len(selected_books) < 4:
                vec_fb = gemini_client.models.embed_content(model="gemini-embedding-001", contents="népszerű klasszikus és modern irodalom", config=types.EmbedContentConfig(output_dimensionality=768)).embeddings[0].values
                res_fb = self.db.collection.query(query_embeddings=[vec_fb], n_results=10, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
                if res_fb['ids'] and res_fb['ids'][0]:
                    for p_target in res_fb['metadatas'][0]:
                        if p_target['id'] not in seen_ids:
                            selected_books.append(p_target); seen_ids.add(p_target['id'])
                        if len(selected_books) >= 5: break

            main_book = selected_books[0]
            vision_prompt = f"Atmospheric cinematic image prompt for '{main_book['title']}' by {main_book['author']}. Core setting mood. Vertical 9:16."
            p_img = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=200, messages=[{"role": "user", "content": vision_prompt}]).content[0].text
            img_path, vid_path = "social_img.jpg", "social_video.mp4"
            with open(img_path, 'wb') as f: f.write(requests.get(f"https://image.pollinations.ai/prompt/{urllib.parse.quote(p_img)}?width=1080&height=1920&nologo=true").content)
            has_video = self._create_video(img_path, vid_path)
            
            authors_text = "\n".join([f"📖 {a['name']} ({a.get('nationality', 'Világirodalom')}): {a['bio']}" for a in authors_list])
            books_context = "\n".join([f"- {b['title']} by {b['author']}" for b in selected_books])

            # --- ANTI-AI SLOP SYSTEM PROMPT (SZIGORÚ EMBERI TÓNUS) ---
            final_prompt = (
                f"Írj egy posztot az Antikvarius.ro FB oldalára. STÍLUS ÉS SZABÁLYOK:\n"
                f"- Tónus: Végtelenül emberi, természetes, mintha egy barátodnak ajánlanád. Kerüld az AI-szagú szavakat és a gépies lelkesedést.\n"
                f"- Emoji diéta: Csak 1-2 emojit használj, ott ahol tényleg fontos. Ne spammeld.\n"
                f"- Első sor dőlt betűvel: *— ✨ Érzés: [Válassz egyet] —*\n"
                f"- Kötelező: A 'link a kommentben' mondat legvégére tegyél egy lefelé mutató ujjat (👇)!\n\n"
                f"Szerzők: {authors_text}\nKönyvek: {books_context}"
            )
            post_text = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=1500, system="You are an expert human bookstore curator. You despise AI clichés and formal corporate tone.", messages=[{"role": "user", "content": final_prompt}]).content[0].text

            raw_fingerprint = post_text[30:130] if len(post_text) > 130 else post_text[:100]
            memory_data = {
                "fingerprint": raw_fingerprint,
                "links": [{"id": b['id'], "title": b['title'], "author": b['author'], "url": b['url']} for b in selected_books]
            }
            with open(SOCIAL_MEMORY_FILE, "w", encoding="utf-8") as f: json.dump(memory_data, f, ensure_ascii=False)

            fb_id, fb_token = os.getenv("FB_PAGE_ID"), os.getenv("FB_PAGE_TOKEN")
            if fb_id and fb_token:
                api_data = {'access_token': fb_token, 'message': post_text, 'published': 'false', 'unpublished_content_type': 'DRAFT'}
                if has_video:
                    v_data = api_data.copy(); v_data['description'] = v_data.pop('message')
                    requests.post(f"https://graph.facebook.com/v19.0/{fb_id}/videos", data=v_data, files={'source': open(vid_path, 'rb')})
                else:
                    r_p = requests.post(f"https://graph.facebook.com/v19.0/{fb_id}/photos", data={'access_token': fb_token, 'published': 'true', 'no_story': 'true'}, files={'source': open(img_path, 'rb')})
                    mid = r_p.json().get('id')
                    if mid: 
                        api_data['attached_media'] = json.dumps([{'media_fbid': mid}])
                        requests.post(f"https://graph.facebook.com/v19.0/{fb_id}/feed", data=api_data)
            
            self.send_morning_email(post_text, memory_data['links'])
            for p in [img_path, vid_path]:
                if os.path.exists(p): os.remove(p)
            print("✅ [SOCIAL] Kész.")
        except Exception as e: print(f"❌ HIBA: {e}")

# --- FASTAPI ---
updater = AutoUpdater(db_handler); bot = BooksyBrain(db_handler); social_agent = BooksySocialAgent(db_handler); scheduler = BackgroundScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # SZIGORÚ SORREND: 3:00 Szinkronizáció, 8:00 Poszt generálás
    scheduler.add_job(updater.run_daily_update, CronTrigger(hour=3, minute=0, timezone=LOCAL_TZ))
    scheduler.add_job(social_agent.run_night_generation, CronTrigger(hour=8, minute=0, timezone=LOCAL_TZ))
    scheduler.start(); yield; scheduler.shutdown()

app = FastAPI(lifespan=lifespan); app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_headers=["*"], allow_methods=["*"])
class ChatRequest(BaseModel): message: str; context_url: Optional[str] = ""; session_id: Optional[str] = ""
class InitRequest(BaseModel): url: str; session_id: str; ui_lang: str = "hu"
@app.get("/")
def home(): return {"status": "V199 Online", "project": "Booksy"}
@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url, req.session_id)
@app.post("/init-chat")
def init_chat(req: InitRequest): return {"ui_lang": req.ui_lang, "bubble_text": "Szia!", "placeholder": "Keresel valamit?"}

# --- TEST ENDPOINT (SZIGORÚ SORRENDDEL: SYNC -> GEN) ---
@app.post("/test-social-night")
def test_night(bt: BackgroundTasks): 
    def full_workflow():
        updater.run_daily_update() # Először frissítjük a DB-t az XML-ből
        social_agent.run_night_generation() # Utána generálunk posztot
    bt.add_task(full_workflow)
    return {"status": "Full Sync & Generation Started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)