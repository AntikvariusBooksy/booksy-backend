# BOOKSY BRAIN - V203 (THE AGENTIC VISION EDITION)
# VERZIÓ: V203 - GEMINI IMAGEN 3 + CLAUDE PROMPT AGENT + 15S COMMENT DELAY
# MEGJEGYZÉS: ADATBÁZIS SZINKRON ÁTMENETILEG KIKAPCSOLVA A TESZTHEZ.

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

# --- CONFIG & CLIENTS ---
load_dotenv()
LOCAL_TZ = pytz.timezone('Europe/Bucharest')
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
SOCIAL_MEMORY_FILE = "./booksy_db/social_memory.json"

try:
    import PIL.Image
    import PIL.ImageOps
    from moviepy.editor import ImageClip, concatenate_videoclips
    import moviepy.video.fx.all as vfx
    MOVIEPY_AVAILABLE = True
except Exception as e:
    MOVIEPY_AVAILABLE = False

# --- UTILS ---
def log_event(msg):
    now = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 🤖 {msg}")

def normalize_fingerprint(text):
    if not text: return ""
    return re.sub(r'\W+', '', text).lower()

# --- DB HANDLER ---
class DBHandler:
    def __init__(self):
        if not os.path.exists("./booksy_db"): os.makedirs("./booksy_db")
        self.client = chromadb.PersistentClient(path="./booksy_db")
        self.collection = self.client.get_or_create_collection(name="booksy_collection_gemini_v2")

db_handler = DBHandler()

# --- SERVICES ---
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
        return {"reply": "Chat funkció aktív.", "products": []}

    def _trigger_fb_comment(self, force_post_id=None):
        try:
            log_event(f"Indítás: FB Komment Bot. Force ID: {force_post_id}")
            log_event("Várakozás 15 másodpercig (Meta Cache Sync)...")
            time.sleep(15)

            fb_id, fb_token = os.getenv("FB_PAGE_ID"), os.getenv("FB_PAGE_TOKEN")
            if not os.path.exists(SOCIAL_MEMORY_FILE): return {"reply": "❌ Nincs memória fájl.", "products": []}
            with open(SOCIAL_MEMORY_FILE, "r", encoding="utf-8") as f: memory = json.load(f)
            
            target_post_id = force_post_id
            if not target_post_id:
                log_event("Poszt keresése ujjlenyomat alapján...")
                r = requests.get(f"https://graph.facebook.com/v19.0/{fb_id}/posts?access_token={fb_token}&limit=10")
                posts = r.json().get('data', [])
                fingerprint = normalize_fingerprint(memory.get("fingerprint", ""))
                for p in posts:
                    if fingerprint in normalize_fingerprint(p.get("message", "")):
                        target_post_id = p["id"]; break
            
            if not target_post_id: return {"reply": "❌ Poszt nem található.", "products": []}

            # LIVE CHECK & SINGLE COMMENT
            comment_text = "📚 A mai válogatásunk kincseit itt éred el:\n\n"
            for book in memory.get("links", []):
                res = self.db.collection.get(ids=[book.get('id', 'None')])
                status = " ❌ (Már el is kelt!)" if (res['metadatas'] and res['metadatas'][0].get('stock') == 'outofstock') else ""
                author = f"{book['author']} - " if (book.get('author') and book['author'] != 'Ismeretlen') else ""
                comment_text += f"📖 {author}{book['title']}{status}\n🔗 {book['url']}\n\n"
            
            comment_text += "Aki kapja, marja! 😉"
            log_event(f"Komment küldése a {target_post_id} ID-ra...")
            c_res = requests.post(f"https://graph.facebook.com/v19.0/{target_post_id}/comments", data={'access_token': fb_token, 'message': comment_text})
            
            if "id" in c_res.text:
                log_event("Komment sikeresen elküldve.")
                return {"reply": "✅ Komment sikeresen kiment!", "products": []}
            else:
                log_event(f"FB Hiba: {c_res.text}")
                return {"reply": f"❌ FB hiba: {c_res.text}", "products": []}
        except Exception as e:
            log_event(f"Rendszerhiba: {e}")
            return {"reply": f"❌ Hiba: {e}", "products": []}

class BooksySocialAgent:
    def __init__(self, db: DBHandler):
        self.db = db
        self.claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _create_video(self, img_path, out_path):
        if not MOVIEPY_AVAILABLE: return False
        try:
            log_event("Videó renderelés indítása (1920x1920 square)...")
            clip = ImageClip(img_path).set_duration(5)
            # 3% zoom effektus
            zoomed = clip.resize(lambda t: 1 + 0.03 * t).set_position('center').set_duration(5)
            final = concatenate_videoclips([zoomed, zoomed.fx(vfx.time_mirror)])
            final.write_videofile(out_path, fps=24, codec="libx264", audio=False, logger=None)
            return True
        except Exception as e:
            log_event(f"Videó hiba: {e}")
            return False

    def run_night_generation(self):
        log_event("Agentic Generálás indítása (Reggel 8:00)...")
        try:
            # 1. Könyv válogatás a meglévő DB-ből
            res = self.db.collection.get(limit=10, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
            if not res['metadatas']: 
                log_event("Hiba: Adatbázis üres vagy nincs raktáron lévő könyv.")
                return
            
            # Véletlenszerű válogatás a diverzitásért
            sample = random.sample(res['metadatas'], min(5, len(res['metadatas'])))
            main_book = sample[0]
            log_event(f"Fő könyv kiválasztva: {main_book['title']}")

            # --- AGENTIC VISUAL PIPELINE ---
            
            # STEP 1: Gemini elemzés (Text)
            log_event("Step 1: Gemini könyv-elemzés indítása...")
            analysis_prompt = f"""Elemezd ki ezt a könyvet: '{main_book['title']}' írta {main_book.get('author','valaki')}. 
            Leírás: {main_book.get('text_preview','')}.
            Határozd meg a műfaját, a vizuális motívumait, a korabeli környezetet és az alapvető hangulatát. 
            Adj egy tömör vizuális összefoglalót angolul!"""
            
            gem_res = gemini_client.models.generate_content(model="gemini-2.0-flash", contents=[analysis_prompt])
            visual_context = gem_res.text
            log_event("Gemini elemzés kész.")

            # STEP 2: Claude Cinematic Prompt
            log_event("Step 2: Claude vizuális rendezői prompt generálás...")
            claude_prompt = f"""Te egy filmes látványtervező vagy. Az alábbi elemzés alapján írj egy képgenerálási promptot:
            KONTEXTUS: {visual_context}
            SZABÁLYOK:
            - Ne mutass könyveket vagy könyvtárat, hanem a történet VILÁGÁT.
            - Stílus: Hyper-realistic, cinematic, atmospheric lighting.
            - Arány: 1:1 Square.
            - Nincs szöveg vagy betű a képen.
            - Csak a promptot küldd vissza angolul!"""
            
            c_res = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=300, messages=[{"role": "user", "content": claude_prompt}])
            final_img_prompt = c_res.content[0].text
            log_event(f"Claude prompt kész: {final_img_prompt[:50]}...")

            # STEP 3: Gemini Imagen 3 Generation
            log_event("Step 3: Google Imagen 3 képgenerálás (1920x1920)...")
            img_path = "social_img.jpg"
            img_response = gemini_client.models.generate_images(
                model='imagen-3.0-generate-001',
                prompt=final_img_prompt,
                config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio='1:1')
            )
            img_response.generated_images[0].image.save(img_path)
            
            # Szoftveres kényszerítés 1920x1920-ra
            img_obj = PIL.Image.open(img_path)
            img_resized = PIL.ImageOps.fit(img_obj, (1920, 1920), PIL.Image.Resampling.LANCZOS)
            img_resized.save(img_path)
            log_event("Kép mentve és méretezve.")

            # Videó és FB poszt folyamat
            vid_path = "social_video.mp4"
            has_video = self._create_video(img_path, vid_path)

            # Szövegírás (Anti-AI Slop)
            log_event("Poszt szöveg generálása...")
            text_prompt = f"Írj egy emberi posztot. Fő könyv: {main_book['title']}. Egyéb könyvek: {', '.join([b['title'] for b in sample[1:]])}. 👇 a végére."
            post_text = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=1000, system="Human bookstore curator tone. No AI slop.", messages=[{"role": "user", "content": text_prompt}]).content[0].text

            # Mentés memóriába
            memory_data = {
                "fingerprint": post_text[:100],
                "links": [{"id": b['id'], "title": b['title'], "author": b['author'], "url": b['url']} for b in sample]
            }
            with open(SOCIAL_MEMORY_FILE, "w", encoding="utf-8") as f: json.dump(memory_data, f, ensure_ascii=False)

            # Feltöltés (DRAFT)
            fb_id, fb_token = os.getenv("FB_PAGE_ID"), os.getenv("FB_PAGE_TOKEN")
            api_data = {'access_token': fb_token, 'message': post_text, 'published': 'false', 'unpublished_content_type': 'DRAFT'}
            if has_video:
                v_data = api_data.copy(); v_data['description'] = v_data.pop('message')
                requests.post(f"https://graph.facebook.com/v19.0/{fb_id}/videos", data=v_data, files={'source': open(vid_path, 'rb')})
                log_event("Videó vázlat feltöltve.")
            
            log_event("Folyamat sikeresen lezárult.")
        except Exception as e:
            log_event(f"KRITIKUS HIBA: {e}")

# --- FASTAPI ---
bot = BooksyBrain(db_handler); social_agent = BooksySocialAgent(db_handler); scheduler = BackgroundScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # SZINKRON KIKAPCSOLVA A TESZTHEZ
    scheduler.add_job(social_agent.run_night_generation, CronTrigger(hour=8, minute=0, timezone=LOCAL_TZ))
    scheduler.start(); yield; scheduler.shutdown()

app = FastAPI(lifespan=lifespan); app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_headers=["*"], allow_methods=["*"])
class ChatRequest(BaseModel): message: str; context_url: Optional[str] = ""; session_id: Optional[str] = ""
@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url, req.session_id)

@app.post("/test-social-night")
def test_night(bt: BackgroundTasks): 
    # CSÖKKENTETT TESZT: Szinkron nélkül, csak generálás
    bt.add_task(social_agent.run_night_generation)
    return {"status": "V203 Agentic Test Started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)