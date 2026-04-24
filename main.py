# BOOKSY BRAIN - V225 (THE OMNI-RADAR EDITION)
# VERZIÓ: V225 - DEEP ATTACHMENT URL SEARCH ACROSS PUBLISHED_POSTS, FEED, AND POSTS

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os, time, requests, hashlib, re, json, random, unicodedata, html, urllib.parse, gc, chromadb, pytz, smtplib
import numpy as np
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

CLAUDE_MODEL = "claude-sonnet-4-6"
XML_FEED_URL = "https://www.antikvarius.ro/wp-content/uploads/woo-feed/google/xml/booksyfullfeed.xml"
TEMP_FILE = "temp_feed.xml"
SOCIAL_MEMORY_FILE = "./booksy_db/social_memory.json"

try:
    import PIL.Image
    if not hasattr(PIL.Image, 'ANTIALIAS'): 
        PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
    import PIL.ImageOps
    from PIL import ImageDraw, ImageFont
    from moviepy.editor import ImageClip, concatenate_videoclips, CompositeVideoClip
    import moviepy.video.fx.all as vfx
    MOVIEPY_AVAILABLE = True
except Exception as e:
    MOVIEPY_AVAILABLE = False

# --- UTILS ---
def log_event(msg):
    now = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 🤖 {msg}")

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
    except Exception as e: log_event(f"⚠️ XML Parse Hiba: {e}")
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
            if r.status_code != 200: return False
            with open(TEMP_FILE, 'wb') as f:
                for chunk in r.iter_content(8192): f.write(chunk)
            return True
        except: return False

    def run_daily_update(self):
        log_event("🚀 [SYNC] Indítás (XML -> DB)...")
        if not self.download_feed(): 
            log_event("❌ SZINKRON HIBA: Nem sikerült letölteni az XML Feed-et.")
            return False
            
        unique_books = {}
        try:
            for _, elem in ET.iterparse(TEMP_FILE, events=("end",)):
                if elem.tag.split('}')[-1].lower() in ['item', 'post']:
                    d = {c.tag.split('}')[-1].lower(): (c.text or "") for c in elem}
                    bid = d.get('id') or d.get('post_id')
                    if bid:
                        raw_desc = f"{d.get('description', '')} {d.get('shortdescription', '')}"
                        ext = extract_metadata_from_html(raw_desc)
                        
                        clean_title = html.unescape(d.get('title', 'Nincs cím'))
                        clean_author = html.unescape(d.get('author') or ext['author'])
                        
                        raw_avail = d.get('availability', 'instock').lower().replace('_', '').replace(' ', '')
                        stock_status = "instock" if raw_avail == "instock" else "outofstock"
                        
                        unique_books[bid] = {
                            "id": bid, "title": clean_title, "url": d.get('link', ''),
                            "image_url": d.get('image_link', ''), "price": clean_price_raw(d.get('sale_price') or d.get('price')),
                            "publisher": ext['publisher'], "author": clean_author,
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
            log_event("✅ [SYNC] Kész.")
            return True
        except Exception as e: 
            log_event(f"❌ SZINKRON HIBA (Feldolgozás): {e}")
            return False

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
        return {"reply": "Chat funkció aktív. Jelenleg a Social Agent fut.", "products": []}

    def _trigger_fb_comment(self, force_post_id=None):
        try:
            log_event(f"Indítás: FB Komment Bot. Force ID: {force_post_id}")

            fb_id, fb_token = os.getenv("FB_PAGE_ID"), os.getenv("FB_PAGE_TOKEN")
            if not os.path.exists(SOCIAL_MEMORY_FILE): return {"reply": "❌ Nincs memória fájl.", "products": []}
            with open(SOCIAL_MEMORY_FILE, "r", encoding="utf-8") as f: memory = json.load(f)
            
            if not memory.get("links") or len(memory["links"]) == 0:
                return {"reply": "❌ Memória fájl hibás, nincs könyv.", "products": []}

            target_post_id = force_post_id
            
            if not target_post_id:
                media_id = memory.get("media_id")
                search_title = normalize_fingerprint(memory["links"][0].get("title", "")) if memory.get("links") else ""
                
                log_event("Omni-Radar: Publikus poszt/Reels keresése az összes végponton...")
                
                # A 3 legerősebb Meta végpont a posztok megtalálására (A description mező törölve a biztonságért)
                endpoints = [
                    f"https://graph.facebook.com/v19.0/{fb_id}/published_posts?access_token={fb_token}&limit=15&fields=id,message,attachments",
                    f"https://graph.facebook.com/v19.0/{fb_id}/feed?access_token={fb_token}&limit=15&fields=id,message,attachments",
                    f"https://graph.facebook.com/v19.0/{fb_id}/posts?access_token={fb_token}&limit=15&fields=id,message,attachments"
                ]
                
                found = False
                for ep in endpoints:
                    if found: break
                    try:
                        r = requests.get(ep)
                        if r.status_code != 200: continue
                        posts = r.json().get('data', [])
                        
                        for p in posts:
                            # 1. Keresés mély URL röntgennel a Media ID alapján
                            if media_id:
                                atts = p.get('attachments', {}).get('data', [])
                                for att in atts:
                                    target = att.get('target', {})
                                    # Szöveggé alakítjuk az URL-eket, hogy megkeressük bennük az ID-t
                                    target_id = str(target.get('id', ''))
                                    target_url = str(target.get('url', ''))
                                    att_url = str(att.get('url', ''))
                                    
                                    if target_id == str(media_id) or str(media_id) in target_url or str(media_id) in att_url:
                                        found = True
                                        break
                                    
                                    # Mélyebb rétegek (subattachments - sokszor itt van a videó elrejtve)
                                    for sub in att.get('subattachments', {}).get('data', []):
                                        sub_target = sub.get('target', {})
                                        if str(sub_target.get('id', '')) == str(media_id) or str(media_id) in str(sub_target.get('url', '')) or str(media_id) in str(sub.get('url', '')):
                                            found = True
                                            break
                            
                            # 2. Biztonsági keresés: Cím alapján (Ha az API törölte volna a csatolmányt)
                            if not found and search_title:
                                msg = normalize_fingerprint(p.get("message", ""))
                                if search_title in msg:
                                    found = True
                            
                            if found:
                                target_post_id = p["id"]
                                ep_name = ep.split('?')[0].split('/')[-1]
                                log_event(f"✅ Találat a '{ep_name}' végponton! Poszt azonosítója: {target_post_id}")
                                break
                    except Exception as loop_e:
                        log_event(f"Hiba a végpont ellenőrzésénél: {loop_e}")
            
            if not target_post_id: 
                err_msg = "❌ Célpont poszt (vagy Reels) nem található a publikus hírfolyamban. Biztosan közzétetted a vázlatot?"
                log_event(err_msg)
                return {"reply": err_msg, "products": []}

            comment_text = "📚 A mai válogatásunk kincseit itt éred el:\n\n"
            for book in memory.get("links", []):
                res = self.db.collection.get(ids=[book.get('id', 'None')])
                status = " ❌ (Már el is kelt!)" if (res['metadatas'] and res['metadatas'][0].get('stock') == 'outofstock') else ""
                author = f"{book['author']} - " if (book.get('author') and book['author'] != 'Ismeretlen') else ""
                comment_text += f"📖 {author}{book['title']}{status}\n🔗 {book['url']}\n\n"
            
            comment_text += "Aki kapja, marja! 😉"
            log_event(f"Komment küldése a(z) {target_post_id} azonosítóra...")
            c_res = requests.post(f"https://graph.facebook.com/v19.0/{target_post_id}/comments", data={'access_token': fb_token, 'message': comment_text})
            
            if "id" in c_res.text:
                log_event("✅ Komment sikeresen rögzítve a poszthoz/Reels-hez.")
                return {"reply": "✅ Komment sikeresen kiment!", "products": []}
            else:
                log_event(f"FB Hiba: {c_res.text}")
                return {"reply": f"❌ FB hiba a kommentelésnél: {c_res.text}", "products": []}
        except Exception as e:
            log_event(f"Rendszerhiba: {e}")
            return {"reply": f"❌ Hiba: {e}", "products": []}

class BooksySocialAgent:
    def __init__(self, db: DBHandler):
        self.db = db
        self.claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def send_morning_email(self, post_text, memory_links):
        try:
            sender, password = os.getenv("SMTP_SENDER"), os.getenv("SMTP_PASSWORD")
            admin_emails = [e.strip() for e in os.getenv("ADMIN_EMAIL", "").split(",") if e.strip()]
            if not sender: return
            
            links_body = ""
            for b in memory_links:
                author_display = f"{b['author']} - " if b.get('author') and b['author'] != 'Ismeretlen' else ""
                links_body += f"📖 {author_display}{b['title']}\n🔗 {b['url']}\n\n"

            server = smtplib.SMTP(os.getenv("SMTP_SERVER", "mail.antikvarius.ro"), 26, timeout=20)
            server.starttls(); server.login(sender, password)
            for admin in admin_emails:
                msg = MIMEMultipart()
                msg['From'] = f"Booksy AI <{sender}>"; msg['To'] = admin
                msg['Subject'] = f"✅ Booksy Social Vázlat ({datetime.now(LOCAL_TZ).strftime('%Y-%m-%d')})"
                body = f"Üdv!\n\nA FB vázlat elkészült a Drafts mappába.\n\nMiután Business Suite-ban rákattintottál a Publikálás gombra, a chatben használd a /booklink admin123 parancsot a kommenthez!\n\nSZÖVEG:\n{post_text}\n\nKOMMENTBE MEGY (MÁSOLHATÓ):\n{links_body}"
                msg.attach(MIMEText(body, 'plain', 'utf-8'))
                server.send_message(msg)
            server.quit()
            log_event("Értesítő e-mail sikeresen elküldve.")
        except Exception as e: log_event(f"📧 Email hiba: {e}")

    def _prepare_visual_layers(self, raw_img_path, overlay_path, fallback_path, title, author):
        try:
            log_event("Vizuális rétegek (Overlay és Fallback) generálása Bounding Box algoritmussal...")
            img = PIL.Image.open(raw_img_path).convert("RGBA")
            width, height = img.size

            # KÖTELEZŐ: Manuálisan feltöltött lokális TTF használata!
            font_path = "Montserrat-Bold.ttf"
            use_bbox = False
            best_font = ImageFont.load_default()

            if os.path.exists(font_path):
                use_bbox = True
            else:
                log_event("❌ FIGYELEM: A 'Montserrat-Bold.ttf' nem található a gyökérkönyvtárban! Kérlek töltsd fel a Git-be. Alapértelmezett apró betű aktív.")

            overlay = PIL.Image.new('RGBA', (width, height), (0,0,0,0))
            draw = ImageDraw.Draw(overlay)
            
            bar_height = int(height * 0.15) 
            bar_y = height - bar_height
            draw.rectangle([(0, bar_y), (width, height)], fill=(0, 0, 0, 180)) 
            
            author_text = f"{author} - " if author and author != "Ismeretlen" else ""
            full_text = f"{author_text}{title}"

            if use_bbox:
                target_width = int(width * 0.80)
                target_height = int(bar_height * 0.60)
                font_size = 10
                
                try:
                    while True:
                        test_font = ImageFont.truetype(font_path, font_size + 1)
                        bbox = test_font.getbbox(full_text)
                        text_w = bbox[2] - bbox[0]
                        text_h = bbox[3] - bbox[1]
                        
                        if text_w > target_width or text_h > target_height:
                            break
                        font_size += 1
                        best_font = test_font
                except Exception as e:
                    log_event(f"Betűtípus méretezési hiba: {e}")
                
            bbox = best_font.getbbox(full_text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            x_center = (width - text_w) / 2
            y_center = bar_y + (bar_height - text_h) / 2 - bbox[1]

            draw.text((x_center, y_center), full_text, font=best_font, fill=(255, 255, 255, 255))

            overlay.save(overlay_path, "PNG")
            
            combined = PIL.Image.alpha_composite(img, overlay)
            combined.convert("RGB").save(fallback_path, "JPEG")
            
            log_event("Rétegek és tipográfia sikeresen mentve.")
            return True
        except Exception as e:
            log_event(f"Hiba a vizuális rétegek készítésénél: {e}")
            return False

    def _create_video(self, raw_img_path, overlay_path, out_path):
        if not MOVIEPY_AVAILABLE: return False
        try:
            log_event("Videó renderelés indítása (Ipari Frame-by-Frame Stamping)...")
            
            clip = ImageClip(raw_img_path).set_duration(5)
            zoomed = clip.resize(lambda t: 1 + 0.03 * t).set_position('center')
            fixed_bg = CompositeVideoClip([zoomed], size=(1920, 1920)).set_duration(5)
            bg_loop = concatenate_videoclips([fixed_bg, fixed_bg.fx(vfx.time_mirror)])
            
            if os.path.exists(overlay_path):
                overlay_img = PIL.Image.open(overlay_path).convert("RGBA")
                
                def stamp_overlay(frame_array):
                    clean_frame = np.clip(frame_array, 0, 255).astype(np.uint8)
                    pil_frame = PIL.Image.fromarray(clean_frame).convert("RGBA")
                    pil_frame.alpha_composite(overlay_img)
                    return np.array(pil_frame.convert("RGB"))
                
                final_video = bg_loop.fl_image(stamp_overlay)
            else:
                final_video = bg_loop

            final_video.write_videofile(out_path, fps=24, codec="libx264", audio=False, ffmpeg_params=["-pix_fmt", "yuv420p"], logger=None)
            return True
        except Exception as e:
            log_event(f"Videó hiba: {e}")
            return False

    def run_night_generation(self):
        log_event("Agentic Generálás indítása...")
        raw_img_path = "social_raw.jpg"
        overlay_path = "social_overlay.png"
        fallback_img_path = "social_fallback.jpg"
        vid_path = "social_video.mp4"
        
        try:
            today_date = datetime.now(LOCAL_TZ).strftime('%B %d')
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

            if len(selected_books) < 4:
                vec_fb = gemini_client.models.embed_content(model="gemini-embedding-001", contents="népszerű klasszikus és modern irodalom", config=types.EmbedContentConfig(output_dimensionality=768)).embeddings[0].values
                res_fb = self.db.collection.query(query_embeddings=[vec_fb], n_results=10, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
                if res_fb['ids'] and res_fb['ids'][0]:
                    for p_target in res_fb['metadatas'][0]:
                        if p_target['id'] not in seen_ids:
                            selected_books.append(p_target); seen_ids.add(p_target['id'])
                        if len(selected_books) >= 5: break

            main_book = selected_books[0]
            log_event(f"Fő könyv kiválasztva: {main_book['title']}")

            log_event("Step 1: Gemini (2.5-flash) könyv-elemzés indítása...")
            analysis_prompt = f"""Elemezd ki ezt a könyvet: '{main_book['title']}' írta {main_book.get('author','valaki')}. Leírás: {main_book.get('text_preview','')}. Határozd meg a műfaját, a vizuális motívumait, a korabeli környezetet és az alapvető hangulatát. Adj egy tömör vizuális összefoglalót angolul!"""
            gem_res = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=[analysis_prompt])
            log_event("Gemini elemzés kész.")

            log_event("Step 2: Claude vizuális rendezői prompt generálás...")
            claude_prompt = f"""Te egy filmes látványtervező vagy. Az alábbi elemzés alapján írj képgenerálási promptot. KONTEXTUS: {gem_res.text} SZABÁLYOK: Ne mutass könyveket. Stílus: Hyper-realistic, cinematic. Arány: 1:1 Square. Nincs szöveg. Csak a promptot küldd!"""
            c_res = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=300, messages=[{"role": "user", "content": claude_prompt}])
            final_img_prompt = c_res.content[0].text
            log_event(f"Claude prompt kész.")

            log_event("Step 3: Flux API képgenerálás (1920x1920)...")
            flux_url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(final_img_prompt)}?width=1920&height=1920&nologo=true&model=flux"
            r_img = requests.get(flux_url, timeout=90)
            if r_img.status_code == 200:
                with open(raw_img_path, 'wb') as f: f.write(r_img.content)
            else:
                raise Exception(f"Flux generálási hiba HTTP {r_img.status_code}")
            
            img_obj = PIL.Image.open(raw_img_path)
            img_resized = PIL.ImageOps.fit(img_obj, (1920, 1920), PIL.Image.Resampling.LANCZOS)
            img_resized.save(raw_img_path)
            
            self._prepare_visual_layers(raw_img_path, overlay_path, fallback_img_path, main_book['title'], main_book.get('author', ''))
            has_video = self._create_video(raw_img_path, overlay_path, vid_path)

            log_event("Poszt szöveg generálása (Anti-AI Slop)...")
            authors_text = "\n".join([f"📖 {a['name']} ({a.get('nationality', 'Világirodalom')}): {a['bio']}" for a in authors_list])
            books_context = "\n".join([f"- {b['title']} by {b['author']}" for b in selected_books])

            text_prompt = (
                f"Írj egy posztot az Antikvarius.ro FB oldalára. STÍLUS:\n"
                f"- Tónus: Végtelenül emberi, kerüld a marketinges blablát és az AI sablonokat.\n"
                f"- Emoji diéta: Csak 1-2 emojit használj, ott ahol fontos.\n"
                f"- Első sor dőlt betűvel: *— ✨ Érzés: [Válassz egyet] —*\n"
                f"- Kötelező: A 'link a kommentben' mondat legvégére tegyél egy 👇 emojit!\n\n"
                f"Szerzők: {authors_text}\nKönyvek: {books_context}"
            )
            post_text = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=1000, system="Human bookstore curator tone. No AI slop.", messages=[{"role": "user", "content": text_prompt}]).content[0].text

            memory_data = {
                "fingerprint": post_text[:100],
                "links": [{"id": b['id'], "title": b['title'], "author": b['author'], "url": b['url']} for b in selected_books]
            }

            fb_id, fb_token = os.getenv("FB_PAGE_ID"), os.getenv("FB_PAGE_TOKEN")
            
            # --- VISSZAÁLLÍTOTT EREDETI VIDEÓ FELTÖLTÉS (MŰKÖDŐ VERZIÓBÓL) ---
            api_data = {'access_token': fb_token, 'message': post_text, 'published': 'false', 'unpublished_content_type': 'DRAFT'}
            
            if has_video:
                log_event("Videó vázlat feltöltése a bevált, eredeti protokollal...")
                v_data = api_data.copy()
                v_data['description'] = v_data.pop('message')
                r_v = requests.post(f"https://graph.facebook.com/v19.0/{fb_id}/videos", data=v_data, files={'source': open(vid_path, 'rb')})
                if r_v.status_code == 200:
                    vid_id = str(r_v.json().get('id'))
                    memory_data['media_id'] = vid_id
                    log_event(f"✅ Videó vázlat SIKERESEN feltöltve a Facebookra! (Media ID: {vid_id})")
                else:
                    log_event(f"⚠️ FB Videó hiba: {r_v.text}")
            else:
                upload_img = fallback_img_path if os.path.exists(fallback_img_path) else raw_img_path
                log_event("Videó hiányában a képes vázlat kerül feltöltésre...")
                r_p = requests.post(f"https://graph.facebook.com/v19.0/{fb_id}/photos", data={'access_token': fb_token, 'published': 'false'}, files={'source': open(upload_img, 'rb')})
                if r_p.status_code == 200:
                    photo_id = str(r_p.json().get('id'))
                    memory_data['media_id'] = photo_id
                    log_event(f"✅ Képes vázlat SIKERESEN feltöltve a Facebookra! (Media ID: {photo_id})")
                else:
                    log_event(f"❌ FB Kép hiba: {r_p.text}")
            
            with open(SOCIAL_MEMORY_FILE, "w", encoding="utf-8") as f: 
                json.dump(memory_data, f, ensure_ascii=False)
                
            self.send_morning_email(post_text, memory_data['links'])
            log_event("Folyamat sikeresen lezárult.")
            
            for p in [raw_img_path, overlay_path, fallback_img_path, vid_path]:
                if os.path.exists(p): os.remove(p)

        except Exception as e:
            log_event(f"KRITIKUS HIBA: {e}")

# --- MASTER CASCADE ROUTINE ---
updater = AutoUpdater(db_handler); bot = BooksyBrain(db_handler); social_agent = BooksySocialAgent(db_handler); scheduler = BackgroundScheduler()

def master_morning_routine():
    log_event("🌅 Master Láncreakció Indítása: DB Sync (KIKAPCSOLVA A TESZTHEZ) -> Social Post")
    social_agent.run_night_generation()


# --- FASTAPI LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(master_morning_routine, CronTrigger(hour=7, minute=0, timezone=LOCAL_TZ))
    scheduler.start(); yield; scheduler.shutdown()

app = FastAPI(lifespan=lifespan); app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_headers=["*"], allow_methods=["*"])
class ChatRequest(BaseModel): message: str; context_url: Optional[str] = ""; session_id: Optional[str] = ""
class InitRequest(BaseModel): url: str; session_id: str; ui_lang: str = "hu"

@app.get("/")
def home(): return {"status": "V225 Online", "project": "Booksy"}

@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url, req.session_id)

@app.post("/init-chat")
def init_chat(req: InitRequest): return {"ui_lang": req.ui_lang, "bubble_text": "Szia!", "placeholder": "Keresel valamit?"}

@app.post("/test-social-night")
def test_night(bt: BackgroundTasks): 
    bt.add_task(social_agent.run_night_generation)
    return {"status": "V225 Agentic Video Test Started"}

@app.post("/test-cascade")
def test_cascade(bt: BackgroundTasks):
    bt.add_task(master_morning_routine)
    return {"status": "V225 Full Cascade Test Started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)