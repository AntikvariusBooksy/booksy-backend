# BOOKSY BRAIN - V233 (THE MASTERPIECE EDITION - CASCADE ACTIVE)
# VERZIÓ: V233 - ZERO-TOLERANCE GRAMMAR + 35MM REALISTIC VISUALS + FORCED MARKETING BRIDGE + TROJAN API + DB SYNC ACTIVE

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os, time, requests, hashlib, re, json, random, unicodedata, html, urllib.parse, gc, chromadb, pytz, smtplib, traceback
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
from openai import OpenAI
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
            log_event(f"Indítás: FB Komment Bot (Trojan). Force ID: {force_post_id}")

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
                            if media_id:
                                atts = p.get('attachments', {}).get('data', [])
                                for att in atts:
                                    target = att.get('target', {})
                                    target_id = str(target.get('id', ''))
                                    target_url = str(target.get('url', ''))
                                    att_url = str(att.get('url', ''))
                                    
                                    if target_id == str(media_id) or str(media_id) in target_url or str(media_id) in att_url:
                                        found = True; break
                                    
                                    for sub in att.get('subattachments', {}).get('data', []):
                                        sub_target = sub.get('target', {})
                                        if str(sub_target.get('id', '')) == str(media_id) or str(media_id) in str(sub_target.get('url', '')) or str(media_id) in str(sub.get('url', '')):
                                            found = True; break
                                if found: break
                            
                            if not found and search_title:
                                msg = normalize_fingerprint(p.get("message", ""))
                                if search_title in msg: found = True
                            
                            if found:
                                target_post_id = p["id"]
                                ep_name = ep.split('?')[0].split('/')[-1]
                                log_event(f"✅ Találat a '{ep_name}' végponton! Poszt azonosítója: {target_post_id}")
                                break
                    except Exception as loop_e: log_event(f"Hiba a végpont ellenőrzésénél: {loop_e}")
            
            if not target_post_id: 
                err_msg = "❌ Célpont poszt (vagy Reels) nem található a publikus hírfolyamon."
                log_event(err_msg)
                return {"reply": err_msg, "products": []}

            # --- TRÓJAI FALÓ PROTOKOLL ---
            clean_hook_text = "📚 A mai válogatásunk kincseit és a könyvek elérhetőségét a válaszban találjátok! Aki kapja, marja! 😉👇"
            log_event(f"Trójai Faló 1. Lépés: Tiszta horog küldése a(z) {target_post_id} azonosítóra...")
            c_res = requests.post(f"https://graph.facebook.com/v19.0/{target_post_id}/comments", data={'access_token': fb_token, 'message': clean_hook_text})
            
            c_data = c_res.json()
            if "id" in c_data:
                parent_comment_id = c_data["id"]
                log_event(f"✅ Horog beakadt (ID: {parent_comment_id}). Trójai Faló 2. Lépés: Linkes rakomány küldése válaszként...")
                
                payload_text = ""
                for book in memory.get("links", []):
                    res = self.db.collection.get(ids=[book.get('id', 'None')])
                    status = " ❌ (Már el is kelt!)" if (res['metadatas'] and res['metadatas'][0].get('stock') == 'outofstock') else ""
                    author = f"{book['author']} - " if (book.get('author') and book['author'] != 'Ismeretlen') else ""
                    m_desc = book.get('marketing_desc', '')
                    
                    payload_text += f"📖 {author}{book['title']}{status}\n{m_desc}\n🔗 {book['url']}\n\n"
                
                r_res = requests.post(f"https://graph.facebook.com/v19.0/{parent_comment_id}/comments", data={'access_token': fb_token, 'message': payload_text.strip()})
                
                if "id" in r_res.json():
                    log_event("✅ Rakomány (Válasz) sikeresen rögzítve a horgon.")
                    return {"reply": "✅ Komment (Trójai Faló) sikeresen kiment!", "products": []}
                else:
                    log_event(f"FB Hiba a válasznál: {r_res.text}")
                    return {"reply": f"❌ FB hiba a válasz-kommentelésnél: {r_res.text}", "products": []}
            else:
                log_event(f"FB Hiba a főkommentnél: {c_res.text}")
                return {"reply": f"❌ FB hiba a főkommentelésnél: {c_res.text}", "products": []}

        except Exception as e:
            log_event(f"Rendszerhiba: {e}")
            return {"reply": f"❌ Hiba: {e}", "products": []}

class BooksySocialAgent:
    def __init__(self, db: DBHandler):
        self.db = db
        self.claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def send_error_email(self, error_details):
        try:
            sender, password = os.getenv("SMTP_SENDER"), os.getenv("SMTP_PASSWORD")
            admin_emails = [e.strip() for e in os.getenv("ADMIN_EMAIL", "").split(",") if e.strip()]
            if not sender or not admin_emails: return
            
            server = smtplib.SMTP(os.getenv("SMTP_SERVER", "mail.antikvarius.ro"), 26, timeout=20)
            server.starttls(); server.login(sender, password)
            for admin in admin_emails:
                msg = MIMEMultipart()
                msg['From'] = f"Booksy AI <{sender}>"; msg['To'] = admin
                msg['Subject'] = f"⚠️ KRITIKUS HIBA: Booksy Social Agent ({datetime.now(LOCAL_TZ).strftime('%Y-%m-%d')})"
                body = f"Üdv!\n\nA napi Facebook vázlat generálása során váratlan hiba történt. A folyamat megszakadt.\n\nRészletek a fejlesztőnek:\n\n{error_details}"
                msg.attach(MIMEText(body, 'plain', 'utf-8'))
                server.send_message(msg)
            server.quit()
            log_event("⚠️ Hiba e-mail sikeresen elküldve az adminoknak.")
        except Exception as e: log_event(f"📧 Hiba az error e-mail küldésénél: {e}")

    def send_morning_email(self, post_text, memory_links):
        try:
            sender, password = os.getenv("SMTP_SENDER"), os.getenv("SMTP_PASSWORD")
            admin_emails = [e.strip() for e in os.getenv("ADMIN_EMAIL", "").split(",") if e.strip()]
            if not sender: return
            
            links_body = ""
            for b in memory_links:
                author_display = f"{b['author']} - " if b.get('author') and b['author'] != 'Ismeretlen' else ""
                links_body += f"📖 {author_display}{b['title']}\n{b.get('marketing_desc', '')}\n🔗 {b['url']}\n\n"

            server = smtplib.SMTP(os.getenv("SMTP_SERVER", "mail.antikvarius.ro"), 26, timeout=20)
            server.starttls(); server.login(sender, password)
            for admin in admin_emails:
                msg = MIMEMultipart()
                msg['From'] = f"Booksy AI <{sender}>"; msg['To'] = admin
                msg['Subject'] = f"✅ Booksy Social Vázlat ({datetime.now(LOCAL_TZ).strftime('%Y-%m-%d')})"
                body = f"Üdv!\n\nA FB vázlat elkészült a Drafts mappába.\n\nMiután Business Suite-ban rákattintottál a Publikálás gombra, a chatben használd a /booklink admin123 parancsot a kommenthez!\n\nSZÖVEG:\n{post_text}\n\nKOMMENTBE MEGY (MÁSOLHATÓ):\n{links_body.strip()}"
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

            font_path = "Montserrat-Bold.ttf"
            use_bbox = False
            best_font = ImageFont.load_default()

            if os.path.exists(font_path): use_bbox = True
            else: log_event("❌ FIGYELEM: A 'Montserrat-Bold.ttf' nem található! Alapértelmezett betű aktív.")

            overlay = PIL.Image.new('RGBA', (width, height), (0,0,0,0))
            draw = ImageDraw.Draw(overlay)
            bar_height = int(height * 0.15); bar_y = height - bar_height
            draw.rectangle([(0, bar_y), (width, height)], fill=(0, 0, 0, 180)) 
            
            author_text = f"{author} - " if author and author != "Ismeretlen" else ""
            full_text = f"{author_text}{title}"

            if use_bbox:
                target_width = int(width * 0.80); target_height = int(bar_height * 0.60); font_size = 10
                try:
                    while True:
                        test_font = ImageFont.truetype(font_path, font_size + 1)
                        bbox = test_font.getbbox(full_text)
                        if (bbox[2]-bbox[0]) > target_width or (bbox[3]-bbox[1]) > target_height: break
                        font_size += 1; best_font = test_font
                except: pass
                
            bbox = best_font.getbbox(full_text); text_w = bbox[2] - bbox[0]; text_h = bbox[3] - bbox[1]
            draw.text(((width - text_w) / 2, bar_y + (bar_height - text_h) / 2 - bbox[1]), full_text, font=best_font, fill=(255, 255, 255, 255))
            overlay.save(overlay_path, "PNG")
            combined = PIL.Image.alpha_composite(img, overlay); combined.convert("RGB").save(fallback_path, "JPEG")
            log_event("Rétegek mentve.")
            return True
        except Exception as e: log_event(f"Hiba a vizuális rétegeknél: {e}"); return False

    def _create_video(self, raw_img_path, overlay_path, out_path):
        if not MOVIEPY_AVAILABLE: return False
        try:
            log_event("Videó renderelés indítása...")
            clip = ImageClip(raw_img_path).set_duration(5)
            zoomed = clip.resize(lambda t: 1 + 0.03 * t).set_position('center')
            fixed_bg = CompositeVideoClip([zoomed], size=(1920, 1920)).set_duration(5)
            bg_loop = concatenate_videoclips([fixed_bg, fixed_bg.fx(vfx.time_mirror)])
            if os.path.exists(overlay_path):
                overlay_img = PIL.Image.open(overlay_path).convert("RGBA")
                def stamp_overlay(frame_array):
                    pil_frame = PIL.Image.fromarray(np.clip(frame_array, 0, 255).astype(np.uint8)).convert("RGBA")
                    pil_frame.alpha_composite(overlay_img)
                    return np.array(pil_frame.convert("RGB"))
                final_video = bg_loop.fl_image(stamp_overlay)
            else: final_video = bg_loop
            final_video.write_videofile(out_path, fps=24, codec="libx264", audio=False, ffmpeg_params=["-pix_fmt", "yuv420p"], logger=None)
            return True
        except Exception as e: log_event(f"Videó hiba: {e}"); return False

    def run_night_generation(self):
        log_event("Agentic Generálás indítása (V233 Masterpiece)...")
        raw_img_path = "social_raw.jpg"; overlay_path = "social_overlay.png"; fallback_img_path = "social_fallback.jpg"; vid_path = "social_video.mp4"
        
        try:
            # --- STEP 1: STRICT WIKIPEDIA + GEMINI AUTHOR SELECTION ---
            today_date = datetime.now(LOCAL_TZ).strftime('%B %d')
            log_event(f"Step 1: Élő API lekérdezés a mai napról ({today_date}) és Gemini Kereszttűz (SZIGORÚ ÍRÓ SZŰRŐ)...")
            
            r_wiki = requests.get(f"https://en.wikipedia.org/api/rest_v1/feed/onthisday/births/{datetime.now(LOCAL_TZ).strftime('%m/%d')}", headers={'User-Agent': 'BooksyBot/1.0'})
            wiki_text = "Nem található adat."
            if r_wiki.status_code == 200:
                births = [p.get('text', '') for p in r_wiki.json().get('births', []) if any(kw in p.get('text', '').lower() for kw in ['writer', 'author', 'poet', 'novelist'])]
                wiki_text = "\n".join(births[:30])
            
            author_prompt = (
                f"Ma {today_date} van. Itt egy nyers lista a Wikipédiáról a mai napon született személyekről: {wiki_text}\n"
                f"Végezz élő internetes kutatást! Válaszd ki pontosan a legrelevánsabb 6 embert, de KIZÁRÓLAG KÖNYVÍRÓKAT "
                f"(regényíró, költő, esszéista, sci-fi író, tudományos-ismeretterjesztő). SZIGORÚAN TILOS listázni filmrendezőket, "
                f"képregényrajzolókat, zenészeket, animátorokat, forgatókönyvírókat. Csak azokat, akik klasszikus értelemben vett könyveket írtak!\n"
                f"Prioritás: Ha a listában van magyar vagy román író, kötelezően vedd be! A többit klasszikusokkal töltsd fel.\n"
                f"Készíts róluk 'mini lexikon' megemlékezést (1-2 mondat/író). SZIGORÚ KIMENET: Csak és kizárólag XML formátum:\n"
                f"<authors><author><name>Író Neve</name><nationality>Nemzetiség</nationality><bio>Rövid életrajz, stílus és legismertebb műve.</bio></author></authors>"
            )
            gem_authors_res = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=[author_prompt])
            authors_list = safe_authors_parse(gem_authors_res.text)[:6]

            # --- STEP 2: INVENTORY CHECK (CHROMADB RADAR) ---
            log_event("Step 2: Raktárkészlet ellenőrzése a kiválasztott írók alapján...")
            selected_books, seen_ids = [], set()
            for author in authors_list:
                vec = gemini_client.models.embed_content(model="gemini-embedding-001", contents=author['name'], config=types.EmbedContentConfig(output_dimensionality=768)).embeddings[0].values
                res = self.db.collection.query(query_embeddings=[vec], n_results=3, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
                if res['ids'] and res['ids'][0]:
                    for p_target in res['metadatas'][0]:
                        if p_target['id'] not in seen_ids:
                            selected_books.append(p_target); seen_ids.add(p_target['id']); break

            if len(selected_books) < 3:
                log_event("Figyelem: Kevesebb mint 3 könyv van a mai íróktól. Kiegészítés klasszikusokkal a kosár feltöltéséhez...")
                vec_fb = gemini_client.models.embed_content(model="gemini-embedding-001", contents="népszerű klasszikus és modern irodalom", config=types.EmbedContentConfig(output_dimensionality=768)).embeddings[0].values
                res_fb = self.db.collection.query(query_embeddings=[vec_fb], n_results=10, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
                if res_fb['ids'] and res_fb['ids'][0]:
                    for p_target in res_fb['metadatas'][0]:
                        if p_target['id'] not in seen_ids:
                            selected_books.append(p_target); seen_ids.add(p_target['id'])
                        if len(selected_books) >= 5: break

            main_book = selected_books[0]; log_event(f"Vizuális Fókusz Könyv: {main_book['title']} by {main_book.get('author', '')}")

            # --- STEP 3: GENERATE MARKETING DESCRIPTIONS (CLAUDE) ---
            log_event("Step 3: Könyvajánlók megírása (Egymondatos zamatos marketing + Grammatikai Szigor)...")
            for b in selected_books:
                desc_prompt = f"Könyv: {b['title']} - {b['author']}. Rövid infó: {b.get('text_preview', '')}. Írj EGYETLEN, magával ragadó, zamatos magyar nyelvű marketing mondatot, ami meghozza a kedvet az olvasáshoz! Tökéletes nyelvhelyességgel, megfelelő ékezetekkel (ő, ű) és mondatzáró írásjellel. Ne csak a tartalmat írd le, add el az élményt. Csak a mondatot add vissza!"
                b['marketing_desc'] = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=100, messages=[{"role": "user", "content": desc_prompt}]).content[0].text.strip()
            
            # --- STEP 4: VISUAL FOCUS (DEEP SCAN & 35MM REALISTIC DALL-E) ---
            log_event("Step 4: A fókusz-könyv mélyelemzése (Gemini) és 35mm-es DALL-E 3 képgenerálás...")
            analysis_prompt = f"Végezz alapos netes kutatást és elemezd ki ezt a KÖNYVET: '{main_book['title']}' írta {main_book.get('author','valaki')}. Alapadat: {main_book.get('text_preview','')}. Tárd fel a pontos, valós cselekményt, kulcsjeleneteket és vizuális motívumokat hallucináció nélkül! Adj egy tömör, pontos vizuális összefoglalót angolul."
            gem_res = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=[analysis_prompt])
            
            claude_prompt = (
                f"Te egy filmes látványtervező vagy. Vesd össze a Gemini elemzését a saját irodalmi tudásoddal, és írj egy DALL-E 3 képgenerálási promptot angolul. "
                f"Elemzés: {gem_res.text} SZIGORÚ SZABÁLYOK: A promptnak 100%-ban meg kell felelnie az OpenAI biztonsági irányelveinek (G-rated). "
                f"Nincs erőszak vagy felkavaró utalás. Emberek és arcok megengedettek a képen, DE ha embert ábrázolsz, annak SZIGORÚAN maximálisan "
                f"élethűnek, fotorealisztikusnak és anatómiailag hibátlannak kell lennie! Hallucináció, elfolyó részletek vagy torzulás szigorúan tilos! "
                f"Stílus: 35mm-es filmkocka, anamorfikus lencse, természetes szemcsézettség (film grain), mély árnyékok és organikus textúrák. "
                f"Mintha egy 90-es évekbeli klasszikus kosztümös filmből vágták volna ki. Zéró 'műanyag' vagy tipikus AI hatás. Nincs szöveg a képen. Csak a promptot küldd!"
            )
            c_res = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=300, messages=[{"role": "user", "content": claude_prompt}])
            final_img_prompt = c_res.content[0].text
            
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            try:
                img_res = openai_client.images.generate(
                    model="dall-e-3",
                    prompt=final_img_prompt,
                    size="1024x1024",
                    quality="hd",
                    n=1
                )
                img_url = img_res.data[0].url
                log_event("✅ DALL-E 3 kép sikeresen legenerálva az elsődleges prompttal.")
            except Exception as dalle_err:
                log_event(f"⚠️ DALL-E 3 Hiba: {dalle_err}. Biztonsági Fallback aktiválása...")
                safe_prompt = "A beautiful, cinematic, 35mm film frame of an antique book on a dark wooden table lit by a single candle. Deep shadows, film grain, organic textures, completely safe, no people, abstract and atmospheric."
                img_res = openai_client.images.generate(
                    model="dall-e-3",
                    prompt=safe_prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1
                )
                img_url = img_res.data[0].url
                log_event("✅ DALL-E 3 kép sikeresen legenerálva a biztonsági fallback prompttal.")

            r_img = requests.get(img_url, timeout=90)
            if r_img.status_code == 200:
                with open(raw_img_path, 'wb') as f: f.write(r_img.content)
            else:
                raise Exception(f"DALL-E letöltési hiba HTTP {r_img.status_code}")
            
            img_obj = PIL.Image.open(raw_img_path)
            img_resized = PIL.ImageOps.fit(img_obj, (1920, 1920), PIL.Image.Resampling.LANCZOS); img_resized.save(raw_img_path)
            self._prepare_visual_layers(raw_img_path, overlay_path, fallback_img_path, main_book['title'], main_book.get('author', ''))
            has_video = self._create_video(raw_img_path, overlay_path, vid_path)

            # --- STEP 5: POST TEXT DRAFTING (FORCED BRIDGE) ---
            log_event("Step 5: Napi Lexikon poszt szöveg generálása (CopySEO Vázlat - Kötelező Híddal)...")
            authors_text = "\n".join([f"📖 {a['name']} ({a.get('nationality', 'Világirodalom')}): {a['bio']}" for a in authors_list])

            draft_prompt = (
                f"Írj egy posztot az Antikvarius.ro FB oldalára. Koncepció: Napi 'irodalmi naptár' és mini lexikon.\n"
                f"SZIGORÚ SZERKEZETI SZABÁLYOK:\n"
                f"1. A poszt LÉGELSŐ sora kötelezően egy Facebook NLP érzelem címke legyen pontosan így: [Érzés: inspirált 🌟] vagy [Érzés: nosztalgikus 📚].\n"
                f"2. MINI LEXIKON: Készíts megemlékezést az alábbi 6, ma született íróról:\n{authors_text}\n\n"
                f"3. KÖTELEZŐ IRODALMI HÍD (MARKETING ÁTVEZETÉS): A lexikon után kötelezően írj egy kifinomult, de erőteljes átvezetőt (minimum 3-4 mondat). "
                f"A narratíva: Magyarázd el az olvasónak, hogy bár a szülinapos írók ritka kincsek és a köteteik ma épp más szerencsés gyűjtők polcait díszítik nálunk, "
                f"az ő szellemiségüket ma is megtalálják a polcainkon. Külön emeld ki a válogatás első darabját: '{main_book['title']}' by {main_book.get('author', '')}. "
                f"Ez az átvezetés KÖTELEZŐ, nem maradhat ki!\n\n"
                f"TOVÁBBI SZABÁLYOK:\n"
                f"- Tónus: Zamatos, választékos, gyönyörű magyar nyelvezet. Úgy írj, mint egy szenvedélyes, művelt antikvárius.\n"
                f"- ZÉRÓ LINK a posztban!\n"
            )
            draft_text = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=1000, system="Professional CopySEO tone. No URLs allowed.", messages=[{"role": "user", "content": draft_prompt}]).content[0].text
            
            # --- STEP 6: 2-STEP LECTORING (ZERO TOLERANCE) ---
            log_event("Step 6: Kétlépcsős Lektorálás (Zero-Tolerance Nyelvtani Ellenőrzés)...")
            lector_prompt = (
                f"Az alábbi Facebook poszt vázlatot lektoráld! Végezz kőkemény, karakterenkénti nyelvtani és stilisztikai ellenőrzést. "
                f"Különös figyelmet fordíts a mondatzáró írásjelekre (minden felsorolás és mondat végén legyen pont vagy megfelelő írásjel!), "
                f"a kettős ékezetekre (ő, ű helyes használata) és a gépelési hibákra. "
                f"Legyen tökéletesen magyaros, zamatos, választékos, mentes az anglicizmusoktól (tükörfordításoktól) és a fogalmazási hibáktól. "
                f"Őrizd meg az NLP '[Érzés: ...]' címkét a legelső sorban, és ellenőrizd a 'KÖTELEZŐ IRODALMI HÍD' jelenlétét. "
                f"NE írj bevezetőt, csak a tökéletes, végleges poszt szövegét add vissza!\n\n"
                f"VÁZLAT:\n{draft_text}"
            )
            post_text = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=1000, messages=[{"role": "user", "content": lector_prompt}]).content[0].text

            # --- STEP 7: INTEGRITY CHECKS (HARDCODED CTA & NLP FALLBACK) ---
            log_event("Step 7: Belső Python Integritás-vizsgálat (NLP & CTA)...")
            if not re.search(r'\[Érzés:.*?\]', post_text):
                post_text = "[Érzés: inspirált 🌟]\n\n" + post_text

            if "keressétek az első kommentben" not in post_text.lower():
                post_text += "\n\nA mai válogatásunkat és a könyvek elérhetőségét keressétek az első kommentben! 👇"

            # --- STEP 8: PUBLISH & MEMORY ---
            memory_data = {"fingerprint": post_text[:100], "links": [{"id": b['id'], "title": b['title'], "author": b['author'], "url": b['url'], "marketing_desc": b.get('marketing_desc', '')} for b in selected_books]}
            fb_id, fb_token = os.getenv("FB_PAGE_ID"), os.getenv("FB_PAGE_TOKEN")
            
            if has_video:
                r_v = requests.post(f"https://graph.facebook.com/v19.0/{fb_id}/videos", data={'access_token': fb_token, 'description': post_text, 'published': 'false', 'unpublished_content_type': 'DRAFT'}, files={'source': open(vid_path, 'rb')})
                if r_v.status_code == 200:
                    vid_id = str(r_v.json().get('id')); memory_data['media_id'] = vid_id
                    log_event(f"✅ Videó vázlat kész! (ID: {vid_id})")
            else:
                upload_img = fallback_img_path if os.path.exists(fallback_img_path) else raw_img_path
                r_p = requests.post(f"https://graph.facebook.com/v19.0/{fb_id}/photos", data={'access_token': fb_token, 'message': post_text, 'published': 'false'}, files={'source': open(upload_img, 'rb')})
                if r_p.status_code == 200:
                    photo_id = str(r_p.json().get('id')); memory_data['media_id'] = photo_id
                    log_event(f"✅ Képes vázlat kész! (ID: {photo_id})")
            
            with open(SOCIAL_MEMORY_FILE, "w", encoding="utf-8") as f: json.dump(memory_data, f, ensure_ascii=False)
            self.send_morning_email(post_text, memory_data['links']); log_event("Kész.")
            
        except Exception as e:
            err_trace = traceback.format_exc()
            log_event(f"❌ KRITIKUS RENDSZERHIBA: {e}")
            self.send_error_email(err_trace)
        finally:
            for p in [raw_img_path, overlay_path, fallback_img_path, vid_path]:
                if os.path.exists(p): os.remove(p)

# --- MASTER CASCADE ROUTINE ---
updater = AutoUpdater(db_handler); bot = BooksyBrain(db_handler); social_agent = BooksySocialAgent(db_handler); scheduler = BackgroundScheduler()

def master_morning_routine():
    log_event("🌅 Master Láncreakció Indítása: DB Sync -> Social Post")
    try:
        sync_success = updater.run_daily_update()
        if not sync_success:
            log_event("⚠️ Figyelem: A szinkronizáció nem sikerült. Biztonsági protokoll: Korábbi adatok használata.")
    except Exception as e:
        log_event(f"⚠️ Váratlan hiba a szinkronnál: {e}. Biztonsági protokoll aktiválva.")
    
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
def home(): return {"status": "V233 Online", "project": "Booksy"}

@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url, req.session_id)

@app.post("/init-chat")
def init_chat(req: InitRequest): return {"ui_lang": req.ui_lang, "bubble_text": "Szia!", "placeholder": "Keresel valamit?"}

@app.post("/test-social-night")
def test_night(bt: BackgroundTasks): 
    bt.add_task(social_agent.run_night_generation)
    return {"status": "V233 Agentic Masterpiece Test Started"}

@app.post("/test-cascade")
def test_cascade(bt: BackgroundTasks):
    bt.add_task(master_morning_routine)
    return {"status": "V233 Full Cascade Test Started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)