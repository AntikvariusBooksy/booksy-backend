# BOOKSY BRAIN - V149 (GEMINI RESEARCHER + CLAUDE DIRECTOR HYBRID)
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

import google.generativeai as genai
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

# Google Gemini Konfiguráció
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

XML_FEED_URL = os.getenv("XML_FEED_URL", "https://www.antikvarius.ro/wp-content/uploads/woo-feed/google/xml/booksyfullfeed.xml")
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
        self.collection = self.client.get_or_create_collection(name="booksy_collection_gemini")

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
        print("🔄 [FRISSÍTÉS] XML feed letöltése és Gemini vektorizálás indul...")
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
            
            ids_batch, emb_texts_batch, metadatas_batch = [], [], []
            for bid, book_data in unique_books_buffer.items():
                emb_text = f"SKU: {bid}. Nyelv: {book_data['lang']}. Cím: {book_data['title']}. Szerző: {book_data['author']}. Leírás: {book_data['description'][:800]}"
                clean_meta = book_data.copy()
                del clean_meta['description'] 
                clean_meta['text_preview'] = book_data['description'][:150]
                
                ids_batch.append(bid)
                emb_texts_batch.append(emb_text[:8000])
                metadatas_batch.append(clean_meta)
                
                if len(ids_batch) >= 50:
                    try:
                        result = genai.embed_content(model="models/text-embedding-004", content=emb_texts_batch)
                        self.db.collection.upsert(ids=ids_batch, embeddings=result['embedding'], metadatas=metadatas_batch)
                    except Exception as e: print(f"Hiba a Gemini embeddingnél: {e}")
                    ids_batch, emb_texts_batch, metadatas_batch = [], [], []
                    time.sleep(1) 
            
            if ids_batch: 
                try:
                    result = genai.embed_content(model="models/text-embedding-004", content=emb_texts_batch)
                    self.db.collection.upsert(ids=ids_batch, embeddings=result['embedding'], metadatas=metadatas_batch)
                except: pass
            
            if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)
            print("✅ [FRISSÍTÉS] Kész. Gemini adatbázis naprakész.")
        except Exception as e: pass

class ChatRequest(BaseModel): message: str; context_url: Optional[str] = ""; session_id: Optional[str] = ""
class InitRequest(BaseModel): url: str; session_id: str; ui_lang: str = "ro"

class BooksyBrain:
    def __init__(self, db: DBHandler):
        self.db = db
        self.user_session_cache = {}
        self.client_claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def process(self, msg, context_url, session_id):
        try:
            # 1. Agy (Gemini elemzi a szándékot)
            model_json = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})
            prompt_intent = f"Intent analysis for bookstore. Input: '{msg}'. JSON output MUST be EXACTLY: {{\"intent\": \"search\"|\"policy\", \"query\": \"query\"}}"
            analysis = model_json.generate_content(prompt_intent).text
            intent_data = json.loads(analysis)
            
            # 2. Kutató (Gemini keres az adatbázisban)
            query_text = intent_data.get('query', msg)
            vec = genai.embed_content(model="models/text-embedding-004", content=query_text)['embedding']
            
            final_reply, final_products = "", []
            
            # 3. Hang (Claude 4.6 fogalmazza meg a választ)
            if intent_data.get('intent') == "policy":
                res = self.db.collection.query(query_embeddings=[vec], n_results=2, where={"type": "policy"})
                ctx = "".join([m.get('text', '') for m in res['metadatas'][0]]) if res['ids'] else ""
                
                reply_res = self.client_claude.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=800,
                    temperature=0.5,
                    system="You are the elegant, helpful Hungarian Booksy Assistant. Respond in Hungarian based on the provided context.",
                    messages=[{"role": "user", "content": f"Context: {ctx}\nUser asks: {msg}"}]
                )
                final_reply = reply_res.content[0].text
            else:
                res = self.db.collection.query(query_embeddings=[vec], n_results=5, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
                if res['ids']:
                    ctx_text = ""
                    for meta in res['metadatas'][0]:
                        p_price = clean_price_raw(meta.get('price'))
                        final_products.append({"title": meta.get('title'), "price": p_price, "url": meta.get('url'), "image": meta.get('image_url')})
                        ctx_text += f"- {meta.get('title')} by {meta.get('author', 'Unknown')} ({p_price}). Kategória: {meta.get('category', '')}\n"
                    
                    reply_res = self.client_claude.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=800,
                        temperature=0.7,
                        system="You are the elegant, helpful Hungarian Booksy CopySEO Assistant. Write a warm, highly convincing recommendation in Hungarian based ONLY on the provided list. Do not hallucinate prices or books.",
                        messages=[{"role": "user", "content": f"Available books: {ctx_text}\nUser asks: {msg}"}]
                    )
                    final_reply = reply_res.content[0].text
                else: final_reply = "Sajnos nem találtam tökéletesen megfelelő könyvet a raktáron."
            
            self.user_session_cache[session_id] = msg
            return {"reply": final_reply, "products": final_products}
        except Exception as e: 
            print(f"Chat hiba: {e}")
            return {"reply": "Hiba történt a keresés során.", "products": []}

    def negotiate_handshake(self, url, session_id, ui_lang):
        try:
            model_json = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})
            res = model_json.generate_content(f"Generate a JSON greeting in {ui_lang}. Format: {{\"ui_lang\": \"{ui_lang}\", \"bubble_text\": \"...\", \"placeholder\": \"...\"}}").text
            return json.loads(res)
        except: return {"ui_lang": ui_lang, "bubble_text": "Miben segíthetek?", "placeholder": "Keresel valamit?"}

class BooksySocialAgent:
    def __init__(self, db: DBHandler):
        self.db = db
        self.client_claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) 

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
                model="claude-sonnet-4-6", 
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
        print("🕒 [SOCIAL] Agentikus Generálás indul (V149 - GEMINI RESEARCHER + CLAUDE DIRECTOR)...")
        calendar = self._get_agentic_calendar()
        ünnepeltek = calendar.get("authors", [])
        
        poszt_adatai = []
        if ünnepeltek:
            for író in ünnepeltek:
                vec = genai.embed_content(model="models/text-embedding-004", content=író['name'])['embedding']
                res = self.db.collection.query(query_embeddings=[vec], n_results=1, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
                if res['ids'] and res['ids'][0]:
                    meta = res['metadatas'][0][0]
                    if normalize_text(író['name'].split()[-1]) in normalize_text(str(meta.get('author', ''))):
                        poszt_adatai.append({"author": író['name'], "bio": író.get('bio', ''), "title": meta.get('title'), "url": meta.get('url'), "price": clean_price_raw(meta.get('price')), "preview": meta.get('text_preview', ''), "category": meta.get('category', '')})

        has_author_books = len(poszt_adatai) > 0
        fallback_adatai = []
        if not has_author_books:
            themes = ["ritka antikvár könyv", "izgalmas krimik", "klasszikus magyar szépirodalom", "történelmi szakkönyvek"]
            vec = genai.embed_content(model="models/text-embedding-004", content=random.choice(themes))['embedding']
            fallback_res = self.db.collection.query(query_embeddings=[vec], n_results=10, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
            if fallback_res['ids'] and fallback_res['ids'][0]:
                for i in random.sample(range(len(fallback_res['ids'][0])), min(3, len(fallback_res['ids'][0]))):
                    f_meta = fallback_res['metadatas'][0][i]
                    fallback_adatai.append({"author": f_meta.get('author', 'Ismeretlen'), "title": f_meta.get('title'), "url": f_meta.get('url'), "price": clean_price_raw(f_meta.get('price')), "preview": f_meta.get('text_preview', ''), "category": f_meta.get('category', '')})

        konyv_cim = poszt_adatai[0]['title'] if has_author_books else (fallback_adatai[0]['title'] if fallback_adatai else "Antikvár kincsek")
        konyv_szerzo = poszt_adatai[0].get('author', 'Ismeretlen') if has_author_books else (fallback_adatai[0].get('author', 'Ismeretlen') if fallback_adatai else "Ismeretlen")
        konyv_tartalom = poszt_adatai[0].get('preview', '') if has_author_books else (fallback_adatai[0].get('preview', '') if fallback_adatai else "")
        konyv_kategoria = poszt_adatai[0].get('category', '') if has_author_books else (fallback_adatai[0].get('category', '') if fallback_adatai else "")

        # --- 1. A KUTATÓ (GEMINI 1.5 FLASH) ---
        research_prompt = f"Provide a concise but highly visual and thematic summary of the book '{konyv_cim}' by '{konyv_szerzo}'. Include the core message and one iconic scene or atmospheric setting. If you don't know the exact book, describe the general atmosphere based on this context: '{konyv_tartalom}' and category: '{konyv_kategoria}'. Respond in English."
        
        try:
            print("🔍 [GEMINI] Könyv tartalmának és hangulatának mély kutatása...")
            research_model = genai.GenerativeModel('gemini-1.5-flash')
            deep_book_context = research_model.generate_content(research_prompt).text
        except Exception as e:
            print(f"❌ Gemini hiba a kutatásnál: {e}")
            deep_book_context = f"{konyv_tartalom} Category: {konyv_kategoria}"

        # --- 2. A RENDEZŐ (CLAUDE 4.6 - Kép prompt) ---
        img_director_prompt = f"Based on this book context: '{deep_book_context}', write a single, detailed English prompt for a text-to-image AI (like Midjourney or Flux). The image must be a photorealistic, cinematic atmospheric scene that captures the soul of the book. CRITICAL RULES: NO TEXT, NO WORDS, NO LETTERS, NO FACES. Just the pure visual atmosphere or iconic setting. Output ONLY the prompt text, nothing else."
        
        try:
            print("🎬 [CLAUDE] Képgenerálási prompt (Rendező) írása...")
            img_res = self.client_claude.messages.create(
                model="claude-sonnet-4-6", 
                max_tokens=300,
                temperature=0.7,
                messages=[{"role": "user", "content": img_director_prompt}]
            )
            final_img_prompt = img_res.content[0].text
        except Exception as e:
            print(f"❌ Claude API hiba (Rendező): {e}")
            final_img_prompt = f"A photorealistic, cinematic atmospheric scene inspired by the book '{konyv_cim}'. High-end photography, 8k, lifelike textures. NO TEXT. NO WORDS. NO FACES."

        # --- 3. A GRAFIKUS (POLLINATIONS) ---
        video_path, image_path = "social_video.mp4", "social_img.jpg"
        media_url, is_video = "", False
        try:
            print("🎨 [IMAGE] Ingyenes atmoszferikus kép generálása (Flux modell)...")
            encoded_prompt = urllib.parse.quote(final_img_prompt)
            free_img_url = f"[https://image.pollinations.ai/prompt/](https://image.pollinations.ai/prompt/){encoded_prompt}?width=1024&height=1024&nologo=true"
            img_data = requests.get(free_img_url, timeout=30).content
            with open(image_path, 'wb') as f: f.write(img_data)
            is_video = self._create_infinite_loop_video(image_path, video_path)
        except Exception as e: print(f"❌ Kép hiba: {e}")

        # --- 4. A SZÖVEGÍRÓ (CLAUDE 4.6) ---
        marketing_prompt = f"Act as Booksy CopySEO, the ultimate marketing expert. Write an engaging Facebook post in Hungarian. State clearly that TODAY is the birthday of: {json.dumps(ünnepeltek)}. Holiday: {calendar.get('holiday')}. Books to recommend: {json.dumps(poszt_adatai if has_author_books else fallback_adatai)}. Use the exact provided URLs. Keep the tone elegant and persuasive. Do not hallucinate."
        
        try:
            print("🖋️ [CLAUDE] Marketing szövegírás folyamatban...")
            post_res = self.client_claude.messages.create(
                model="claude-sonnet-4-6", 
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
                if is_video:
                    res = requests.post(
                        f"[https://graph.facebook.com/v19.0/](https://graph.facebook.com/v19.0/){fb_page_id}/videos", 
                        data={'access_token': fb_token, 'description': post_text, 'published': 'false', 'unpublished_content_type': 'DRAFT'}, 
                        files={'source': open(video_path, 'rb')}
                    )
                else:
                    res = requests.post(
                        f"[https://graph.facebook.com/v19.0/](https://graph.facebook.com/v19.0/){fb_page_id}/photos", 
                        data={"message": post_text, "published": False, "unpublished_content_type": "DRAFT", "access_token": fb_token},
                        files={'source': open(image_path, 'rb')}
                    )
                
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
def home(): return {"status": "Booksy V149 (GEMINI RESEARCHER + CLAUDE DIRECTOR)"}
@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url, req.session_id)
@app.post("/init-chat")
def init_chat(req: InitRequest): return bot.negotiate_handshake(req.url, req.session_id, req.ui_lang)
@app.post("/test-social-night")
def test_night(bt: BackgroundTasks): bt.add_task(social_agent.run_night_generation); return {"status": "Triggered"}
@app.post("/test-social-morning")
def test_morning(bt: BackgroundTasks): bt.add_task(social_agent.send_morning_email); return {"status": "Triggered"}
@app.post("/force-update")
def force_update(bt: BackgroundTasks): bt.add_task(updater.run_daily_update); return {"status": "Frissítés elindítva a háttérben"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)