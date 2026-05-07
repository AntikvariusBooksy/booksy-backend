# BOOKSY BRAIN - V253 (THE HTML ANALYTICS EDITION)
# VERZIÓ: V253 - HTML EMAIL ANALYTICS + NO MARKDOWN + ALL PREVIOUS FIXES (1M% INTEGRITY)

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os, time, requests, hashlib, re, json, random, unicodedata, html, urllib.parse, gc, chromadb, pytz, smtplib, traceback, sqlite3
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
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import markdownify
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

# --- CONFIG & CLIENTS ---
load_dotenv()
LOCAL_TZ = pytz.timezone('Europe/Bucharest')
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

CLAUDE_MODEL = "claude-sonnet-4-6"
XML_FEED_URL = "https://www.antikvarius.ro/wp-content/uploads/woo-feed/google/xml/booksyfullfeed.xml"
TEMP_FILE = "temp_feed.xml"
SOCIAL_MEMORY_FILE = "./booksy_db/social_memory.json"
CHAT_DB_FILE = "./booksy_db/chat_logs.db"
STORE_POLICIES_FILE = "./booksy_db/store_policies.json"
ADMIN_EMAILS = ["bookmankiado@gmail.com", "joomla900@gmail.com"]

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

# --- UTILS & GDPR FILTERS ---
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

def extract_xml_tag(text: str, tag: str) -> str:
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    return text.strip()

def clean_pii(text):
    if not text: return ""
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL TÖRÖLVE]', text)
    text = re.sub(r'(\+?\d{1,3}[\s-]?)?\(?\d{2,3}\)?[\s-]?\d{3}[\s-]?\d{3,4}', '[TELEFON TÖRÖLVE]', text)
    return text

def get_geo_from_ip(ip_address):
    if not ip_address or ip_address in ["127.0.0.1", "localhost", "::1"]: return "Ismeretlen", "Ismeretlen"
    try:
        r = requests.get(f"http://ip-api.com/json/{ip_address}?fields=countryCode,regionName", timeout=2)
        if r.status_code == 200:
            data = r.json()
            return data.get("countryCode", "Ismeretlen"), data.get("regionName", "Ismeretlen")
    except: pass
    return "Ismeretlen", "Ismeretlen"

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

# --- DB HANDLERS ---
class DBHandler:
    def __init__(self):
        if not os.path.exists("./booksy_db"): os.makedirs("./booksy_db")
        self.client = chromadb.PersistentClient(path="./booksy_db")
        self.collection = self.client.get_or_create_collection(name="booksy_collection_gemini_v2")

class AnalyticsDB:
    def __init__(self):
        self.db_path = CHAT_DB_FILE
        self._init_db()

    def _init_db(self):
        try:
            if not os.path.exists("./booksy_db"): os.makedirs("./booksy_db")
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS chat_logs
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                          session_id TEXT, user_msg TEXT, bot_reply TEXT, context_url TEXT,
                          geo_country TEXT, geo_region TEXT, ui_language TEXT, chat_language TEXT,
                          target_catalog TEXT, offered_book_ids TEXT, zero_match_flag BOOLEAN,
                          latency_ms INTEGER, device_type TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS analytics_reports
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, report_type TEXT, target_date TEXT,
                          content TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            conn.commit(); conn.close()
        except Exception as e: log_event(f"❌ AnalyticsDB Init Hiba: {e}")

    def log_chat(self, data: dict):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''INSERT INTO chat_logs 
                         (session_id, user_msg, bot_reply, context_url, geo_country, geo_region, 
                          ui_language, chat_language, target_catalog, offered_book_ids, zero_match_flag, latency_ms, device_type) 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (data.get('session_id'), data.get('user_msg'), data.get('bot_reply'), data.get('context_url'),
                       data.get('geo_country'), data.get('geo_region'), data.get('ui_language'), data.get('chat_language'),
                       data.get('target_catalog'), data.get('offered_book_ids'), data.get('zero_match_flag'),
                       data.get('latency_ms'), data.get('device_type')))
            conn.commit(); conn.close()
        except Exception as e: log_event(f"⚠️ Chat Log Hiba: {e}")

    def save_report(self, report_type: str, target_date: str, content: str):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("INSERT INTO analytics_reports (report_type, target_date, content) VALUES (?, ?, ?)", (report_type, target_date, content))
            conn.commit(); conn.close()
        except Exception as e: log_event(f"⚠️ Report Mentés Hiba: {e}")

    def get_logs_for_date(self, target_date_str: str):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT * FROM chat_logs WHERE date(timestamp) = ?", (target_date_str,))
            rows = c.fetchall()
            col_names = [description[0] for description in c.description]
            conn.close()
            return [dict(zip(col_names, row)) for row in rows]
        except Exception as e: log_event(f"⚠️ Log lekérdezési hiba: {e}"); return []

    def get_reports_for_period(self, report_type: str, date_prefix: str):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT content FROM analytics_reports WHERE report_type = ? AND target_date LIKE ?", (report_type, f"{date_prefix}%"))
            rows = c.fetchall()
            conn.close()
            return [r[0] for r in rows]
        except Exception as e: log_event(f"⚠️ Riport lekérdezési hiba: {e}"); return []

    def cleanup_old_logs(self):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            thirty_days_ago = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')
            c.execute("DELETE FROM chat_logs WHERE date(timestamp) < ?", (thirty_days_ago,))
            deleted = c.rowcount
            conn.commit(); conn.close()
            log_event(f"🧹 Takarítás: {deleted} db 30 napnál régebbi nyers log törölve (Riportok megmaradtak).")
        except Exception as e: log_event(f"⚠️ Takarítás hiba: {e}")

db_handler = DBHandler()
analytics_db = AnalyticsDB()

# --- AI ANALYTICS AGENT ---
class AIAnalyticsAgent:
    def __init__(self):
        self.claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.report_emails = ADMIN_EMAILS

    def _get_market_trends(self, context="napi"):
        prompt = (f"Keress rá a weben a legfrissebb e-kereskedelmi és könyvpiaci trendekre. SZIGORÚ prioritási "
                  f"sorrend a {context} adatokhoz: 1. Romániai piac, 2. Magyarországi piac, 3. Európai trendek, 4. Világpiac. "
                  f"Mik a legújabb keresett műfajok?")
        for attempt in range(3):
            try:
                res = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=[prompt])
                return res.text
            except Exception as e:
                if attempt < 2:
                    log_event(f"⚠️ Gemini API Hiba (Market Trends): {e}. Újra 3 mp múlva...")
                    time.sleep(3)
                else:
                    return "Piaci trendek lekérése sikertelen."

    def _send_analytics_email(self, subject: str, body: str):
        try:
            sender, password = os.getenv("SMTP_SENDER"), os.getenv("SMTP_PASSWORD")
            if not sender: return
            server = smtplib.SMTP(os.getenv("SMTP_SERVER", "mail.antikvarius.ro"), 26, timeout=20)
            server.starttls(); server.login(sender, password)
            for admin in self.report_emails:
                msg = MIMEMultipart()
                msg['From'] = f"{Header('Booksy Analytics', 'utf-8')} <{sender}>"
                msg['To'] = admin
                msg['Subject'] = Header(subject, 'utf-8')
                # --- V253: PLAIN TEXT HELYETT HTML FORMÁTUM A SZÉP MEGJELENÉSÉRT ---
                msg.attach(MIMEText(body, 'html', 'utf-8'))
                # ------------------------------------------------------------------
                server.send_message(msg)
            server.quit()
            log_event(f"📧 Analitika E-mail ({subject}) kiküldve.")
        except Exception as e: log_event(f"📧 Hiba Analitika küldésnél: {e}")

    def generate_daily_report(self):
        log_event("📊 Napi AI Analitika Indítása (T+1)...")
        target_dt = datetime.now(LOCAL_TZ) - timedelta(days=1)
        target_date_str = target_dt.strftime('%Y-%m-%d')
        
        logs = analytics_db.get_logs_for_date(target_date_str)
        market_trends = self._get_market_trends("napi")
        
        # --- V253: HTML FORMÁZÁSI SZABÁLY HOZZÁADVA ---
        analytics_rule = (f"KÖTELEZŐ SZABÁLY: A nyelvezet legyen üzleti, vezetői, laikusok számára is érthető, emberi! "
                          f"Zéró kód-zsargon vagy technikai kifejezés! Szigorúan TILOS olyan szavakat használni, mint "
                          f"'session_id', 'zero_match_flag', 'latencia ms-ban', 'geo_country', 'log bejegyzés' stb. "
                          f"Ehelyett fogalmazz így: 'Egy látogató', 'Nincs találat a raktárban', 'A válaszidő 29 másodperc volt'. "
                          f"Fókuszálj a tiszta üzleti összefüggésekre, marketing stratégiára és levonható következtetésekre. "
                          f"Szekciónként maximum a 3 legfontosabb, üzletileg kritikus megállapítást és akciótervet emeld ki röviden.\n"
                          f"KÖTELEZŐ HTML FORMÁZÁS: A teljes riportot SZIGORÚAN tiszta HTML kódban írd meg (használj <h2>, <h3>, <ul>, <li>, <strong>, <br> tageket)! "
                          f"NE használj Markdown-t (zéró csillag, zéró hashtag)! A felsorolásokat (pl. keresési témák) MINDIG <ul><li> listába tedd! "
                          f"NE tegyél markdown blokk (backticks) jelzést az elejére és végére, csak tisztán a HTML szöveget add vissza!")
        
        if not logs or len(logs) == 0:
            system_prompt = "Válságmenedzser és Üzleti Elemző vagy. Ma nulla interakció volt a chaten."
            user_msg = f"Piaci adatok: {market_trends}\n\nKészíts Napi Riportot arról, mi okozhatta a zéró forgalmat! Vizsgálj meg UX hibákat vagy piaci okokat. Készíts HTML listát!\n{analytics_rule}"
        elif len(logs) < 5:
            system_prompt = "E-kereskedelmi (CRO) és Marketing Elemző vagy."
            user_msg = f"Napi interakciók ({len(logs)} db):\n{logs}\n\nPiaci adatok: {market_trends}\n\nKészíts Napi Riportot! Fókusz: Hogyan vegyük rá az embereket a chat használatára? Javasolj egy bevonó stratégiát a RO/HU trendek alapján. Zéró markdown diagram.\n{analytics_rule}"
        else:
            system_prompt = "Profi Marketing Elemző, Webdesigner és Menedzsment Stratéga vagy."
            user_msg = (f"Napi interakciók ({len(logs)} db):\n{logs}\n\nPiaci trendek:\n{market_trends}\n\n"
                        f"Készíts átfogó Napi Riportot. Fókuszok:\n"
                        f"1. Földrajzi & Nyelvi Eloszlás (RO vs HU IP-k - Erdély fókusz).\n"
                        f"2. Készlet & Beszerzés (milyen könyveket kerestek hiába).\n"
                        f"3. Proaktív Frontend UX súrlódások (mit rontottunk el a boltban).\n"
                        f"4. 🔮 Webdevmk AI Előrejelzés a következő napokra!\n"
                        f"Szigorú forma: Zéró diagram. Csak jól tagolt HTML listák és százalékok.\n{analytics_rule}")

        for attempt in range(3):
            try:
                res = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=4096, system=system_prompt, messages=[{"role": "user", "content": user_msg}])
                report = res.content[0].text.strip()
                
                # Biztonsági tisztítás a markdown backtickek eltávolítására
                md_fence = "`" * 3
                report = report.replace(md_fence + "html", "").replace(md_fence, "").strip()
                
                analytics_db.save_report("daily", target_date_str, report)
                self._send_analytics_email(f"📊 Napi Booksy AI Üzleti Jelentés ({target_date_str})", report)
                analytics_db.cleanup_old_logs() 
                log_event("✅ Napi Analitika befejezve.")
                break
            except Exception as e:
                if attempt < 2:
                    log_event(f"⚠️ Claude API Hiba (Daily Report): {e}. Újra 3 mp múlva...")
                    time.sleep(3)
                else:
                    log_event(f"❌ Napi Analitika végleges hiba: {e}")

    def generate_monthly_report(self):
        now = datetime.now(LOCAL_TZ)
        last_month_dt = now.replace(day=1) - timedelta(days=1)
        target_month_str = last_month_dt.strftime('%Y-%m')
        
        log_event(f"📈 Havi AI Analitika Indítása ({target_month_str})...")
        daily_reports = analytics_db.get_reports_for_period("daily", target_month_str)
        if not daily_reports: return
        
        market_trends = self._get_market_trends("havi")
        compiled_reports = "\n\n---NAPI JELENTÉS---\n\n".join(daily_reports)
        
        analytics_rule = (f"KÖTELEZŐ SZABÁLY: A nyelvezet legyen üzleti, vezetői, laikusok számára is érthető, emberi! "
                          f"Zéró kód-zsargon vagy technikai kifejezés! Fókuszálj a tiszta üzleti összefüggésekre és levonható következtetésekre. "
                          f"Szekciónként maximum a 3 legfontosabb, üzletileg kritikus megállapítást emeld ki röviden.\n"
                          f"KÖTELEZŐ HTML FORMÁZÁS: A teljes riportot SZIGORÚAN tiszta HTML kódban írd meg! NE használj Markdown-t! "
                          f"A felsorolásokat MINDIG <ul><li> listába tedd! Csak tisztán a HTML szöveget add vissza!")
        
        prompt = (f"A mellékelt szöveg az elmúlt hónap összes napi jelentése. Piaci havi trendek: {market_trends}\n\n"
                  f"Készíts vezetői HAVI JELENTÉST. Fókusz: Forgalmi források, erdélyi (RO IP, HU nyelvű) piac, hiánycikkek, "
                  f"és UX frontend javaslatok. Végezetül: '🔮 Webdevmk AI Előrejelzés a következő hónapra'. Csak HTML listák és százalékok.\n{analytics_rule}")
        
        for attempt in range(3):
            try:
                res = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=6000, system="Üzleti Stratéga vagy.", messages=[{"role": "user", "content": prompt}])
                report = res.content[0].text.strip()
                
                md_fence = "`" * 3
                report = report.replace(md_fence + "html", "").replace(md_fence, "").strip()
                
                analytics_db.save_report("monthly", target_month_str, report)
                self._send_analytics_email(f"📈 HAVI Booksy AI Menedzsment Riport ({target_month_str})", report)
                log_event("✅ Havi Analitika befejezve.")
                break
            except Exception as e:
                if attempt < 2:
                    log_event(f"⚠️ Claude API Hiba (Monthly Report): {e}. Újra 3 mp múlva...")
                    time.sleep(3)
                else:
                    log_event(f"❌ Havi Analitika végleges hiba: {e}")

    def generate_yearly_report(self):
        target_year_str = str(datetime.now(LOCAL_TZ).year - 1)
        log_event(f"👑 ÉVES AI Stratégiai Analitika Indítása ({target_year_str})...")
        
        monthly_reports = analytics_db.get_reports_for_period("monthly", target_year_str)
        if not monthly_reports: return
        
        market_trends = self._get_market_trends("éves jövőkutatási")
        compiled_reports = "\n\n---HAVI JELENTÉS---\n\n".join(monthly_reports)
        
        analytics_rule = (f"KÖTELEZŐ SZABÁLY: A nyelvezet legyen üzleti, vezetői, laikusok számára is érthető, emberi! "
                          f"Zéró kód-zsargon vagy technikai kifejezés! Fókuszálj a tiszta üzleti összefüggésekre és levonható következtetésekre. "
                          f"Szekciónként maximum a 3 legfontosabb, üzletileg kritikus megállapítást emeld ki röviden.\n"
                          f"KÖTELEZŐ HTML FORMÁZÁS: A teljes riportot SZIGORÚAN tiszta HTML kódban írd meg! NE használj Markdown-t! "
                          f"A felsorolásokat MINDIG <ul><li> listába tedd! Csak tisztán a HTML szöveget add vissza!")
        
        prompt = (f"A mellékelt szöveg az elmúlt év 12 havi jelentése. Globális Éves Trendek: {market_trends}\n\n"
                  f"Készíts ÉVES Menedzsment Riportot! Értékeld a ROI-t, terjeszkedési statisztikákat (RO vs HU), "
                  f"frontend UX tanulságokat, majd egy '🔮 Webdevmk AI Éves Előrejelzés és Beszerzés' szekciót. "
                  f"Csak HTML listás, diagrammentes struktúra.\n{analytics_rule}")
        
        for attempt in range(3):
            try:
                res = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=8000, system="Vezérigazgatói Tanácsadó vagy.", messages=[{"role": "user", "content": prompt}])
                report = res.content[0].text.strip()
                
                md_fence = "`" * 3
                report = report.replace(md_fence + "html", "").replace(md_fence, "").strip()
                
                analytics_db.save_report("yearly", target_year_str, report)
                self._send_analytics_email(f"👑 ÉVES Booksy AI Stratégiai Iránytű ({target_year_str})", report)
                log_event("✅ Éves Analitika befejezve.")
                break
            except Exception as e:
                if attempt < 2:
                    log_event(f"⚠️ Claude API Hiba (Yearly Report): {e}. Újra 3 mp múlva...")
                    time.sleep(3)
                else:
                    log_event(f"❌ Éves Analitika végleges hiba: {e}")

# --- UPDATER & LIVE POLICY SCRAPER ---
class AutoUpdater:
    def __init__(self, db: DBHandler): self.db = db
    
    def fetch_store_policies(self):
        log_event("📖 [RAG] Céges Kódex (ÁSZF, Szállítás, Fizetés, Kapcsolat) letöltése...")
        urls = [
            "https://www.antikvarius.ro/hu/kapcsolat/",
            "https://www.antikvarius.ro/hu/szallitasi-informaciok/",
            "https://www.antikvarius.ro/hu/fizetesi-informaciok/",
            "https://www.antikvarius.ro/hu/altalanos-szerzodesi-es-felhasznalasi-feltetelek/"
        ]
        policies_text = ""
        for url in urls:
            try:
                cache_buster = int(time.time())
                r = requests.get(f"{url}?v={cache_buster}", headers={"Cache-Control": "no-cache"}, timeout=20)
                if r.status_code == 200:
                    soup = BeautifulSoup(r.content, 'html.parser')
                    for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                        script.extract()
                    text = soup.get_text(separator=' ', strip=True)
                    policies_text += f"\n\n--- FORRÁS: {url} ---\n{text[:6000]}"
            except Exception as e:
                log_event(f"⚠️ Hiba a {url} beolvasásakor: {e}")
        
        if policies_text:
            with open(STORE_POLICIES_FILE, "w", encoding="utf-8") as f:
                json.dump({"policies": policies_text}, f, ensure_ascii=False)
            log_event("✅ Céges Kódex sikeresen frissítve az élő weboldalról (Cache-Busting aktív).")

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
                            "stock": stock_status, "type": "book"
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

# --- SOCIAL AGENT ---
class BooksySocialAgent:
    def __init__(self, db: DBHandler):
        self.db = db
        self.claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def send_error_email(self, error_details):
        try:
            sender, password = os.getenv("SMTP_SENDER"), os.getenv("SMTP_PASSWORD")
            if not sender: return
            server = smtplib.SMTP(os.getenv("SMTP_SERVER", "mail.antikvarius.ro"), 26, timeout=20)
            server.starttls(); server.login(sender, password)
            for admin in ADMIN_EMAILS:
                msg = MIMEMultipart()
                msg['From'] = f"{Header('Booksy AI', 'utf-8')} <{sender}>"
                msg['To'] = admin
                msg['Subject'] = Header(f"⚠️ KRITIKUS HIBA: Booksy Social Agent ({datetime.now(LOCAL_TZ).strftime('%Y-%m-%d')})", 'utf-8')
                body = f"Üdv!\n\nA napi Facebook vázlat generálása során váratlan hiba történt.\n\nRészletek:\n\n{error_details}"
                msg.attach(MIMEText(body, 'plain', 'utf-8'))
                server.send_message(msg)
            server.quit()
        except Exception as e: log_event(f"📧 Hiba az error e-mailnél: {e}")

    def send_morning_email(self, post_text, memory_links, hook_text="", reels_text=""):
        try:
            sender, password = os.getenv("SMTP_SENDER"), os.getenv("SMTP_PASSWORD")
            if not sender: return
            links_body = ""
            for b in memory_links:
                author_display = f"{b['author']} - " if b.get('author') and b['author'] != 'Ismeretlen' else ""
                links_body += f"📖 {author_display}{b['title']}\n{b.get('marketing_desc', '')}\n🔗 {b['url']}\n\n"

            server = smtplib.SMTP(os.getenv("SMTP_SERVER", "mail.antikvarius.ro"), 26, timeout=20)
            server.starttls(); server.login(sender, password)
            for admin in ADMIN_EMAILS:
                msg = MIMEMultipart()
                msg['From'] = f"{Header('Booksy AI', 'utf-8')} <{sender}>"
                msg['To'] = admin
                msg['Subject'] = Header(f"✅ Booksy Social Vázlatok - {datetime.now(LOCAL_TZ).strftime('%Y-%m-%d')}", 'utf-8')
                body = (
                    f"Üdv!\n\nA Facebook vázlatok elkészültek a Drafts mappába.\n"
                    f"=========================\nFŐ KÉPES POSZT SZÖVEGE:\n=========================\n{post_text}\n\n"
                    f"=========================\nREELS VIDEÓ SZÖVEGE:\n=========================\n{reels_text}\n\n"
                    f"=========================\nDIREKT KOMMENTBE MEGY (EGYBEN):\n=========================\n{hook_text}\n\n{links_body.strip()}"
                )
                msg.attach(MIMEText(body, 'plain', 'utf-8'))
                server.send_message(msg)
            server.quit()
            log_event("Értesítő e-mail sikeresen elküldve.")
        except Exception as e: log_event(f"📧 Email hiba: {e}")

    def _prepare_visual_layers(self, raw_img_path, overlay_path, fallback_path, title, author):
        try:
            img = PIL.Image.open(raw_img_path).convert("RGBA")
            width, height = img.size
            font_path = "Montserrat-Bold.ttf"
            best_font = ImageFont.load_default()
            use_bbox = os.path.exists(font_path)
            
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
            return True
        except: return False

    def _create_video(self, raw_img_path, overlay_path, out_path):
        if not MOVIEPY_AVAILABLE: return False
        try:
            gc.collect() 
            clip = ImageClip(raw_img_path).set_duration(5)
            zoomed = clip.resize(lambda t: 1 + 0.03 * t).set_position('center')
            fixed_bg = CompositeVideoClip([zoomed], size=(1080, 1080)).set_duration(5) 
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
            gc.collect() 
            return True
        except: return False

    def _trigger_fb_comment(self, force_post_id=None):
        try:
            fb_id, fb_token = os.getenv("FB_PAGE_ID"), os.getenv("FB_PAGE_TOKEN")
            if not os.path.exists(SOCIAL_MEMORY_FILE): return {"reply": "❌ Nincs memória fájl.", "products": [], "zero_match_flag": True}
            with open(SOCIAL_MEMORY_FILE, "r", encoding="utf-8") as f: memory = json.load(f)
            
            if not memory.get("links") or len(memory["links"]) == 0:
                return {"reply": "❌ Memória fájl hibás, nincs könyv.", "products": [], "zero_match_flag": True}

            target_post_id = force_post_id
            if not target_post_id:
                media_id = memory.get("media_id") 
                fingerprint_search = normalize_fingerprint(memory.get("fingerprint", ""))
                endpoints = [
                    f"https://graph.facebook.com/v19.0/{fb_id}/published_posts?access_token={fb_token}&limit=15&fields=id,message,attachments",
                    f"https://graph.facebook.com/v19.0/{fb_id}/feed?access_token={fb_token}&limit=15&fields=id,message,attachments"
                ]
                
                found = False
                for ep in endpoints:
                    if found: break
                    try:
                        r = requests.get(ep)
                        if r.status_code != 200: continue
                        for p in r.json().get('data', []):
                            if media_id:
                                for att in p.get('attachments', {}).get('data', []):
                                    if str(media_id) in str(att.get('target', {}).get('id', '')): found = True; break
                            if not found and fingerprint_search and fingerprint_search in normalize_fingerprint(p.get("message", "")): found = True
                            
                            if found: target_post_id = p["id"]; break
                    except: pass
            
            if not target_post_id: return {"reply": "❌ Célpont poszt nem található.", "products": [], "zero_match_flag": True}

            payload_text = memory.get("hook_text", "📚 A mai válogatásunk kincseit itt találjátok! 👇") + "\n\n"
            for book in memory.get("links", []):
                author = f"{book['author']} - " if (book.get('author') and book['author'] != 'Ismeretlen') else ""
                payload_text += f"📖 {author}{book['title']}\n{book.get('marketing_desc', '')}\n🔗 {book['url']}\n\n"
            
            r_res = requests.post(f"https://graph.facebook.com/v19.0/{target_post_id}/comments", data={'access_token': fb_token, 'message': payload_text.strip()})
            if "id" in r_res.json(): return {"reply": "✅ Komment sikeresen kiment!", "products": [], "zero_match_flag": False}
            else: return {"reply": f"❌ FB hiba: {r_res.text}", "products": [], "zero_match_flag": True}
        except Exception as e: return {"reply": f"❌ Hiba: {e}", "products": [], "zero_match_flag": True}

    def run_night_generation(self):
        log_event("Agentic Generálás (V253 HTML Analytics Edition)...")
        raw_img_path = "social_raw.jpg"; overlay_path = "social_overlay.png"; fallback_img_path = "social_fallback.jpg"; vid_path = "social_video.mp4"
        
        try:
            now_dt = datetime.now(LOCAL_TZ)
            hu_months = ["Január", "Február", "Március", "Április", "Május", "Június", "Július", "Augusztus", "Szeptember", "Október", "November", "December"]
            hu_date_str = f"{hu_months[now_dt.month-1]} {now_dt.day}."
            today_date = now_dt.strftime('%B %d')
            
            r_wiki = requests.get(f"https://en.wikipedia.org/api/rest_v1/feed/onthisday/births/{now_dt.strftime('%m/%d')}", headers={'User-Agent': 'BooksyBot/1.0'})
            wiki_text = "Nem található adat."
            if r_wiki.status_code == 200:
                births = [p.get('text', '') for p in r_wiki.json().get('births', []) if any(kw in p.get('text', '').lower() for kw in ['writer', 'author', 'poet', 'novelist'])]
                wiki_text = "\n".join(births[:30])
            
            author_prompt = (
                f"Ma {today_date} van. Itt egy nyers lista a Wikipédiáról a mai napon született személyekről: {wiki_text}\n"
                f"Végezz élő internetes kutatást! Válaszd ki pontosan a legrelevánsabb 6 embert, de KIZÁRÓLAG KÖNYVÍRÓKAT "
                f"(regényíró, költő, esszéista, sci-fi író, tudományos-ismeretterjesztő). SZIGORÚAN TILOS listázni filmrendezőket, "
                f"képregényrajzolókat, színészeket, zenészeket, modelleket, animátorokat, forgatókönyvírókat! Csak klasszikus könyvírókat!\n"
                f"Prioritás: Ha a listában van magyar vagy román író, kötelezően vedd be! A többit klasszikusokkal töltsd fel.\n"
                f"Készíts róluk 'mini lexikon' megemlékezést (1-2 mondat/író). SZIGORÚ KIMENET: Csak și kizárólag XML formátum:\n"
                f"<authors><author><name>Író Neve</name><nationality>Nemzetiség</nationality><bio>Rövid életrajz și műve.</bio></author></authors>"
            )
            
            gem_authors_res = None
            for attempt in range(3):
                try:
                    gem_authors_res = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=[author_prompt])
                    break
                except Exception as e:
                    wait_time = 3 * (attempt + 1)
                    if attempt < 2:
                        log_event(f"⚠️ Gemini Hiba (Szerzők): {e}. Újra {wait_time} mp múlva...")
                        time.sleep(wait_time)
                    else:
                        raise Exception(f"Végzetes Gemini hiba a szerzőknél: {e}")
            
            authors_list = safe_authors_parse(gem_authors_res.text)[:6]

            selected_books, seen_ids = [], set()
            for author in authors_list:
                vec = None
                for attempt in range(3):
                    try:
                        vec = gemini_client.models.embed_content(model="gemini-embedding-001", contents=author['name'], config=types.EmbedContentConfig(output_dimensionality=768)).embeddings[0].values
                        break
                    except Exception as e:
                        wait_time = 3 * (attempt + 1)
                        if attempt < 2: time.sleep(wait_time)
                        else: vec = None
                
                if vec:
                    res = self.db.collection.query(query_embeddings=[vec], n_results=3, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
                    if res['ids'] and res['ids'][0]:
                        for p_target in res['metadatas'][0]:
                            if p_target['id'] not in seen_ids:
                                selected_books.append(p_target); seen_ids.add(p_target['id']); break

            if len(selected_books) < 3:
                vec_fb = None
                for attempt in range(3):
                    try:
                        vec_fb = gemini_client.models.embed_content(model="gemini-embedding-001", contents="klasszikus irodalom", config=types.EmbedContentConfig(output_dimensionality=768)).embeddings[0].values
                        break
                    except Exception as e:
                        wait_time = 3 * (attempt + 1)
                        if attempt < 2: time.sleep(wait_time)
                        else: vec_fb = None
                
                if vec_fb:
                    res_fb = self.db.collection.query(query_embeddings=[vec_fb], n_results=10, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
                    if res_fb['ids'] and res_fb['ids'][0]:
                        for p_target in res_fb['metadatas'][0]:
                            if p_target['id'] not in seen_ids:
                                selected_books.append(p_target); seen_ids.add(p_target['id'])
                            if len(selected_books) >= 5: break

            main_book = selected_books[0]
            for b in selected_books:
                desc_prompt = f"Könyv: {b['title']} - {b['author']}. Írj EGY zamatos magyar marketing mondatot! ZÉRÓ MARKDOWN."
                for attempt in range(3):
                    try:
                        b['marketing_desc'] = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=250, messages=[{"role": "user", "content": desc_prompt}]).content[0].text.strip()
                        break
                    except Exception as e:
                        wait_time = 3 * (attempt + 1)
                        if attempt < 2:
                            log_event(f"⚠️ Claude Hiba (Marketing): {e}. Újra {wait_time} mp múlva...")
                            time.sleep(wait_time)
                        else:
                            b['marketing_desc'] = "Egy lenyűgöző ritkaság a kínálatunkból, amely minden könyvtár méltó dísze lehet."
            
            gem_res_text = "Antique book."
            for attempt in range(3):
                try:
                    gem_res_text = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=[f"Elemezd: '{main_book['title']}'. Angol vizuális összefoglaló."]).text
                    break
                except Exception as e:
                    wait_time = 3 * (attempt + 1)
                    if attempt < 2: time.sleep(wait_time)
            
            c_res_text = "Antique book on a dark wooden table lit by a single candle, highly detailed."
            for attempt in range(3):
                try:
                    visual_prompt_instructions = (
                        f"Képzeld el, hogy te egy Oscar-díjas operatőr vagy. Elemzés a könyvről: {gem_res_text}. "
                        f"Tervezz meg EGYETLEN lélegzetelállító filmkockát, ami a könyv csúcsjelenetét (climax) ábrázolja "
                        f"valós helyszínen, valós szereplőkkel. Írj hozzá angol nyelvű képgeneráló promptot!\n"
                        f"KÖTELEZŐ VIZUÁLIS SZABÁLYOK: Shot on 35mm anamorphic lens, raw photography, hyper-realistic, "
                        f"cinematography, natural lighting, cool color grading, real human skin texture, no CGI, no 3D render. "
                        f"Szigorúan kerüld a tipikus műanyag vagy meleg sárgás DALL-E hatást!\n"
                        f"KÖTELEZŐ SZÖVEG SZABÁLY: A kompozíció tartalmazhat környezeti szöveget (pl. egy régi cégér, újság, "
                        f"falon lévő levél részlete), DE ez a szöveg SZIGORÚAN a könyv történelmi/földrajzi kontextusának "
                        f"megfelelő nyelven (magyarul vagy románul) és helyesírással szerepeljen! Zéró hallucinált angol szöveg!"
                    )
                    c_res_text = self.claude.messages.create(
                        model=CLAUDE_MODEL, 
                        max_tokens=300, 
                        messages=[{"role": "user", "content": visual_prompt_instructions}]
                    ).content[0].text
                    break
                except Exception as e:
                    wait_time = 3 * (attempt + 1)
                    if attempt < 2: time.sleep(wait_time)
            
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            img_url = None
            
            for attempt in range(3):
                try:
                    img_res = openai_client.images.generate(model="gpt-image-2", prompt=c_res_text, size="1024x1024", quality="high", n=1)
                    img_url = img_res.data[0].url
                    break
                except Exception as e:
                    wait_time = 3 * (attempt + 1)
                    if attempt < 2:
                        log_event(f"⚠️ GPT-Image-2 Hiba: {e}. Újra {wait_time} mp múlva...")
                        time.sleep(wait_time)
                    else:
                        log_event(f"❌ GPT-Image-2 végleg elszállt. Váltás DALL-E 3 tartalékra!")

            if not img_url:
                try:
                    img_res = openai_client.images.generate(model="dall-e-3", prompt=c_res_text, size="1024x1024", quality="hd", n=1)
                    img_url = img_res.data[0].url
                except Exception as e:
                    log_event(f"⚠️ DALL-E 3 Hiba: {e}. Váltás alapértelmezett DALL-E 3 promptra.")
                    img_res = openai_client.images.generate(model="dall-e-3", prompt="A highly realistic cinematic shot on 35mm film of a classical library scene, natural cool lighting.", size="1024x1024", quality="standard", n=1)
                    img_url = img_res.data[0].url

            r_img = requests.get(img_url, timeout=90)
            with open(raw_img_path, 'wb') as f: f.write(r_img.content)
            img_obj = PIL.Image.open(raw_img_path)
            PIL.ImageOps.fit(img_obj, (1080, 1080), PIL.Image.Resampling.LANCZOS).save(raw_img_path)
            self._prepare_visual_layers(raw_img_path, overlay_path, fallback_img_path, main_book['title'], main_book.get('author', ''))
            has_video = self._create_video(raw_img_path, overlay_path, vid_path)

            authors_text = "\n".join([f"📖 {a['name'].upper()}: {a['bio']}" for a in authors_list])
            draft_prompt = f"Írj FB posztot. Cím: {hu_date_str} — IRODALMI NAPTÁR\n{authors_text}\nZéró markdown!"
            
            draft_text = f"Cím: {hu_date_str} — IRODALMI NAPTÁR\n{authors_text}"
            for attempt in range(3):
                try:
                    draft_text = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=2000, system="CopySEO.", messages=[{"role": "user", "content": draft_prompt}]).content[0].text
                    break
                except Exception as e:
                    wait_time = 3 * (attempt + 1)
                    if attempt < 2: time.sleep(wait_time)
            
            lector_prompt = (f"Lektoráld szigorúan! ZÉRÓ MARKDOWN. "
                             f"KÖTELEZŐ SZABÁLY: A tiszta, végleges posztot KIZÁRÓLAG <final_post> și </final_post> tagek közé tedd! "
                             f"SZIGORÚAN TILOS a poszt végére jókívánságot sau lezárást írni!\n\nVÁZLAT:\n{draft_text}")
            
            raw_lexicon = f"<final_post>{draft_text}</final_post>"
            for attempt in range(3):
                try:
                    raw_lexicon = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=2000, messages=[{"role": "user", "content": lector_prompt}]).content[0].text
                    break
                except Exception as e:
                    wait_time = 3 * (attempt + 1)
                    if attempt < 2: time.sleep(wait_time)
            lektored_lexicon = extract_xml_tag(raw_lexicon, "final_post") 

            bridge_prompt = (f"Írj egy 3 mondatos átvezetést ehhez a könyvhöz: „{main_book['title']}”. ZÉRÓ MARKDOWN. "
                             f"A tiszta szöveget tedd <bridge> și </bridge> tagek közé!")
            
            raw_bridge = "<bridge>A mai válogatásunkban rejlő kincsek felfedezésre várnak.</bridge>"
            for attempt in range(3):
                try:
                    raw_bridge = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=500, messages=[{"role": "user", "content": bridge_prompt}]).content[0].text
                    break
                except Exception as e:
                    wait_time = 3 * (attempt + 1)
                    if attempt < 2: time.sleep(wait_time)
            bridge_text = extract_xml_tag(raw_bridge, "bridge")

            post_text = lektored_lexicon
            if not re.search(r'\[Érzés:.*?\]', post_text): post_text = "[Érzés: inspirált 🌟]\n\n" + post_text
            post_text += "\n\n" + bridge_text

            book_titles = ", ".join([b['title'] for b in selected_books])
            hook_prompt = (f"Te egy értékesítő vagy. Írj 1-2 mondatos bevezetőt a linkek elé: {book_titles}. "
                           f"A linkek már ott lesznek alattad, így NE kérd be őket tőlem și NE kérdezz vissza! "
                           f"A tiszta szöveget KIZÁRÓLAG <hook> și </hook> tagek közé tedd! Zéró markdown, használj 👇 emojit.")
            
            raw_hook = "<hook>📚 A mai válogatásunk kincseit itt találjátok! 👇</hook>"
            for attempt in range(3):
                try:
                    raw_hook = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=150, messages=[{"role": "user", "content": hook_prompt}]).content[0].text
                    break
                except Exception as e:
                    wait_time = 3 * (attempt + 1)
                    if attempt < 2: time.sleep(wait_time)
            hook_text = extract_xml_tag(raw_hook, "hook")

            reels_prompt = (f"Ma {hu_date_str} van! Írj 2-3 mondatos pörgős videó szöveget: {', '.join([a['name'].upper() for a in authors_list[:3]])} ma született. Zéró markdown. "
                            f"A tiszta szöveget KIZÁRÓLAG <reels_text> și </reels_text> tagek közé tedd!")
            
            raw_reels = f"<reels_text>Fedezd fel a mai napon, {hu_date_str} született klasszikusokat!</reels_text>"
            for attempt in range(3):
                try:
                    raw_reels = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=250, messages=[{"role": "user", "content": reels_prompt}]).content[0].text
                    break
                except Exception as e:
                    wait_time = 3 * (attempt + 1)
                    if attempt < 2: time.sleep(wait_time)
            reels_text = extract_xml_tag(raw_reels, "reels_text") + "\n\nRészletek, könyvajánló și a napi irodalmi lexikon a legújabb képes posztunkban a feeden! 👇"

            memory_data = {"fingerprint": post_text[:100], "hook_text": hook_text, "links": [{"id": b['id'], "title": b['title'], "author": b['author'], "url": b['url'], "marketing_desc": b.get('marketing_desc', '')} for b in selected_books]}
            fb_id, fb_token = os.getenv("FB_PAGE_ID"), os.getenv("FB_PAGE_TOKEN")
            
            upload_img = fallback_img_path if os.path.exists(fallback_img_path) else raw_img_path
            r_photo = requests.post(f"https://graph.facebook.com/v19.0/{fb_id}/photos", data={'access_token': fb_token, 'published': 'false'}, files={'source': open(upload_img, 'rb')})
            if r_photo.status_code == 200:
                photo_fbid = str(r_photo.json().get('id'))
                post_data = {'access_token': fb_token, 'message': post_text, 'published': 'false', 'unpublished_content_type': 'DRAFT', 'attached_media[0]': f'{{"media_fbid":"{photo_fbid}"}}'}
                r_draft = requests.post(f"https://graph.facebook.com/v19.0/{fb_id}/feed", data=post_data)
                if r_draft.status_code == 200: memory_data['media_id'] = str(r_draft.json().get('id'))
                else: memory_data['media_id'] = photo_fbid
            
            if has_video:
                requests.post(f"https://graph.facebook.com/v19.0/{fb_id}/videos", data={'access_token': fb_token, 'description': reels_text, 'published': 'false', 'unpublished_content_type': 'DRAFT'}, files={'source': open(vid_path, 'rb')})
            
            with open(SOCIAL_MEMORY_FILE, "w", encoding="utf-8") as f: json.dump(memory_data, f, ensure_ascii=False)
            self.send_morning_email(post_text, memory_data['links'], hook_text, reels_text); log_event("Kész.")
            
        except Exception as e:
            self.send_error_email(traceback.format_exc())
        finally:
            for p in [raw_img_path, overlay_path, fallback_img_path, vid_path]:
                if os.path.exists(p): os.remove(p)

class BooksyBrain:
    def __init__(self, db: DBHandler):
        self.db = db
        self.claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def process(self, msg, context_url, session_id):
        if msg.strip().startswith("/booklink"):
            parts = msg.strip().split()
            admin_pass = os.getenv("COMMENT_PASSWORD", "admin123")
            if len(parts) >= 2 and parts[1] == admin_pass:
                force_id = parts[2] if len(parts) >= 3 else None
                agent = BooksySocialAgent(self.db)
                return agent._trigger_fb_comment(force_id)
            else: return {"reply": "🤖 Téves parancs sau hibás jelszó.", "products": [], "zero_match_flag": True}
        if msg.strip().startswith("/"): return {"reply": "🤖 Rendszerparancs felismerve.", "products": [], "zero_match_flag": True}

        try:
            policy_text = "A céges szabályzatok jelenleg nem elérhetők."
            if os.path.exists(STORE_POLICIES_FILE):
                with open(STORE_POLICIES_FILE, "r", encoding="utf-8") as f:
                    policy_text = json.load(f).get("policies", "")

            vec = None
            for attempt in range(3):
                try:
                    vec_req = gemini_client.models.embed_content(model="gemini-embedding-001", contents=msg, config=types.EmbedContentConfig(output_dimensionality=768))
                    vec = vec_req.embeddings[0].values
                    break
                except Exception as e:
                    wait_time = 3 * (attempt + 1)
                    if attempt < 2: time.sleep(wait_time)

            db_res = {'ids': [], 'metadatas': []}
            if vec:
                db_res = self.db.collection.query(query_embeddings=[vec], n_results=4, where={"$and": [{"stock": "instock"}, {"type": "book"}]})
            
            zero_match = True
            raw_products = []
            context_text = "Nem találtam megfelelő könyvet a raktárban."
            
            if db_res['ids'] and db_res['ids'][0]:
                zero_match = False
                raw_products = db_res['metadatas'][0]
                
                for p in raw_products:
                    if 'image_url' in p and 'image' not in p:
                        p['image'] = p['image_url']

                context_text = "\n".join([f"Könyv: {p['title']} - {p.get('author','')} - Ár: {p.get('price','')}. Infó: {p.get('text_preview','')}" for p in raw_products])
            
            prompt = (f"Te Booksy vagy, az Antikvarius.ro profi asszisztense. A felhasználó kérdése: '{msg}'.\n\n"
                      f"<company_policies>\n{policy_text}\n</company_policies>\n\n"
                      f"SZIGORÚ SZABÁLYOK (SÉRTHETETLEN):\n"
                      f"1. FIGYELEM: Ha a kérdés adminisztratív (szállítás, cím, ÁSZF), KÖTELEZŐEN a <policy_only> taggel KEZDD A VÁLASZOD! Ne ajánlj könyveket, ha a kérdés csak a szállítási díjra sau a bolt címére vonatkozik!\n"
                      f"2. Ha a kérdés szállításra, fizetésre, kapcsolatra sau ÁSZF-re vonatkozik, KIZÁRÓLAG a <company_policies> alapján válaszolj! 0% hallucináció.\n"
                      f"3. A szállítás díja fix! Nincs ingyenes szállítás semmilyen súlyra/összegre. Kommunikáld marketingesen: mivel fix a díj, minél több könyvet vesznek, annál jobban megéri! "
                      f"KÖTELEZŐ SZABÁLY: Utánvétes fizetés (Ramburs / Plata la livrare) KIZÁRÓLAG Románián belül lehetséges! Más országokba (pl. HU, EU) CSAK online bankkártyás fizetés engedélyezett!\n"
                      f"4. Kiemelten ügyelj a magyar szakmai terminológiára și a helyesírásra! Tilos a gépelési hiba sau nem létező, értelmetlen ragozott szavak használata!\n"
                      f"5. A választ kötelezően AZON A NYELVEN fogalmazd meg, ahogy a felhasználó kérdezett!\n\n"
                      f"Raktár infó:\n{context_text}\n\n"
                      f"Zéró markdown!")
            
            reply_text = "Sajnos hiba történt. Kérlek próbáld újra!"
            for attempt in range(3):
                try:
                    reply_res = self.claude.messages.create(model=CLAUDE_MODEL, max_tokens=1000, system="Professional CopySEO tone. Multi-language Support.", messages=[{"role": "user", "content": prompt}])
                    reply_text = reply_res.content[0].text.strip()
                    break
                except Exception as e:
                    wait_time = 3 * (attempt + 1)
                    if attempt < 2:
                        log_event(f"⚠️ Claude Hiba (RAG): {e}. Újra {wait_time} mp múlva...")
                        time.sleep(wait_time)
                    else:
                        log_event(f"❌ Végzetes hiba a chaten: {e}")

            final_products = []
            if "<policy_only>" in reply_text:
                reply_text = reply_text.replace("<policy_only>", "").replace("</policy_only>", "").strip()
                final_products = []
            else:
                seen_titles = set()
                for p in raw_products:
                    clean_title = p.get('title', '').strip().lower()
                    if clean_title not in seen_titles:
                        seen_titles.add(clean_title)
                        final_products.append(p)

            return {"reply": reply_text, "products": final_products, "zero_match_flag": zero_match}
        except Exception as e:
            log_event(f"Bot hiba: {e}")
            return {"reply": "Sajnos hiba történt. Kérlek próbáld újra!", "products": [], "zero_match_flag": True}

# --- MASTER CASCADE & SCHEDULING ---
updater = AutoUpdater(db_handler)
bot = BooksyBrain(db_handler)
social_agent = BooksySocialAgent(db_handler)
analytics_agent = AIAnalyticsAgent()
scheduler = BackgroundScheduler()

def master_morning_routine():
    log_event("🌅 Master Láncreakció Indítása (V253)")
    updater.fetch_store_policies()
    
    # --- V253: XML SYNC FELFÜGGESZTVE TESZTELÉSHEZ ---
    # try:
    #     sync_success = updater.run_daily_update()
    #     if not sync_success: log_event("⚠️ Szinkronizációs hiba, korábbi adatok használata.")
    # except Exception as e: log_event(f"⚠️ Váratlan hiba a szinkronnál: {e}")
    # --------------------------------------------------
    
    social_agent.run_night_generation()

def daily_analytics_job():
    try: analytics_agent.generate_daily_report()
    except Exception as e: log_event(f"⚠️ Napi Analitika Hiba: {e}")

def monthly_analytics_job():
    if datetime.now(LOCAL_TZ).day == 1:
        try: analytics_agent.generate_monthly_report()
        except Exception as e: log_event(f"⚠️ Havi Analitika Hiba: {e}")

def yearly_analytics_job():
    now = datetime.now(LOCAL_TZ)
    if now.month == 1 and now.day == 1:
        try: analytics_agent.generate_yearly_report()
        except Exception as e: log_event(f"⚠️ Éves Analitika Hiba: {e}")

# --- FASTAPI APP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(master_morning_routine, CronTrigger(hour=7, minute=0, timezone=LOCAL_TZ))
    scheduler.add_job(daily_analytics_job, CronTrigger(hour=8, minute=0, timezone=LOCAL_TZ))
    scheduler.add_job(monthly_analytics_job, CronTrigger(hour=8, minute=15, timezone=LOCAL_TZ))
    scheduler.add_job(yearly_analytics_job, CronTrigger(hour=8, minute=30, timezone=LOCAL_TZ))
    scheduler.start(); yield; scheduler.shutdown()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_headers=["*"], allow_methods=["*"])

class ChatRequest(BaseModel): 
    message: str; context_url: Optional[str] = ""; session_id: Optional[str] = ""
    device_type: Optional[str] = "Desktop"; ui_lang: Optional[str] = "hu"
    chat_lang: Optional[str] = "hu"; target_catalog: Optional[str] = "mixed"

class InitRequest(BaseModel): url: str; session_id: str; ui_lang: str = "hu"

@app.get("/")
def home(): return {"status": "V253 Online (HTML Analytics Edition)", "project": "Booksy"}

@app.post("/chat")
def chat(req: ChatRequest, request: Request): 
    start_time = time.time()
    bot_response = bot.process(req.message, req.context_url, req.session_id)
    latency = int((time.time() - start_time) * 1000)
    
    if req.message.strip().startswith("/"): 
        return bot_response
    
    client_ip = request.client.host if request.client else None
    geo_country, geo_region = get_geo_from_ip(client_ip)
    safe_user_msg = clean_pii(req.message)
    offered_ids = ",".join([p.get("id", "") for p in bot_response.get("products", [])]) if bot_response.get("products") else ""
    
    log_data = {
        "session_id": req.session_id, "user_msg": safe_user_msg, "bot_reply": bot_response.get("reply", "")[:200],
        "context_url": req.context_url, "geo_country": geo_country, "geo_region": geo_region,
        "ui_language": req.ui_lang, "chat_language": req.chat_lang, "target_catalog": req.target_catalog,
        "offered_book_ids": offered_ids, "zero_match_flag": bot_response.get("zero_match_flag", False),
        "latency_ms": latency, "device_type": req.device_type
    }
    analytics_db.log_chat(log_data)
    return bot_response

@app.post("/init-chat")
def init_chat(req: InitRequest): return {"ui_lang": req.ui_lang, "bubble_text": "Szia!", "placeholder": "Keresel valamit?"}

@app.post("/test-social-night")
def test_night(bt: BackgroundTasks): 
    bt.add_task(social_agent.run_night_generation)
    return {"status": "V253 Social Night Started"}

@app.post("/test-cascade")
def test_cascade(bt: BackgroundTasks):
    bt.add_task(master_morning_routine)
    return {"status": "V253 Full Cascade Started"}

@app.post("/test-daily-analytics")
def test_daily_analytics(bt: BackgroundTasks):
    bt.add_task(analytics_agent.generate_daily_report)
    return {"status": "V253 Daily Analytics Started."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)