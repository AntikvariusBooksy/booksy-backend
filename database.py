__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import time
import requests
import re
import unicodedata
import html
import sqlite3
import traceback
import chromadb
import pytz
import json
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import markdownify
from dotenv import load_dotenv

load_dotenv()
LOCAL_TZ = pytz.timezone('Europe/Bucharest')

XML_FEED_URL = "https://www.antikvarius.ro/wp-content/uploads/woo-feed/google/xml/booksyfullfeed.xml"
TEMP_FILE = "temp_feed.xml"
CHAT_DB_FILE = "./booksy_db/chat_logs.db"
STORE_POLICIES_FILE = "./booksy_db/store_policies.json"
ADMIN_EMAILS = ["bookmankiado@gmail.com", "joomla900@gmail.com"]

def log_event(msg):
    now = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 🤖 {msg}")

def normalize_text(text):
    if not text: return ""
    return ''.join(c for c in unicodedata.normalize('NFD', str(text).lower()) if unicodedata.category(c) != 'Mn')

def clean_price_raw(raw_price):
    if not raw_price: return "0 RON"
    cleaned = re.sub(r"[^\d.,]", "", str(raw_price).strip())
    return f"{cleaned} RON" if cleaned else str(raw_price)

def html_to_markdown_clean(raw_html):
    """HTML táblázatokat és listákat AI által olvasható markdownba alakít."""
    if not raw_html: return ""
    try: 
        # A markdownify segít megtartani a táblázat szerkezetét
        return markdownify.markdownify(raw_html, heading_style="ATX", strip=['script', 'style']).strip()
    except: 
        return str(raw_html)

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

def update_store_policies():
    """Letölti a céges szabályzatokat, kinyeri a lényeget és JSON-ba menti a gyors eléréshez."""
    log_event("🔄 Céges Szabályzatok (Policies) háttérfrissítése indul...")
    urls = [
        "https://www.antikvarius.ro/contact/",
        "https://www.antikvarius.ro/termeni-si-conditii-de-utilizare/",
        "https://www.antikvarius.ro/informatii-despre-plata/",
        "https://www.antikvarius.ro/informatii-despre-livrare/"
    ]
    policies_data = ""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                # A WooCommerce/Breakdance oldalak fő tartalmi része
                content = soup.select_one('.entry-content') or soup.find('main') or soup.find('body')
                if content:
                    # Letisztítjuk a felesleges HTML-t, de a táblázatokat megtartjuk!
                    md_text = html_to_markdown_clean(str(content))
                    # Csak a lényeget tartjuk meg, hogy ne szálljon el a prompt mérete
                    policies_data += f"--- FORRÁS: {url} ---\n{md_text[:2500]}\n\n"
        except Exception as e:
            log_event(f"⚠️ Hiba a {url} beolvasásakor: {e}")
            
    if policies_data:
        try:
            with open(STORE_POLICIES_FILE, "w", encoding="utf-8") as f:
                json.dump({"last_updated": str(datetime.now(LOCAL_TZ)), "content": policies_data}, f, ensure_ascii=False, indent=4)
            log_event("✅ Céges Szabályzatok sikeresen elmentve!")
        except Exception as e:
            log_event(f"⚠️ Hiba a szabályzat mentésekor: {e}")

def get_store_policies() -> str:
    """Gyorsan beolvassa a lementett szabályzatot a memóriából."""
    try:
        if os.path.exists(STORE_POLICIES_FILE):
            with open(STORE_POLICIES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("content", "")
    except Exception as e:
        log_event(f"⚠️ Hiba a szabályzat betöltésekor: {e}")
    return "A szabályzatok jelenleg nem elérhetők."

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
                          latency_ms INTEGER, device_type TEXT, trigger_type TEXT DEFAULT 'manual')''')
            c.execute('''CREATE TABLE IF NOT EXISTS analytics_reports
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, report_type TEXT, target_date TEXT,
                          content TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            
            try:
                c.execute("ALTER TABLE chat_logs ADD COLUMN trigger_type TEXT DEFAULT 'manual'")
            except sqlite3.OperationalError:
                pass 
                
            conn.commit(); conn.close()
        except Exception as e: log_event(f"❌ AnalyticsDB Init Hiba: {e}")

    def log_chat(self, data: dict):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''INSERT INTO chat_logs 
                         (session_id, user_msg, bot_reply, context_url, geo_country, geo_region, 
                          ui_language, chat_language, target_catalog, offered_book_ids, zero_match_flag, latency_ms, device_type, trigger_type) 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (data.get('session_id'), data.get('user_msg'), data.get('bot_reply'), data.get('context_url'),
                       data.get('geo_country'), data.get('geo_region'), data.get('ui_language'), data.get('chat_language'),
                       data.get('target_catalog'), data.get('offered_book_ids'), data.get('zero_match_flag'),
                       data.get('latency_ms'), data.get('device_type'), data.get('trigger_type', 'manual')))
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