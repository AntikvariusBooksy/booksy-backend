# BOOKSY BRAIN - V255 (THE PROACTIVE AGENT EDITION)
# VERZIÓ: V255 - MULTI-MODEL RAG + PROACTIVE HOOKS + ANALYTICS (FACEBOOK/DALL-E PURGED)

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os, time, requests, re, json, unicodedata, html, sqlite3, traceback
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import chromadb
import pytz
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import markdownify
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

from google import genai
from google.genai import types
import anthropic 
from openai import OpenAI
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from typing import List, Optional, Dict, Any

# --- CONFIG & CLIENTS ---
load_dotenv()
LOCAL_TZ = pytz.timezone('Europe/Bucharest')

# AI Kliensek Inicializálása (A 3 lépcsős agy)
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# JAVÍTVA: A működő modell verzió
CLAUDE_MODEL = "claude-sonnet-5" # Szóvivő/Értékesítő
OPENAI_MODEL = "gpt-4o-mini" # Karmester/Szándékfelismerő

XML_FEED_URL = "https://www.antikvarius.ro/wp-content/uploads/woo-feed/google/xml/booksyfullfeed.xml"
TEMP_FILE = "temp_feed.xml"
CHAT_DB_FILE = "./booksy_db/chat_logs.db"
STORE_POLICIES_FILE = "./booksy_db/store_policies.json"
ADMIN_EMAILS = ["bookmankiado@gmail.com", "joomla900@gmail.com"]

# --- UTILS & GDPR FILTERS ---
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
    if not raw_html: return ""
    try: return markdownify.markdownify(raw_html, heading_style="ATX", strip=['script', 'style']).strip()
    except: return str(raw_html)

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
                          latency_ms INTEGER, device_type TEXT, trigger_type TEXT DEFAULT 'manual')''')
            c.execute('''CREATE TABLE IF NOT EXISTS analytics_reports
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, report_type TEXT, target_date TEXT,
                          content TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            
            # JAVÍTVA: Automatikus SQL migráció a hiányzó oszlophoz
            try:
                c.execute("ALTER TABLE chat_logs ADD COLUMN trigger_type TEXT DEFAULT 'manual'")
            except sqlite3.OperationalError:
                pass # Az oszlop már létezik
                
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

# --- AI ANALYTICS AGENT ---
class AIAnalyticsAgent:
    def __init__(self):
        self.report_emails = ADMIN_EMAILS

    def _get_market_trends(self, context="napi"):
        prompt = (f"Keress rá a weben a legfrissebb e-kereskedelmi és könyvpiaci trendekre. SZIGORÚ prioritási "
                  f"sorrend a {context} adatokhoz: 1. Romániai piac, 2. Magyarországi piac, 3. Európai trendek. "
                  f"Mik a legújabb keresett műfajok?")
        for attempt in range(3):
            try:
                res = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=[prompt])
                return res.text
            except Exception as e:
                if attempt < 2: time.sleep(3)
                else: return "Piaci trendek lekérése sikertelen."

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
                msg.attach(MIMEText(body, 'html', 'utf-8'))
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
        
        analytics_rule = (f"KÖTELEZŐ SZABÁLY: A nyelvezet legyen üzleti, vezetői, laikusok számára is érthető, emberi! "
                          f"Zéró kód-zsargon vagy technikai kifejezés! Fogalmazz így: 'Egy látogató', 'Nincs találat a raktárban'. "
                          f"KÖTELEZŐ HTML FORMÁZÁS: A teljes riportot SZIGORÚAN tiszta HTML kódban írd meg! NE használj Markdown-t! "
                          f"Csak tisztán a HTML szöveget add vissza (ne tedd backticks közé)!")
        
        if not logs or len(logs) == 0:
            system_prompt = "Válságmenedzser és Üzleti Elemző vagy. Ma nulla interakció volt a chaten."
            user_msg = f"Piaci adatok: {market_trends}\n\nKészíts HTML Napi Riportot arról, mi okozhatta a zéró forgalmat!\n{analytics_rule}"
        else:
            system_prompt = "Profi Marketing Elemző és CRO Stratéga vagy."
            user_msg = (f"Napi interakciók ({len(logs)} db):\n{logs}\n\nPiaci trendek:\n{market_trends}\n\n"
                        f"Készíts átfogó Napi Riportot. Fókuszok:\n"
                        f"1. Földrajzi Eloszlás (RO vs HU IP-k).\n"
                        f"2. Készlet & Beszerzés (keresett, de nem talált könyvek - zero match).\n"
                        f"3. Proaktív Triggerek sikeressége (melyik hook hozott konverziót).\n"
                        f"4. 🔮 Webdevmk AI Előrejelzés a következő napokra!\n{analytics_rule}")

        for attempt in range(3):
            try:
                res = claude_client.messages.create(model=CLAUDE_MODEL, max_tokens=4096, system=system_prompt, messages=[{"role": "user", "content": user_msg}])
                report = res.content[0].text.strip()
                report = report.replace("```html", "").replace("```", "").strip()
                
                analytics_db.save_report("daily", target_date_str, report)
                self._send_analytics_email(f"📊 Napi Booksy AI Üzleti Jelentés ({target_date_str})", report)
                analytics_db.cleanup_old_logs() 
                log_event("✅ Napi Analitika befejezve.")
                break
            except Exception as e:
                if attempt < 2: time.sleep(3)
                else: log_event(f"❌ Napi Analitika végleges hiba: {e}")

    def generate_monthly_report(self):
        now = datetime.now(LOCAL_TZ)
        last_month_dt = now.replace(day=1) - timedelta(days=1)
        target_month_str = last_month_dt.strftime('%Y-%m')
        
        log_event(f"📈 Havi AI Analitika Indítása ({target_month_str})...")
        daily_reports = analytics_db.get_reports_for_period("daily", target_month_str)
        if not daily_reports: return
        
        market_trends = self._get_market_trends("havi")
        compiled_reports = "\n\n---NAPI JELENTÉS---\n\n".join(daily_reports)
        
        analytics_rule = (f"KÖTELEZŐ HTML FORMÁZÁS: A teljes riportot SZIGORÚAN tiszta HTML kódban írd meg! NE használj Markdown-t! "
                          f"Csak tisztán a HTML szöveget add vissza!")
        
        prompt = (f"A mellékelt szöveg az elmúlt hónap összes napi jelentése. Piaci havi trendek: {market_trends}\n\n"
                  f"Készíts vezetői HAVI JELENTÉST. Fókusz: Forgalmi források, erdélyi piac, proaktív konverziók "
                  f"és UX frontend javaslatok. Végezetül: '🔮 Webdevmk AI Előrejelzés a következő hónapra'. Csak HTML listák.\n{analytics_rule}\n\nJelentések: {compiled_reports}")
        
        for attempt in range(3):
            try:
                res = claude_client.messages.create(model=CLAUDE_MODEL, max_tokens=6000, system="Üzleti Stratéga vagy.", messages=[{"role": "user", "content": prompt}])
                report = res.content[0].text.strip().replace("```html", "").replace("```", "").strip()
                analytics_db.save_report("monthly", target_month_str, report)
                self._send_analytics_email(f"📈 HAVI Booksy AI Menedzsment Riport ({target_month_str})", report)
                log_event("✅ Havi Analitika befejezve.")
                break
            except Exception as e:
                if attempt < 2: time.sleep(3)
                else: log_event(f"❌ Havi Analitika végleges hiba: {e}")

    def generate_yearly_report(self):
        target_year_str = str(datetime.now(LOCAL_TZ).year - 1)
        log_event(f"👑 ÉVES AI Stratégiai Analitika Indítása ({target_year_str})...")
        
        monthly_reports = analytics_db.get_reports_for_period("monthly", target_year_str)
        if not monthly_reports: return
        
        market_trends = self._get_market_trends("éves jövőkutatási")
        compiled_reports = "\n\n---HAVI JELENTÉS---\n\n".join(monthly_reports)
        
        analytics_rule = (f"KÖTELEZŐ HTML FORMÁZÁS: A teljes riportot SZIGORÚAN tiszta HTML kódban írd meg! NE használj Markdown-t! "
                          f"Csak tisztán a HTML szöveget add vissza!")
        
        prompt = (f"A mellékelt szöveg az elmúlt év 12 havi jelentése. Globális Éves Trendek: {market_trends}\n\n"
                  f"Készíts ÉVES Menedzsment Riportot! Értékeld a ROI-t, a proaktív beavatkozások sikerességét (RO vs HU), "
                  f"frontend UX tanulságokat, majd egy '🔮 Webdevmk AI Éves Előrejelzés és Beszerzés' szekciót. "
                  f"Csak HTML listás, diagrammentes struktúra.\n{analytics_rule}\n\nJelentések: {compiled_reports}")
        
        for attempt in range(3):
            try:
                res = claude_client.messages.create(model=CLAUDE_MODEL, max_tokens=8000, system="Vezérigazgatói Tanácsadó vagy.", messages=[{"role": "user", "content": prompt}])
                report = res.content[0].text.strip().replace("```html", "").replace("```", "").strip()
                analytics_db.save_report("yearly", target_year_str, report)
                self._send_analytics_email(f"👑 ÉVES Booksy AI Stratégiai Iránytű ({target_year_str})", report)
                log_event("✅ Éves Analitika befejezve.")
                break
            except Exception as e:
                if attempt < 2: time.sleep(3)
                else: log_event(f"❌ Éves Analitika végleges hiba: {e}")


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
                r = requests.get(f"{url}?v={int(time.time())}", headers={"Cache-Control": "no-cache"}, timeout=20)
                if r.status_code == 200:
                    soup = BeautifulSoup(r.content, 'html.parser')
                    for script in soup(["script", "style", "nav", "footer", "header", "aside"]): script.extract()
                    text = soup.get_text(separator=' ', strip=True)
                    policies_text += f"\n\n--- FORRÁS: {url} ---\n{text[:6000]}"
            except Exception as e: log_event(f"⚠️ Hiba a {url} beolvasásakor: {e}")
        
        if policies_text:
            with open(STORE_POLICIES_FILE, "w", encoding="utf-8") as f:
                json.dump({"policies": policies_text}, f, ensure_ascii=False)
            log_event("✅ Céges Kódex sikeresen frissítve az élő weboldalról.")

    def run_daily_update(self):
        log_event("🚀 [SYNC] Indítás (XML -> DB)...")
        try:
            r = requests.get(XML_FEED_URL, stream=True, timeout=300)
            if r.status_code != 200: return False
            with open(TEMP_FILE, 'wb') as f:
                for chunk in r.iter_content(8192): f.write(chunk)
        except: return False
            
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
                # Keresési optimalizálás: az embeddings szöveg tartalmazza a kulcsszavakat
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
            log_event(f"❌ SZINKRON HIBA: {e}")
            return False

# --- THE PROACTIVE AGENT (Multi-Model Core) ---
class BooksyProactiveAgent:
    def __init__(self, db: DBHandler):
        self.db = db

    def _get_policies(self):
        if os.path.exists(STORE_POLICIES_FILE):
            with open(STORE_POLICIES_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("policies", "")
        return "Céges szabályzat nem elérhető."

    def _intent_routing(self, msg: str) -> dict:
        """Lépés 1: Gyors OpenAI szándékfelismerés és query kiterjesztés"""
        system_prompt = (
            "Te egy e-kereskedelmi router vagy. Elemezd a bejövő üzenetet. Válaszolj KIZÁRÓLAG JSON formátumban!\n"
            "Lehetséges 'intent' értékek: 'policy' (szállítás, fizetés, kapcsolat, árak, cégadatok), 'search' (konkrét könyv vagy téma keresése), 'general' (egyéb csevegés).\n"
            "Ha 'search', akkor generálj egy 'expanded_query' mezőt, ami 3-5 szemantikus kulcsszóval bővíti a keresést (pl. 'izgalmas könyv' -> 'thriller, krimi, fordulatok, feszültség').\n"
            "Minta JSON: {\"intent\": \"search\", \"expanded_query\": \"eredeti szó, szinonima1, szinonima2\"}"
        )
        try:
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": msg}
                ],
                response_format={ "type": "json_object" },
                max_tokens=150
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            log_event(f"⚠️ Intent Routing Hiba: {e}")
            return {"intent": "search", "expanded_query": msg} # Fallback

    def _vector_search(self, query: str, limit: int = 4) -> list:
        """Lépés 2: Gemini Embeddings alapú villámgyors szemantikus keresés a ChromaDB-ben"""
        try:
            vec_req = gemini_client.models.embed_content(
                model="gemini-embedding-001", 
                contents=query, 
                config=types.EmbedContentConfig(output_dimensionality=768)
            )
            vec = vec_req.embeddings[0].values
            db_res = self.db.collection.query(
                query_embeddings=[vec], 
                n_results=limit, 
                where={"$and": [{"stock": "instock"}, {"type": "book"}]}
            )
            
            if db_res['ids'] and db_res['ids'][0]:
                products = db_res['metadatas'][0]
                seen_titles = set()
                final_products = []
                for p in products:
                    if 'image_url' in p and 'image' not in p: p['image'] = p['image_url']
                    clean_title = p.get('title', '').strip().lower()
                    if clean_title not in seen_titles:
                        seen_titles.add(clean_title)
                        final_products.append(p)
                return final_products
            return []
        except Exception as e:
            log_event(f"⚠️ Vektor Keresés Hiba: {e}")
            return []

    def _generate_claude_response(self, user_msg: str, intent_data: dict, products: list, is_proactive: bool = False, trigger_context: str = "") -> str:
        """Lépés 3: Claude 3.5 megírja az empatikus, konverzióvezérelt végső választ"""
        
        policy_text = self._get_policies()
        context_text = "Nem találtam megfelelő könyvet a raktárban."
        if products:
            context_text = "\n".join([f"Könyv: {p['title']} - {p.get('author','')} - Ár: {p.get('price','')}. Infó: {p.get('text_preview','')}" for p in products])

        system_prompt = (
            f"Te Booksy vagy, a Booksy (antikvarius.ro) prémium online antikváriumának profi értékesítője és asszisztense.\n"
            f"Céges tudásbázisod (ÁSZF, szállítás, kapcsolat):\n<company_policies>\n{policy_text}\n</company_policies>\n\n"
            f"SZIGORÚ SZABÁLYOK (SÉRTHETETLEN):\n"
            f"1. A szállítás díja zónánként FIX! Soha nincs ingyenes szállítás semmilyen rendelési összeg felett. Kommunikáld logikusan: mivel fix a díj, minél több könyvet vesznek, annál jobban eloszlik a költség, jobban megéri!\n"
            f"2. Utánvétes fizetés (Ramburs / Plata la livrare) KIZÁRÓLAG Románián belül lehetséges! Más országokba (pl. Magyarország, EU) CSAK online bankkártyás fizetés engedélyezett!\n"
            f"3. Ha céges infót kérdeznek, KIZÁRÓLAG a <company_policies> adatai alapján válaszolj. Zéró hallucináció.\n"
            f"4. A választ kötelezően AZON A NYELVEN (Magyar vagy Román) fogalmazd meg, ahogy a felhasználó kommunikál.\n"
            f"5. A válaszod legyen empatikus, emberi, elegáns, ne gépies. ZÉRÓ MARKDOWN formázás (nincs csillagozás)!\n"
        )

        user_content = f"Felhasználó üzenete: '{user_msg}'\n\nRaktári találatok a kérésére:\n{context_text}"

        if is_proactive:
            system_prompt += (
                f"\nFIGYELEM: Ez egy PROAKTÍV beavatkozás (a felhasználó nem írt a chatbe, a rendszer dobja fel). "
                f"A kiváltó esemény: {trigger_context}. "
                f"Szólítsd meg kedvesen, és ajánld fel a segítséget vagy a talált könyveket a fenti esemény kontextusában."
            )
            user_content = "Írd meg a proaktív üzenetet a megadott kontextus alapján."

        try:
            res = claude_client.messages.create(
                model=CLAUDE_MODEL, 
                max_tokens=1000, 
                system=system_prompt, 
                messages=[{"role": "user", "content": user_content}]
            )
            return res.content[0].text.strip()
        except Exception as e:
            log_event(f"⚠️ Claude Válaszgenerálási Hiba: {e}")
            return "Sajnos technikai hiba történt. Kérlek, próbáld újra később!"

    def process_chat(self, msg: str) -> dict:
        """Sztenderd chat folyamat feldolgozása (Multi-Model)"""
        # 1. Intent felismerés (OpenAI)
        intent_data = self._intent_routing(msg)
        
        # 2. Vektoros keresés (Gemini -> ChromaDB)
        final_products = []
        if intent_data['intent'] == 'search':
            final_products = self._vector_search(intent_data.get('expanded_query', msg), limit=4)
        
        # 3. Válasz generálás (Claude)
        reply_text = self._generate_claude_response(msg, intent_data, final_products, is_proactive=False)
        
        return {
            "reply": reply_text, 
            "products": final_products, 
            "zero_match_flag": (intent_data['intent'] == 'search' and len(final_products) == 0)
        }

    def process_proactive_trigger(self, trigger_type: str, session_data: dict) -> dict:
        """A weboldalról érkező viselkedési jelek (telemetria) feldolgozása"""
        
        trigger_context = ""
        search_query = ""
        
        if trigger_type == "cart_abandonment":
            trigger_context = "A látogató betett egy könyvet a kosárba, de megállt, vagy el akarja hagyni az oldalt. Emlékeztesd a fix szállítási díj előnyére (több könyv = jobban megéri), és ajánlj hasonló könyveket."
            search_query = session_data.get("last_book_title", "klasszikus irodalom")
        
        elif trigger_type == "zero_match_search":
            search_query = session_data.get("failed_search_term", "")
            trigger_context = f"A weboldal belső keresője nem adott találatot erre: '{search_query}'. Lásd el a látogatót hasonló, releváns könyvekkel a raktárból, hogy ne hagyja el az oldalt."
        
        elif trigger_type == "checkout_hesitation":
            trigger_context = "A látogató sok időt tölt a pénztár oldalon anélkül, hogy vásárolna. Kérdezd meg, elakadt-e, és emlékeztesd, hogy RO-n belül utánvét is van, míg külföldre biztonságos bankkártyás fizetés biztosított."
            search_query = "" # Itt nem kell könyvet keresni
            
        else:
            return {"reply": "", "products": []}

        # Szinonima keresés a raktárban (csak ha releváns a trigger)
        final_products = []
        if search_query:
            intent_data = self._intent_routing(search_query) # Kiterjesztjük a sikertelen/kosárban lévő keresést
            final_products = self._vector_search(intent_data.get('expanded_query', search_query), limit=3)
            
        # Claude megírja a proaktív popup szövegét
        reply_text = self._generate_claude_response(
            user_msg="", 
            intent_data={"intent": "proactive"}, 
            products=final_products, 
            is_proactive=True, 
            trigger_context=trigger_context
        )
        
        return {
            "reply": reply_text,
            "products": final_products,
            "trigger_handled": True
        }


# --- MASTER CASCADE & SCHEDULING ---
updater = AutoUpdater(db_handler)
agent = BooksyProactiveAgent(db_handler)
analytics_agent = AIAnalyticsAgent()
scheduler = BackgroundScheduler()

def master_morning_routine():
    log_event("🌅 Master Láncreakció Indítása (V255)")
    updater.fetch_store_policies()
    try:
        sync_success = updater.run_daily_update()
        if not sync_success: log_event("⚠️ Szinkronizációs hiba, korábbi adatok használata.")
    except Exception as e: log_event(f"⚠️ Váratlan hiba a szinkronnál: {e}")

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

class ProactiveRequest(BaseModel):
    trigger_type: str # 'cart_abandonment', 'zero_match_search', 'checkout_hesitation'
    session_id: str
    context_url: Optional[str] = ""
    failed_search_term: Optional[str] = ""
    last_book_title: Optional[str] = ""
    ui_lang: Optional[str] = "hu"
    device_type: Optional[str] = "Desktop"

class InitRequest(BaseModel): url: str; session_id: str; ui_lang: str = "hu"

@app.get("/")
def home(): return {"status": "V255 Online (Proactive Multi-Model Agent)", "project": "Booksy"}

@app.post("/chat")
def chat(req: ChatRequest, request: Request): 
    start_time = time.time()
    bot_response = agent.process_chat(req.message)
    latency = int((time.time() - start_time) * 1000)
    
    forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip = forwarded_for.split(",")[0].strip() if forwarded_for else (request.client.host if request.client else None)
    geo_country, geo_region = get_geo_from_ip(client_ip)
    
    log_data = {
        "session_id": req.session_id, "user_msg": clean_pii(req.message), "bot_reply": bot_response.get("reply", "")[:200],
        "context_url": req.context_url, "geo_country": geo_country, "geo_region": geo_region,
        "ui_language": req.ui_lang, "chat_language": req.chat_lang, "target_catalog": req.target_catalog,
        "offered_book_ids": ",".join([p.get("id", "") for p in bot_response.get("products", [])]) if bot_response.get("products") else "", 
        "zero_match_flag": bot_response.get("zero_match_flag", False),
        "latency_ms": latency, "device_type": req.device_type, "trigger_type": "manual"
    }
    analytics_db.log_chat(log_data)
    return bot_response

@app.post("/proactive-hook")
def proactive_hook(req: ProactiveRequest, request: Request):
    """ÚJ VÉGPONT: Telemetriai jelek fogadása és proaktív válasz generálása"""
    start_time = time.time()
    
    session_data = {
        "failed_search_term": req.failed_search_term,
        "last_book_title": req.last_book_title
    }
    
    bot_response = agent.process_proactive_trigger(req.trigger_type, session_data)
    latency = int((time.time() - start_time) * 1000)
    
    forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip = forwarded_for.split(",")[0].strip() if forwarded_for else (request.client.host if request.client else None)
    geo_country, geo_region = get_geo_from_ip(client_ip)
    
    log_data = {
        "session_id": req.session_id, "user_msg": f"[TRIGGER]: {req.trigger_type}", "bot_reply": bot_response.get("reply", "")[:200],
        "context_url": req.context_url, "geo_country": geo_country, "geo_region": geo_region,
        "ui_language": req.ui_lang, "chat_language": req.ui_lang, "target_catalog": "mixed",
        "offered_book_ids": ",".join([p.get("id", "") for p in bot_response.get("products", [])]) if bot_response.get("products") else "", 
        "zero_match_flag": False,
        "latency_ms": latency, "device_type": req.device_type, "trigger_type": req.trigger_type
    }
    analytics_db.log_chat(log_data)
    return bot_response

@app.post("/init-chat")
def init_chat(req: InitRequest): return {"ui_lang": req.ui_lang, "bubble_text": "Szia!", "placeholder": "Keresel valamit?"}

@app.post("/test-cascade")
def test_cascade(bt: BackgroundTasks):
    bt.add_task(master_morning_routine)
    return {"status": "V255 Full Cascade Started"}

@app.post("/test-daily-analytics")
def test_daily_analytics(bt: BackgroundTasks):
    bt.add_task(analytics_agent.generate_daily_report)
    return {"status": "V255 Daily Analytics Started."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)