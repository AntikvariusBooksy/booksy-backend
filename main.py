# BOOKSY BRAIN - V101 (HEAVY DUTY PARSER + STRICT PUBLISHER FILTER + AGENTIC TRANSLATION)
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

# --- ÚJ LIBEK A TÖKÉLETES PARSINGHOZ ---
from bs4 import BeautifulSoup
import markdownify

load_dotenv()
XML_FEED_URL = os.getenv("XML_FEED_URL", "https://www.antikvarius.ro/wp-content/uploads/woo-feed/google/xml/booksyfullfeed.xml")
TEMP_FILE = "temp_feed.xml"

# --- URL TUDÁSBÁZIS ---
POLICY_PAGES = [
    {"url": "https://www.antikvarius.ro/termeni-si-conditii-de-utilizare/", "lang": "ro", "name": "Termeni și condiții"},
    {"url": "https://www.antikvarius.ro/informatii-despre-plata/", "lang": "ro", "name": "Informații despre plată"},
    {"url": "https://www.antikvarius.ro/informatii-despre-livrare/", "lang": "ro", "name": "Informații despre livrare"},
    {"url": "https://www.antikvarius.ro/contact/", "lang": "ro", "name": "Contact"},
]

class ChatRequest(BaseModel):
    message: str
    context_url: Optional[str] = "" 
    session_id: Optional[str] = ""

class InitRequest(BaseModel):
    url: str
    session_id: str
    ui_lang: str = "ro" 

# --- HELPEREK (V101 - HEAVY DUTY) ---
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

# --- V101: SMART HTML PARSER ---
def html_to_markdown_clean(raw_html):
    """
    Konvertálja a HTML-t tiszta, olvasható Markdown-ra.
    Kezeli a táblázatokat, listákat, sortöréseket.
    """
    if not raw_html: return ""
    try:
        # Markdownify a legjobb eszköz erre
        md = markdownify.markdownify(raw_html, heading_style="ATX", strip=['script', 'style'])
        # Felesleges üres sorok törlése
        md = re.sub(r'\n\s*\n', '\n\n', md).strip()
        return md
    except:
        return safe_str(raw_html)

def extract_metadata_from_html(raw_html):
    """
    Kinyeri a Kiadót és Szerzőt a HTML struktúrából (pl. táblázatokból).
    """
    meta = {"publisher": None, "author": None}
    if not raw_html: return meta
    
    try:
        soup = BeautifulSoup(raw_html, 'lxml')
        
        # 1. Kiadó Keresés (Táblázatban vagy Szövegben)
        # Keresünk olyan elemet, ami tartalmazza a "Kiadó" vagy "Publisher" szót
        pub_label = soup.find(string=re.compile(r'(?:Kiadó|Publisher|Editura)\s*:', re.IGNORECASE))
        if pub_label:
            # Ha táblázat cellában van (<td>Kiadó:</td><td>Akadémia</td>)
            parent = pub_label.find_parent('td')
            if parent:
                next_td = parent.find_next_sibling('td')
                if next_td:
                    meta['publisher'] = next_td.get_text(strip=True)
            else:
                # Ha sima szövegben van (<b>Kiadó:</b> Akadémia)
                text_content = pub_label.find_parent().get_text(strip=True) if pub_label.find_parent() else pub_label
                match = re.search(r'(?:Kiadó|Publisher|Editura)\s*:\s*(.*?)(?:$|\n|\.|<)', text_content, re.IGNORECASE)
                if match:
                    meta['publisher'] = match.group(1).strip()

        # 2. Szerző Keresés
        auth_label = soup.find(string=re.compile(r'(?:Szerző|Írta|Author|Autor)\s*:', re.IGNORECASE))
        if auth_label:
            parent = auth_label.find_parent('td')
            if parent:
                next_td = parent.find_next_sibling('td')
                if next_td:
                    meta['author'] = next_td.get_text(strip=True)
            else:
                text_content = auth_label.find_parent().get_text(strip=True) if auth_label.find_parent() else auth_label
                match = re.search(r'(?:Szerző|Írta|Author|Autor)\s*:\s*(.*?)(?:$|\n|\.|<)', text_content, re.IGNORECASE)
                if match:
                    meta['author'] = match.group(1).strip()
                    
    except Exception as e:
        print(f"Metadata Parse Error: {e}")
        
    return meta

def extract_all_data(elem) -> Dict[str, Any]:
    data = {}
    for child in elem:
        tag = child.tag.split('}')[-1].lower()
        val = safe_str(child.text)
        if val: data[tag] = val
    return data

# --- ADATBÁZIS ---
class DBHandler:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./booksy_db")
        self.collection = self.client.get_or_create_collection(name="booksy_collection")

# --- UPDATER (V101 Refactored) ---
class AutoUpdater:
    def __init__(self, db: DBHandler):
        self.api_key_openai = os.getenv("OPENAI_API_KEY")
        self.client_ai = OpenAI(api_key=self.api_key_openai)
        self.db = db

    def download_feed(self):
        headers = {'User-Agent': 'BooksyBot/1.0'}
        for attempt in range(3):
            try:
                print(f"⬇️ [DOWNLOAD] XML Feed Letöltés (Kísérlet {attempt+1}/3)...")
                with requests.get(XML_FEED_URL, headers=headers, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with open(TEMP_FILE, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
                file_size = os.path.getsize(TEMP_FILE)
                if file_size < 10000: raise Exception("Túl kicsi fájl.")
                print(f"✅ [DOWNLOAD] Siker! Méret: {file_size / 1024 / 1024:.2f} MB")
                return True
            except Exception as e:
                print(f"⚠️ Hiba: {e}")
                time.sleep(5)
        return False

    def update_policies(self, current_ts):
        print("ℹ️ [POLICY] Információs oldalak intelligens beolvasása...")
        headers = {'User-Agent': 'BooksyBot/1.0'}
        for page in POLICY_PAGES:
            try:
                url = page['url']
                r = requests.get(url, headers=headers, timeout=30)
                if r.status_code == 200:
                    raw_html = r.text
                    # V101: Markdownify használata a tiszta szövegért
                    clean_text = html_to_markdown_clean(raw_html)
                    
                    d_hash = generate_content_hash(clean_text)
                    page_id = f"policy_{generate_content_hash(url)}"
                    
                    try:
                        existing = self.db.collection.get(ids=[page_id], include=['metadatas'])
                        if existing['ids'] and existing['metadatas'][0].get('content_hash') == d_hash:
                            print(f"   ⏩ [SKIP] Policy változatlan: {page['name']}")
                            continue
                    except: pass
                    
                    # Részletesebb embedding, hogy a számokat is lássa
                    emb_text = f"Típus: Szabályzat (ro). Cím: {page['name']}. Tartalom: {clean_text[:8000]}"
                    emb = self.client_ai.embeddings.create(input=emb_text, model="text-embedding-3-small").data[0].embedding
                    meta = {"title": page['name'], "url": url, "text": clean_text, "lang": "ro", "type": "policy", "content_hash": d_hash, "last_seen": current_ts}
                    self.db.collection.upsert(ids=[page_id], embeddings=[emb], metadatas=[meta])
                    print(f"   ✅ [POLICY] Frissítve (Markdown): {page['name']}")
            except Exception as e: print(f"   ❌ Hiba: {e}")

    def run_daily_update(self):
        print(f"🔄 [AUTO] Napi Frissítés (V101 Logic)")
        current_sync_ts = int(time.time())
        self.update_policies(current_sync_ts)
        if not self.download_feed(): return
        try:
            print("🚀 [MODE] Parsing Books (V101 - BeautifulSoup Power)")
            context = ET.iterparse(TEMP_FILE, events=("end",))
            unique_books_buffer = {} 
            count_total_xml_items = 0
            
            for event, elem in context:
                tag_local = elem.tag.split('}')[-1].lower()
                if tag_local in ['item', 'post']:
                    count_total_xml_items += 1
                    try:
                        item_data = extract_all_data(elem)
                        bid = item_data.get('id') or item_data.get('post_id') or item_data.get('g:id')
                        if bid:
                            title = item_data.get('title') or "Nincs cím"
                            raw_desc = f"{item_data.get('description', '')} {item_data.get('shortdescription', '')}"
                            
                            # V101: Metaadat bányászat BeautifulSoup-pal
                            extracted_meta = extract_metadata_from_html(raw_desc)
                            
                            # V101: Tiszta Markdown leírás
                            clean_desc = html_to_markdown_clean(raw_desc)
                            
                            category = html_to_markdown_clean(item_data.get('product_type') or item_data.get('category') or "")
                            
                            # Kiadó prioritás: Extrahált -> XML -> Ismeretlen
                            pub = extracted_meta['publisher'] 
                            if not pub:
                                pub = "Ismeretlen" # Ha nincs sehol
                            
                            # Szerző prioritás
                            auth = extracted_meta['author']
                            if not auth:
                                auth = item_data.get('author') or "Ismeretlen"

                            raw_price = item_data.get('sale_price') or item_data.get('price') or "0"
                            final_ron_price = clean_price_raw(raw_price)
                            
                            cat_norm = normalize_text(category)
                            detected_lang = "hu"
                            if "carti in limba romana" in cat_norm: detected_lang = "ro"
                            elif "magyar nyelvu konyvek" in cat_norm: detected_lang = "hu"
                            
                            book_obj = {"id": bid, "title": title, "url": item_data.get('link', ''), "image_url": item_data.get('image_link', ''), "price": final_ron_price, "publisher": pub, "author": auth, "category": category, "description": clean_desc, "stock": "instock", "lang": detected_lang, "type": "book", "last_seen": current_sync_ts}
                            unique_books_buffer[bid] = book_obj
                    except Exception as e: pass
                    elem.clear()
                    if count_total_xml_items % 5000 == 0: gc.collect()
            
            print(f"✅ [PARSE] Kész! Egyedi SKU-k száma: {len(unique_books_buffer)}")
            print("🚀 [SMART UPLOAD] Hash ellenőrzés...")
            
            ids_batch, embeddings_batch, metadatas_batch = [], [], []
            count_processed, count_skipped, count_uploaded = 0, 0, 0
            
            for bid, book_data in unique_books_buffer.items():
                count_processed += 1
                # Hashben benne van a kiadó is most már!
                hash_input = f"V101|{bid}|{book_data['title']}|{book_data['price']}|{book_data['publisher']}"
                d_hash = generate_content_hash(hash_input)
                book_data['content_hash'] = d_hash
                
                try:
                    existing = self.db.collection.get(ids=[bid], include=['metadatas'])
                    if existing and existing['ids'] and len(existing['ids']) > 0:
                        stored_hash = existing['metadatas'][0].get('content_hash', '')
                        if stored_hash == d_hash:
                            count_skipped += 1
                            if count_processed % 1000 == 0: print(f"⏩ [SKIP] {count_skipped} változatlan könyv...")
                            continue 
                except: pass
                
                emb_text = f"SKU: {bid}. Nyelv: {book_data['lang']}. Cím: {book_data['title']}. Szerző: {book_data['author']}. Ár: {book_data['price']}. Kategória: {book_data['category']}. Kiadó: {book_data['publisher']}. Leírás: {book_data['description'][:800]}"
                
                try:
                    emb = self.client_ai.embeddings.create(input=emb_text[:8000], model="text-embedding-3-small").data[0].embedding
                    clean_meta = book_data.copy()
                    del clean_meta['description'] 
                    clean_meta['text_preview'] = book_data['description'][:100]
                    ids_batch.append(bid)
                    embeddings_batch.append(emb)
                    metadatas_batch.append(clean_meta)
                    count_uploaded += 1
                    
                    if len(ids_batch) >= 50:
                        self.db.collection.upsert(ids=ids_batch, embeddings=embeddings_batch, metadatas=metadatas_batch)
                        ids_batch, embeddings_batch, metadatas_batch = [], [], []
                        print(f"💾 [UPDATE] {count_uploaded} könyv frissítve...")
                except Exception as e: print(f"⚠️ Hiba ({bid}): {e}")
            
            if ids_batch: self.db.collection.upsert(ids=ids_batch, embeddings=embeddings_batch, metadatas=metadatas_batch)
            if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)
            print(f"🏁 [VÉGE] {count_processed} feldolgozva. ⏩ {count_skipped} változatlan. 💾 {count_uploaded} frissítve.")
        except Exception as e: print(f"❌ Hiba: {e}")

# --- BRAIN V101 (AGENTIC) ---
class BooksyBrain:
    def __init__(self):
        self.db = DBHandler()
        self.updater = AutoUpdater(self.db)
        self.client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.user_session_cache = {}

    # --- V100 HANDSHAKE (VÁLTOZATLAN) ---
    def negotiate_handshake(self, url, session_id, ui_lang):
        prompt = f"""
        Act as Booksy, the Smart Bookstore Agent.
        Context:
        - Current URL: "{url}"
        - REQUIRED LANGUAGE: "{ui_lang}" (You MUST use this language).
        
        Task:
        1. Create a short, proactive welcome message (max 6 words) in {ui_lang}.
        2. Create a NATURAL, INVITING placeholder text for the input field in {ui_lang}.
           - Example HU: "Keresel valamit?", "Írd be a címet..."
           - Example RO: "Cauți o carte?", "Scrie titlul..."
           - STRICTLY FORBIDDEN: Do NOT use "Placeholder", "Text de rezerva", or generic filler words.

        Output JSON:
        {{
            "bubble_text": "Welcome text in {ui_lang}",
            "placeholder": "Natural question in {ui_lang}"
        }}
        """
        try:
            response = self.client_ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            data = json.loads(response.choices[0].message.content)
            data['ui_lang'] = ui_lang
            self.user_session_cache[session_id] = f"LANG_PREF:{ui_lang}"
            return data
        except:
            is_hu = ui_lang == "hu"
            return { "ui_lang": ui_lang, "bubble_text": "Miben segíthetek?" if is_hu else "Cu ce te pot ajuta?", "placeholder": "Keresel valamit?" if is_hu else "Cauți o carte?" }

    # --- PIPELINE V101: OKOSÍTOTT SZÁNDÉK FELISMERÉS ---
    def _analyze_intent(self, msg, context):
        prompt = f"""
        You are the Brain of an Antiquarian Bookstore Agent.
        User Input: "{msg}"
        Context: "{context}"
        
        Task: Analyze intent with precision.
        
        1. DETECT PUBLISHER SEARCH: If user asks for a specific publisher (e.g. "Akadémia kiadó könyvei"), extract it into 'specific_publisher'.
        2. DETECT LANGUAGE MISMATCH: If user asks about policy/shipping in Hungarian, but policies are in Romanian, set 'requires_translation' = True.
        
        Output JSON:
        {{
            "intent": "search_book" OR "policy_question" OR "chitchat",
            "search_queries": ["primary search query"],
            "specific_publisher": "Exact Publisher Name" OR null,
            "requires_translation": true/false,
            "filters": {{ "lang": "hu" OR "ro" OR "all", "max_price": number OR null }},
            "user_preferences": "Extract nuances"
        }}
        """
        try:
            response = self.client_ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(response.choices[0].message.content)
        except: return {"intent": "search_book", "search_queries": [msg], "filters": {"lang": "all"}, "specific_publisher": None}

    def _curate_results(self, candidates, user_msg, user_prefs):
        if not candidates: return []
        short_list = []
        for c in candidates:
            meta = c['metadata']
            short_list.append({"id": c['id'], "title": meta.get('title'), "publisher": meta.get('publisher'), "price": meta.get('price'), "author": meta.get('author')})

        prompt = f"""
        You are an Expert Antiquarian Curator.
        User Request: "{user_msg}"
        Candidate Books: {json.dumps(short_list, ensure_ascii=False)}
        Task: Select the TOP 5-8 books. 
        CRITICAL: If the user asked for a specific publisher, ONLY return books from that publisher. Ignore others.
        Output JSON: {{ "selected_ids": ["id1", "id2", ...] }}
        """
        try:
            response = self.client_ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            selected_ids = json.loads(response.choices[0].message.content).get("selected_ids", [])
            final_list = [c for c in candidates if c['id'] in selected_ids]
            return final_list if final_list else candidates[:5]
        except: return candidates[:8]

    def execute_search(self, queries, filters, specific_publisher=None):
        all_candidates = {}
        for query in queries:
            try:
                vec = self.client_ai.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
                
                # Alap szűrő
                where_clause = {"$and": [{"stock": "instock"}, {"type": "book"}]}
                if filters.get('lang') and filters.get('lang') != 'all': where_clause["$and"].append({"lang": filters['lang']})
                
                # V101: SZIGORÚ KIADÓ SZŰRÉS
                if specific_publisher:
                    # Itt egy 'contains' logikát próbálunk, de a Chroma 'where' korlátozott.
                    # Ezért inkább a legpontosabb egyezést erőltetjük, ha tudjuk.
                    # Ha a parserünk jó, akkor a 'publisher' mezőben benne van az "Akadémia Kiadó".
                    # A biztos találat érdekében itt most $contains helyett egy trükköt használunk:
                    # Csak akkor szűrünk DB szinten, ha pontosan tudjuk. Ha nem, a Curator szűr.
                    # De a V101-ben a parser jó, így bízhatunk a Curatorban is, de adjunk neki esélyt.
                    pass 

                results = self.db.collection.query(query_embeddings=[vec], n_results=20, where=where_clause) # Több találat, hogy legyen miből válogatni
                if results['ids']:
                    for i in range(len(results['ids'][0])):
                        bid = results['ids'][0][i]
                        if bid not in all_candidates:
                            all_candidates[bid] = {"id": bid, "score": results['distances'][0][i], "metadata": results['metadatas'][0][i]}
            except Exception as e: print(f"Search error: {e}")
        
        candidate_list = list(all_candidates.values())
        safe_max = filters.get('max_price')
        filtered_list = []
        for c in candidate_list:
            # Utólagos szigorú szűrés, ha van publisher igény
            if specific_publisher:
                book_pub = str(c['metadata'].get('publisher', '')).lower()
                req_pub = specific_publisher.lower().replace("kiadó", "").strip()
                if req_pub not in book_pub: 
                    continue # Eldobjuk, ha nem az a kiadó

            raw_price = c['metadata'].get('price', '0')
            price_val = parse_price_to_float(clean_price_raw(raw_price))
            if price_val is not None and safe_max and price_val > safe_max: continue
            filtered_list.append(c)
        return filtered_list

    def process(self, msg, context_url, session_id):
        try:
            last_context = self.user_session_cache.get(session_id, "")
            analysis = self._analyze_intent(msg, last_context)
            self.user_session_cache[session_id] = msg 
            
            intent = analysis.get('intent')
            queries = analysis.get('search_queries', [msg])
            filters = analysis.get('filters', {})
            prefs = analysis.get('user_preferences', "")
            
            # V101 Features
            specific_publisher = analysis.get('specific_publisher')
            requires_translation = analysis.get('requires_translation', False)

            final_reply, final_products = "", []

            if intent == "policy_question":
                # V101: AGENTIC TRANSLATION
                search_q = msg
                if requires_translation:
                    # Gyors fordítás, hogy megtaláljuk a román szabályzatban
                    trans_prompt = f"Translate this shipping/policy question to Romanian for search purposes: '{msg}'"
                    search_q = self.client_ai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": trans_prompt}]).choices[0].message.content
                
                pol_res = self.db.collection.query(
                    query_embeddings=[self.client_ai.embeddings.create(input=search_q, model="text-embedding-3-small").data[0].embedding],
                    n_results=2, where={"type": "policy"}
                )
                ctx_text = ""
                if pol_res['ids']:
                    for i in range(len(pol_res['ids'][0])): ctx_text += f"--- POLICY ---\n{pol_res['metadatas'][0][i].get('text')}\n"
                
                writer_messages = [{"role": "system", "content": "You are Booksy. Answer the policy question based on context. If context has numbers (shipping cost), share them exactly."}, {"role": "user", "content": f"Context:{ctx_text}\nQ: {msg}"}]
                final_reply = self.client_ai.chat.completions.create(model="gpt-4o-mini", messages=writer_messages).choices[0].message.content

            elif intent == "search_book":
                candidates = self.execute_search(queries, filters, specific_publisher)
                selected_books = self._curate_results(candidates, msg, prefs)
                ctx_text = ""
                for book in selected_books:
                    meta = book['metadata']
                    p_price = clean_price_raw(meta.get('price'))
                    ctx_text += f"--- BOOK ---\nTitle: {meta.get('title')}, Price: {p_price}, Author: {meta.get('author')}, Pub: {meta.get('publisher')}\n"
                    final_products.append({"title": meta.get('title'), "price": p_price, "url": meta.get('url'), "image": meta.get('image_url')})
                
                if not final_products: ctx_text = "No relevant books found."
                writer_system_prompt = f"""
                You are Booksy, the AI Antiquarian. User Request: "{msg}". Nuance: "{prefs}".
                Inventory: {ctx_text}
                Task: Recommend the books. Explain WHY. Be helpful and natural.
                """
                final_reply = self.client_ai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": writer_system_prompt}], temperature=0.2).choices[0].message.content
            else:
                final_reply = self.client_ai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "Be polite and brief."}, {"role": "user", "content": msg}]).choices[0].message.content
            return {"reply": final_reply, "products": final_products}
        except Exception as e:
            print(f"Error: {e}")
            return {"reply": "Hiba történt. Kérlek próbáld újra.", "products": []}

bot = BooksyBrain()
scheduler = BackgroundScheduler()
scheduler.add_job(bot.updater.run_daily_update, CronTrigger(hour=3, minute=0))

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.start()
    yield
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def home(): return {"status": "Booksy V101 (HEAVY DUTY PARSER ACTIVE)"}

@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url, req.session_id)

@app.post("/init-chat")
def init_chat(req: InitRequest):
    return bot.negotiate_handshake(req.url, req.session_id, req.ui_lang)

@app.post("/force-update")
def force(bt: BackgroundTasks):
    bt.add_task(bot.updater.run_daily_update)
    return {"status": "Force Update Started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)