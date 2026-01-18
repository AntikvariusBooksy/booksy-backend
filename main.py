# BOOKSY BRAIN - V101 (ENHANCED POLICY INTELLIGENCE + POLICY-ONLY FORCE UPDATE)
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
from html.parser import HTMLParser
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from typing import List, Optional, Dict, Any

load_dotenv()
XML_FEED_URL = os.getenv("XML_FEED_URL", "https://www.antikvarius.ro/wp-content/uploads/woo-feed/google/xml/booksyfullfeed.xml")
TEMP_FILE = "temp_feed.xml"

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

# --- HELPEREK ---
def normalize_text(text):
    if not text: return ""
    text = str(text).lower()
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def safe_str(val):
    if val is None: return ""
    return html.unescape(str(val).strip())

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text_parts = []
    def handle_data(self, d): self.text_parts.append(d)
    def handle_starttag(self, tag, attrs):
        if tag in ['br', 'p', 'div', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'tr']: self.text_parts.append('\n')
    def handle_endtag(self, tag):
        if tag in ['p', 'div', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'tr']: self.text_parts.append('\n')
    def get_data(self): return ''.join(self.text_parts)

def clean_html_smart(raw_html):
    if not raw_html: return ""
    try:
        s = safe_str(raw_html)
        stripper = MLStripper()
        stripper.feed(s)
        text = stripper.get_data()
        text = text.replace('\xa0', ' ')
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()
    except: return safe_str(raw_html)

def extract_structured_prices(raw_html, clean_text):
    """
    Kinyeri a konkrét árakat és szállítási díjakat a HTML-ből.
    """
    result = {"shipping_prices": [], "payment_info": [], "contact_info": []}
    
    shipping_patterns = [
        r'(?i)(livrare|transport|shipping|cost)\s*[:=]?\s*([0-9.,]+\s*(?:RON|lei|EUR)|\bGratuit\b|\bGratis\b)',
        r'(?i)([0-9.,]+\s*(?:RON|lei))\s*(?:pentru|pentru|for)?\s*(?:livrare|transport)',
        r'(?i)(standard|express|rapid)\s*[:–-]\s*([0-9.,]+\s*(?:RON|lei))',
    ]
    
    for pattern in shipping_patterns:
        matches = re.finditer(pattern, clean_text)
        for match in matches:
            result["shipping_prices"].append(match.group(0).strip())
    
    payment_patterns = [
        r'(?i)(card|cash|ramburs|transfer)\s*[:=]?\s*([0-9.,]+\s*(?:RON|lei|%)|disponibil)',
        r'(?i)(comision|taxa)\s*[:=]?\s*([0-9.,]+\s*(?:RON|lei|%))',
    ]
    
    for pattern in payment_patterns:
        matches = re.finditer(pattern, clean_text)
        for match in matches:
            result["payment_info"].append(match.group(0).strip())
    
    contact_patterns = [
        r'(?i)(telefon|phone|tel)\s*[:=]?\s*([\+0-9\s\-()]+)',
        r'(?i)(email|e-mail)\s*[:=]?\s*([\w\.\-]+@[\w\.\-]+)',
        r'(?i)(adresa|address)\s*[:=]?\s*([^\n]{10,100})',
    ]
    
    for pattern in contact_patterns:
        matches = re.finditer(pattern, clean_text)
        for match in matches:
            result["contact_info"].append(match.group(0).strip())
    
    result["shipping_prices"] = list(set(result["shipping_prices"]))[:5]
    result["payment_info"] = list(set(result["payment_info"]))[:5]
    result["contact_info"] = list(set(result["contact_info"]))[:3]
    
    return result

def extract_all_data(elem) -> Dict[str, Any]:
    data = {}
    for child in elem:
        tag = child.tag.split('}')[-1].lower()
        val = safe_str(child.text)
        if val: data[tag] = val
    return data

def generate_content_hash(data_string): return hashlib.md5(data_string.encode('utf-8')).hexdigest()

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

# --- ADATBÁZIS ---
class DBHandler:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./booksy_db")
        self.collection = self.client.get_or_create_collection(name="booksy_collection")

# --- UPDATER (V101 Enhanced) ---
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
        print("ℹ️ [POLICY] Információs oldalak intelligens beolvasása (V101 Enhanced)...")
        headers = {'User-Agent': 'BooksyBot/1.0'}
        for page in POLICY_PAGES:
            try:
                url = page['url']
                r = requests.get(url, headers=headers, timeout=30)
                if r.status_code == 200:
                    raw_html = r.text
                    clean_text = clean_html_smart(raw_html)
                    if len(clean_text) > 30000: clean_text = clean_text[:30000]
                    
                    structured_data = extract_structured_prices(raw_html, clean_text)
                    
                    enriched_text = clean_text
                    if structured_data['shipping_prices']:
                        enriched_text += f"\n\n=== KONKRÉT SZÁLLÍTÁSI DÍJAK ===\n" + "\n".join(structured_data['shipping_prices'])
                    if structured_data['payment_info']:
                        enriched_text += f"\n\n=== FIZETÉSI INFORMÁCIÓK ===\n" + "\n".join(structured_data['payment_info'])
                    if structured_data['contact_info']:
                        enriched_text += f"\n\n=== KAPCSOLAT ===\n" + "\n".join(structured_data['contact_info'])
                    
                    d_hash = generate_content_hash(enriched_text)
                    page_id = f"policy_{generate_content_hash(url)}"
                    
                    try:
                        existing = self.db.collection.get(ids=[page_id], include=['metadatas'])
                        if existing['ids'] and existing['metadatas'][0].get('content_hash') == d_hash:
                            print(f"   ⏩ [SKIP] Policy változatlan: {page['name']}")
                            continue
                    except: pass
                    
                    emb_text = f"Típus: Szabályzat ({page['lang']}). Cím: {page['name']}. KONKRÉT ÁRAK ÉS DÍJAK: {' | '.join(structured_data['shipping_prices'][:3])}. Teljes tartalom: {enriched_text[:7000]}"
                    
                    emb = self.client_ai.embeddings.create(input=emb_text, model="text-embedding-3-small").data[0].embedding
                    
                    meta = {
                        "title": page['name'], 
                        "url": url, 
                        "text": enriched_text,
                        "lang": page['lang'], 
                        "type": "policy", 
                        "content_hash": d_hash, 
                        "last_seen": current_ts,
                        "shipping_prices": json.dumps(structured_data['shipping_prices'], ensure_ascii=False),
                        "payment_info": json.dumps(structured_data['payment_info'], ensure_ascii=False),
                        "contact_info": json.dumps(structured_data['contact_info'], ensure_ascii=False),
                    }
                    
                    self.db.collection.upsert(ids=[page_id], embeddings=[emb], metadatas=[meta])
                    print(f"   ✅ [POLICY] Frissítve (Strukturált Árral): {page['name']} - {len(structured_data['shipping_prices'])} ár találva")
            except Exception as e: print(f"   ❌ Hiba: {e}")

    def update_books(self, current_ts):
        """Könyvek frissítése XML feed-ből"""
        if not self.download_feed(): return
        try:
            print("🚀 [MODE] Parsing Books (Strict SKU Separation)")
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
                            clean_desc = clean_html_smart(raw_desc) 
                            category = clean_html_smart(item_data.get('product_type') or item_data.get('category') or "")
                            pub = "Ismeretlen"
                            match_pub = re.search(r'(?:Kiadó|Kiadás|Publisher)\s*[:|]\s*(.*?)(?:\n|$)', clean_desc, re.IGNORECASE)
                            if match_pub: pub = match_pub.group(1).strip()
                            if "bookman" in normalize_text(category): pub = "Bookman Kiadó"
                            auth = "Ismeretlen"
                            match_auth = re.search(r'(?:Szerző|Írta|Author|Szerzők)\s*[:|]\s*(.*?)(?:\n|$)', clean_desc, re.IGNORECASE)
                            if match_auth: auth = match_auth.group(1).strip()
                            else: auth = item_data.get('author') or "Ismeretlen"
                            raw_price = item_data.get('sale_price') or item_data.get('price') or "0"
                            final_ron_price = clean_price_raw(raw_price)
                            cat_norm = normalize_text(category)
                            detected_lang = "hu"
                            if "carti in limba romana" in cat_norm: detected_lang = "ro"
                            elif "magyar nyelvu konyvek" in cat_norm: detected_lang = "hu"
                            book_obj = {"id": bid, "title": title, "url": item_data.get('link', ''), "image_url": item_data.get('image_link', ''), "price": final_ron_price, "publisher": pub, "author": auth, "category": category, "description": clean_desc, "stock": "instock", "lang": detected_lang, "type": "book", "last_seen": current_ts}
                            for k, v in item_data.items():
                                if k not in book_obj: book_obj[k] = clean_html_smart(str(v))[:500] 
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
                hash_input = f"V101|{bid}|{book_data['title']}|{book_data['price']}|{book_data.get('condition', '')}"
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

    def run_daily_update(self, books_enabled=True):
        """Teljes frissítés (policy + könyvek opcionálisan)"""
        print(f"🔄 [AUTO] Napi Frissítés (V101 Logic)")
        current_sync_ts = int(time.time())
        self.update_policies(current_sync_ts)
        
        if books_enabled:
            self.update_books(current_sync_ts)
        else:
            print("⏩ [SKIP] Könyv-indexelés kikapcsolva (Policy-Only Mode)")

# --- BRAIN V101 (ENHANCED POLICY AWARENESS) ---
class BooksyBrain:
    def __init__(self):
        self.db = DBHandler()
        self.updater = AutoUpdater(self.db)
        self.client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.user_session_cache = {}

    def negotiate_handshake(self, url, session_id, ui_lang):
        prompt = f"""
        Act as Booksy, the Smart Bookstore Agent.
        
        Context:
        - Current URL: "{url}"
        - REQUIRED LANGUAGE: "{ui_lang}" (You MUST use this language).
        
        Task:
        1. Create a short, proactive welcome message (max 6 words) in {ui_lang}.
           - If specific book URL, mention context.
           - If generic URL, be helpful.
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
        except Exception as e:
            print(f"Handshake Error: {e}")
            is_hu = ui_lang == "hu"
            return {
                "ui_lang": ui_lang,
                "bubble_text": "Miben segíthetek?" if is_hu else "Cu ce te pot ajuta?",
                "placeholder": "Keresel valamit?" if is_hu else "Cauți o carte?"
            }

    def _analyze_intent(self, msg, context):
        prompt = f"""
        You are the Brain of an Antiquarian Bookstore Agent.
        User Input: "{msg}"
        Context: "{context}"
        Task: Analyze intent & generate optimized search parameters.
        Output JSON:
        {{
            "intent": "search_book" OR "policy_question" OR "chitchat",
            "search_queries": ["primary search query", "synonym"],
            "filters": {{ "lang": "hu" OR "ro" OR "all", "max_price": number OR null }},
            "user_preferences": "Extract nuances (gift, cheap, etc.)"
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
        except: return {"intent": "search_book", "search_queries": [msg], "filters": {"lang": "all"}, "user_preferences": ""}

    def _curate_results(self, candidates, user_msg, user_prefs):
        if not candidates: return []
        short_list = []
        for c in candidates:
            meta = c['metadata']
            short_list.append({"id": c['id'], "title": meta.get('title'), "price": meta.get('price'), "author": meta.get('author'), "desc_preview": meta.get('text_preview', '')[:150]})

        prompt = f"""
        You are an Expert Antiquarian Curator.
        User Request: "{user_msg}"
        Preferences: "{user_prefs}"
        Candidate Books: {json.dumps(short_list, ensure_ascii=False)}
        Task: Select the TOP 5-8 books that BEST match the request.
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

    def execute_search(self, queries, filters):
        all_candidates = {}
        for query in queries:
            try:
                vec = self.client_ai.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
                where_clause = {"$and": [{"stock": "instock"}, {"type": "book"}]}
                if filters.get('lang') and filters.get('lang') != 'all': where_clause["$and"].append({"lang": filters['lang']})
                results = self.db.collection.query(query_embeddings=[vec], n_results=15, where=where_clause)
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

            final_reply, final_products = "", []

            if intent == "policy_question":
                pol_res = self.db.collection.query(
                    query_embeddings=[self.client_ai.embeddings.create(input=msg, model="text-embedding-3-small").data[0].embedding],
                    n_results=4,
                    where={"type": "policy"}
                )
                
                ctx_text = ""
                all_shipping_prices = []
                all_payment_info = []
                all_contact_info = []
                
                if pol_res['ids']:
                    for i in range(len(pol_res['ids'][0])): 
                        meta = pol_res['metadatas'][0][i]
                        ctx_text += f"=== {meta.get('title')} ===\n{meta.get('text', '')[:3000]}\n\n"
                        
                        if meta.get('shipping_prices'):
                            try:
                                prices = json.loads(meta['shipping_prices'])
                                all_shipping_prices.extend(prices)
                            except: pass
                        
                        if meta.get('payment_info'):
                            try:
                                payment = json.loads(meta['payment_info'])
                                all_payment_info.extend(payment)
                            except: pass
                        
                        if meta.get('contact_info'):
                            try:
                                contact = json.loads(meta['contact_info'])
                                all_contact_info.extend(contact)
                            except: pass
                
                if all_shipping_prices:
                    ctx_text += f"\n\n🔴 KRITIKUS - KONKRÉT SZÁLLÍTÁSI DÍJAK:\n" + "\n".join(set(all_shipping_prices))
                if all_payment_info:
                    ctx_text += f"\n\n💳 FIZETÉSI MÓDOK:\n" + "\n".join(set(all_payment_info)[:5])
                if all_contact_info:
                    ctx_text += f"\n\n📞 KAPCSOLAT:\n" + "\n".join(set(all_contact_info)[:3])
                
                writer_system_prompt = f"""
You are Booksy, the intelligent bookstore assistant.

CRITICAL INSTRUCTIONS FOR POLICY QUESTIONS:
1. You MUST answer ONLY based on the context provided below.
2. If the user asks about PRICES, SHIPPING COSTS, or FEES, you MUST cite the EXACT amounts from the context.
3. NEVER say vague things like "reasonable prices" or "shipping available" - always give SPECIFIC numbers.
4. If a price is marked as "🔴 KRITIKUS", it is MANDATORY to include it in your response.
5. Format prices clearly: "15 RON" not "15" or "ár: 15".
6. If information is NOT in the context, say "Nu am informații despre acest lucru în documentele noastre."
7. Be conversational but PRECISE.

Context (Policy Documents):
{ctx_text}

User Question: {msg}
"""
                
                final_reply = self.client_ai.chat.completions.create(
                    model="gpt-4o-mini", 
                    messages=[{"role": "system", "content": writer_system_prompt}],
                    temperature=0.1
                ).choices[0].message.content

            elif intent == "search_book":
                candidates = self.execute_search(queries, filters)
                selected_books = self._curate_results(candidates, msg, prefs)
                ctx_text = ""
                for book in selected_books:
                    meta = book['metadata']
                    p_price = clean_price_raw(meta.get('price'))
                    ctx_text += f"--- BOOK ---\nTitle: {meta.get('title')}, Price: {p_price}, Author: {meta.get('author')}, Cond: {meta.get('condition', 'N/A')}\n"
                    final_products.append({"title": meta.get('title'), "price": p_price, "url": meta.get('url'), "image": meta.get('image_url')})
                
                if not final_products: ctx_text = "No relevant books found."
                writer_system_prompt = f"""
                You are Booksy, the AI Antiquarian. User Request: "{msg}". Nuance: "{prefs}".
                Inventory: {ctx_text}
                Task: Recommend the books. Explain WHY. Be helpful and natural. Use user's language.
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
scheduler.add_job(lambda: bot.updater.run_daily_update(books_enabled=True), CronTrigger(hour=3, minute=0))

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.start()
    yield
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def home(): return {"status": "Booksy V101 (ENHANCED POLICY INTELLIGENCE)"}

@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url, req.session_id)

@app.post("/init-chat")
def init_chat(req: InitRequest):
    return bot.negotiate_handshake(req.url, req.session_id, req.ui_lang)

@app.post("/force-update")
def force(bt: BackgroundTasks):
    """Force update - CSAK POLICY (30 másodperc)"""
    bt.add_task(bot.updater.run_daily_update, books_enabled=False)
    return {"status": "Force Update Started (Policy-Only Mode)"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
