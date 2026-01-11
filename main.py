import os
import time
import requests
import hashlib
import re
import unicodedata
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from bs4 import BeautifulSoup
from apscheduler.schedulers.background import BackgroundScheduler
from typing import List, Optional

# --- KONFIGURÁCIÓ ---
load_dotenv()
INDEX_NAME = "booksy-index"
XML_FEED_URL = os.getenv("XML_FEED_URL", "https://www.antikvarius.ro/wp-content/uploads/woo-feed/google/xml/booksyfullfeed.xml")

POLICY_URLS = {
    "KAPCSOLAT": "https://www.antikvarius.ro/contact/",
    "FIZETÉS": "https://www.antikvarius.ro/hu/fizetesi-informaciok/",
    "SZÁLLÍTÁS": "https://www.antikvarius.ro/hu/szallitasi-informaciok/",
    "ÁSZF": "https://www.antikvarius.ro/hu/altalanos-szerzodesi-es-felhasznalasi-feltetelek/"
}

# --- ADATMODELLEK ---
class ChatRequest(BaseModel):
    message: str
    context_url: Optional[str] = "" 

class VisitEvent(BaseModel):
    url: str
    title: str
    time_spent: int 

class SmartHookRequest(BaseModel):
    current_url: str
    current_title: str
    visitor_type: str 
    cart_item_count: int
    history: List[VisitEvent] = [] 
    lang: str

# --- SEGÉDFÜGGVÉNYEK ---
def normalize_text(text):
    if not text: return ""
    text = str(text).lower()
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def generate_content_hash(data_string):
    return hashlib.md5(data_string.encode('utf-8')).hexdigest()

def clean_html(raw_html):
    if not raw_html: return ""
    s = str(raw_html).replace('<br>', ' ').replace('<p>', ' ').replace('</p>', ' ')
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', s)
    cleantext = cleantext.replace("<![CDATA[", "").replace("]]>", "")
    return " ".join(cleantext.split())

def safe_str(val):
    return str(val).strip() if val is not None else ""

def extract_author(short_desc):
    if not short_desc: return ""
    match = re.search(r'(Szerző|Írta):\s*([^<|\n]+)', short_desc, re.IGNORECASE)
    return match.group(2).strip() if match else ""

# --- AUTOMATIZÁLT FRISSÍTŐ MOTOR (V33 LOGIC) ---
class AutoUpdater:
    def __init__(self):
        self.api_key_openai = os.getenv("OPENAI_API_KEY")
        self.api_key_pinecone = os.getenv("PINECONE_API_KEY")
        self.client_ai = OpenAI(api_key=self.api_key_openai)
        self.pc = Pinecone(api_key=self.api_key_pinecone)
        self.index = self.pc.Index(INDEX_NAME)

    def scrape_policy(self):
        print("🔄 [AUTO] Jogi és Kapcsolat információk ellenőrzése...")
        full_policy_text = "[TUDÁSBÁZIS AZ ÜGYFÉLSZOLGÁLATHOZ - FRISSÍTVE: MA]\n"
        for category, url in POLICY_URLS.items():
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.content, 'html.parser')
                    for script in soup(["script", "style", "nav", "footer", "header"]): script.extract()
                    text = ' '.join(soup.get_text(separator=' ').split())
                    full_policy_text += f"\n--- {category} INFORMÁCIÓK ---\n{text[:5000]}\n"
            except Exception as e:
                print(f"⚠️ Hiba a {category} letöltésekor: {e}")

        new_hash = generate_content_hash(full_policy_text)
        try:
            existing = self.index.fetch(ids=["store_policy"])
            if existing and 'vectors' in existing and 'store_policy' in existing['vectors']:
                stored_meta = existing['vectors']['store_policy'].get('metadata', {})
                if stored_meta.get('content_hash', '') == new_hash:
                    print("✅ [AUTO] Jogi infók változatlanok.")
                    return
        except: pass

        try:
            print("💾 [AUTO] Policy frissítés...")
            res = self.client_ai.embeddings.create(input="policy definition", model="text-embedding-3-small")
            self.index.upsert(vectors=[("store_policy", res.data[0].embedding, {"type": "policy", "content": full_policy_text, "content_hash": new_hash})])
            print("✅ [AUTO] Jogi infók frissítve.")
        except Exception as e:
            print(f"❌ [AUTO] Hiba: {e}")

    def update_books_from_feed(self):
        print(f"🔄 [AUTO] Könyv szinkronizáció: {XML_FEED_URL}")
        current_sync_ts = int(time.time())
        try:
            response = requests.get(XML_FEED_URL, stream=True, timeout=120)
            if response.status_code != 200: return
            try: tree = ET.fromstring(response.content)
            except: tree = ET.fromstring(response.content.decode('utf-8', 'ignore'))

            items = tree.findall('.//item')
            if not items: items = tree.findall('.//post')
            print(f"📚 [AUTO] Elemek: {len(items)}")
            
            batch = []
            ns = {'g': 'http://base.google.com/ns/1.0'}
            
            for item in items:
                try:
                    avail_node = item.find('g:availability', ns) or item.find('StockStatus')
                    avail = safe_str(avail_node.text).lower() if avail_node else "in stock"
                    if "out" in avail: continue

                    id_node = item.find('g:id', ns) or item.find('ID')
                    if not id_node or not id_node.text: continue
                    bid = safe_str(id_node.text)

                    sku_node = item.find('g:mpn', ns) or item.find('SKU')
                    sku = safe_str(sku_node.text) if sku_node else ""

                    title_node = item.find('g:title', ns) or item.find('Title')
                    title = safe_str(title_node.text) if title_node else "Nincs cím"

                    desc_node = item.find('g:description', ns) or item.find('Content')
                    desc = clean_html(safe_str(desc_node.text)) if desc_node else ""
                    
                    short_desc_node = item.find('ShortDescription')
                    short_desc = clean_html(safe_str(short_desc_node.text)) if short_desc_node else desc[:500]

                    auth = extract_author(short_desc)

                    cat_node = item.find('g:product_type', ns) or item.find('Productcategories')
                    cat = safe_str(cat_node.text) if cat_node else ""
                    
                    url = safe_str((item.find('g:link', ns) or item.find('Link') or item.find('Permalink')).text)
                    img = safe_str((item.find('g:image_link', ns) or item.find('ImageURL')).text)

                    price_node = item.find('g:price', ns) or item.find('Price')
                    sale_node = item.find('g:sale_price', ns) or item.find('SalePrice')
                    reg = safe_str(price_node.text) if price_node else "0"
                    sale = safe_str(sale_node.text) if sale_node else ""
                    
                    full_txt = f"{title} {auth} {sku} {cat} {desc}"[:9500]
                    d_hash = generate_content_hash(f"{bid}{title}{reg}{sale}{desc[:200]}")

                    need_emb = True
                    try:
                        ex = self.index.fetch(ids=[bid])
                        if ex and 'vectors' in ex and bid in ex['vectors']:
                            if ex['vectors'][bid]['metadata'].get('content_hash') == d_hash:
                                emb = ex['vectors'][bid]['values']
                                need_emb = False
                    except: pass

                    if need_emb:
                        emb = self.client_ai.embeddings.create(input=f"{title}|{auth}|{cat}|{short_desc}"[:8000], model="text-embedding-3-small").data[0].embedding

                    meta = {
                        "title": title, "price": reg, "sale_price": sale, "url": url, "image_url": img, 
                        "lang": "hu", "stock": "instock", "author": auth, "category": cat, 
                        "short_desc": short_desc[:500], "full_search_text": full_txt, 
                        "content_hash": d_hash, "last_seen": current_sync_ts
                    }
                    batch.append((bid, emb, meta))
                    if len(batch) >= 50: self.index.upsert(vectors=batch); batch = []
                except: continue

            if batch: self.index.upsert(vectors=batch)
            print("🧹 [AUTO] Takarítás (Mirror Sync)...")
            try: self.index.delete(filter={"last_seen": {"$lt": current_sync_ts}, "type": {"$ne": "policy"}})
            except: pass

        except Exception as e: print(f"Sync Error: {e}")

    def run_daily_update(self):
        self.scrape_policy()
        self.update_books_from_feed()

# --- BRAIN (KERESŐ & SALES AGENT) ---
class BooksyBrain:
    def __init__(self):
        self.updater = AutoUpdater()
        self.client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.index = Pinecone(api_key=os.getenv("PINECONE_API_KEY")).Index(INDEX_NAME)

    def get_policy(self):
        try: return self.index.fetch(ids=["store_policy"])['vectors']['store_policy']['metadata']['content']
        except: return "Nincs infó."

    def generate_smart_hook(self, req: SmartHookRequest):
        history_summary = "Látogatott oldalak:\n" + "\n".join([f"- {h.title} ({h.time_spent} mp)" for h in req.history[-3:]]) if req.history else "Még csak most érkezett."

        system_prompt = f"""
        You are Booksy, an intelligent antique book sales agent.
        GOAL: Engage the visitor with a short, friendly, and context-aware message.
        CONTEXT:
        - Current Page: {req.current_title} ({req.current_url})
        - Visitor: {req.visitor_type}
        - History: {history_summary}
        CRITICAL RULES:
        1. Shipping is FLAT RATE (FIXED). NEVER free. Encourage bulk orders.
        2. Language: The user is on a page with language code: {req.lang}. Start conversation in THIS language.
        Task: Generate a short hook (max 2 sentences).
        """
        try:
            return self.client_ai.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system", "content":system_prompt}], temperature=0.7).choices[0].message.content.strip()
        except: return "Szia! Segíthetek?"

    def search(self, q, search_lang_filter):
        try:
            # V34: Ha 'all' a filter (minden nyelven), nem szűrünk nyelvre
            norm_q = normalize_text(q)
            stop = ['a','az','egy','es']
            kw = [w for w in norm_q.split() if w not in stop and len(w)>2]
            clean_q = " ".join(kw) if kw else q
            
            vec = self.client_ai.embeddings.create(input=clean_q, model="text-embedding-3-small").data[0].embedding
            
            filt = {"stock": "instock"}
            
            # CSAK AKKOR SZŰRÜNK, HA NEM 'all'
            if search_lang_filter != 'all' and search_lang_filter in ['hu','ro']: 
                filt["lang"] = search_lang_filter
            
            res = self.index.query(vector=vec, top_k=100, include_metadata=True, filter=filt)
            if not res.get('matches'): return []
            
            final = []
            seen = set()
            for m in res['matches']:
                tit = m['metadata'].get('title','')
                if tit in seen: continue
                seen.add(tit)
                score = 0
                if not kw: score = m['score']*100
                else:
                    tn = normalize_text(tit)
                    an = normalize_text(m['metadata'].get('author',''))
                    fn = normalize_text(m['metadata'].get('full_search_text',''))
                    cnt = 0
                    for k in kw:
                        hit=False
                        if k in tn: score+=100; hit=True
                        elif k in an: score+=80; hit=True
                        elif k in fn: score+=20; hit=True
                        if hit: cnt+=1
                    if cnt==len(kw) and len(kw)>1: score+=200
                if kw and score<10: continue
                m['final_relevance'] = score
                final.append(m)
            
            final.sort(key=lambda x: x['final_relevance'], reverse=True)
            return final[:20]
        except: return []

    def process(self, msg, context_url=""):
        try:
            # --- JAVÍTVA: Az üzeneteket EGY listában adjuk át, nem duplán ---
            res = self.client_ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system", "content":"Detect Language (hu/ro) and Intent (SEARCH/INFO). Output: LANG | INTENT"},
                    {"role":"user", "content":msg}
                ],
                temperature=0.1
            )
            # -----------------------------------------------------------------
            
            p = res.choices[0].message.content.split('|')
            user_lang, intent = p[0].strip().lower(), p[1].strip().upper()
        except: user_lang, intent = 'hu', 'SEARCH'

        site_lang = 'hu' if '/hu/' in str(context_url) else 'ro'

        if intent == 'INFO':
            pol = self.get_policy()
            instr = f"Reply in {user_lang.upper()}."
            sys = f"Shipping is FLAT RATE (Fixed per zone), never free. Encourage bulk orders. Context: User is on {site_lang} site."
            ans = self.client_ai.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys},{"role":"system","content":f"Policy:\n{pol}"},{"role":"system","content":instr},{"role":"user","content":msg}]).choices[0].message.content
            return {"reply": ans, "products": []}

        # V34: "MINDEN NYELVEN" DETEKTÁLÁS
        force_all = False
        if "minden nyelven" in msg.lower() or "toate limbile" in msg.lower() or "all languages" in msg.lower():
            force_all = True
            
        search_filter = 'all' if force_all else site_lang

        matches = self.search(msg, search_filter)
        
        # HA NINCS TALÁLAT
        if not matches:
            txt = "Sajnos nem találtam." if user_lang=='hu' else "Nu am găsit."
            
            # Ha NEM "minden nyelven" kerestünk, ajánljuk fel!
            if not force_all:
                if user_lang == 'hu':
                    txt += " (Tipp: Ha a teljes raktárkészletben - pl. román fordításokban - is keresnél, írd mögé: 'minden nyelven'!)"
                else:
                    txt += " (Sfat: Pentru a căuta în tot stocul - inclusiv maghiar - scrie: 'toate limbile'!)"
            
            # Cross-Language figyelmeztetés (Ha nem "minden nyelven" volt és eltér a site)
            if not force_all and user_lang != site_lang:
                if user_lang == 'hu': txt += "\n(Egyébként a román részlegen vagyunk. Átváltsak a magyarra?)"
                else: txt += "\n(Suntem pe secțiunea maghiară. Să trec pe cea română?)"
                
            return {"reply": txt, "products": []}
        
        # HA VAN TALÁLAT
        prods = []
        ctx = ""
        for m in matches:
            meta = m['metadata']
            price = meta.get('sale_price') or meta.get('price')
            p = {"title":meta.get('title'), "price":price, "url":meta.get('url'), "image":meta.get('image_url')}
            prods.append(p)
            ctx += f"- {p['title']} ({price})\n"
            if len(prods)>=8: break
            
        sys = "You are a helpful antique book assistant. Recommend these books."
        
        # OKOS LÁBJEGYZET ÖSSZEÁLLÍTÁSA (V34)
        footer_note = ""
        
        # Csak akkor okoskodunk, ha NEM "minden nyelven" kerestünk
        if not force_all:
            # 1. Cross-Language figyelmeztetés (ez a fontosabb)
            if user_lang != site_lang:
                if user_lang == 'hu': footer_note = "\n\n(Megjegyzés: Ezek a könyvek a román részlegről vannak, ahol jelenleg tartózkodsz. Ha magyar könyveket keresel, kattints a magyar zászlóra!)"
                else: footer_note = "\n\n(Notă: Aceste cărți sunt din secțiunea maghiară. Dacă cauți cărți în română, schimbă limba site-ului!)"
            
            # 2. Ha jó helyen vagyunk, de lehet, hogy máshol is van találat (Smart Tip)
            else:
                if user_lang == 'hu': footer_note = "\n\n💡 Tipp: Csak a magyar részlegen kerestem. Ha a román fordítások is érdekelnek, írd a kereséshez: 'minden nyelven'!"
                else: footer_note = "\n\n💡 Sfat: Am căutat doar în secțiunea română. Dacă vrei să vezi și traducerile maghiare, scrie: 'toate limbile'!"

        instr = f"Reply in {user_lang.upper()}."
        prompt_content = f"User Query: {msg}\nFound Books:\n{ctx}\n\nExplain shortly why these are good matches. Add this note at the end: '{footer_note}'"

        ans = self.client_ai.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys},{"role":"system","content":instr},{"role":"user","content":prompt_content}]).choices[0].message.content
        return {"reply": ans, "products": prods}

bot = BooksyBrain()
scheduler = BackgroundScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(bot.updater.run_daily_update, 'cron', hour=3, minute=0)
    scheduler.start()
    yield
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def home(): return {"status": "Booksy V34 (Omnisearch & Tips)"}

@app.post("/smart-hook")
def smart_hook_endpoint(request: SmartHookRequest): return {"hook": bot.generate_smart_hook(request)}

@app.post("/chat")
def chat(req: ChatRequest): return bot.process(req.message, req.context_url)

@app.post("/force-update")
def force(bt: BackgroundTasks):
    bt.add_task(bot.updater.run_daily_update)
    return {"status": "Started"}