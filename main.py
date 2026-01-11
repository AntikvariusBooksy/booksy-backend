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

# --- KONFIGURÁCIÓ ---
load_dotenv()
INDEX_NAME = "booksy-index"
# A Renderen beállított feed URL
XML_FEED_URL = os.getenv("XML_FEED_URL", "https://www.antikvarius.ro/wp-content/uploads/woo-feed/google/xml/booksyfullfeed.xml")

POLICY_URLS = {
    "FIZETÉS": "https://www.antikvarius.ro/hu/fizetesi-informaciok/",
    "SZÁLLÍTÁS": "https://www.antikvarius.ro/hu/szallitasi-informaciok/",
    "ÁSZF": "https://www.antikvarius.ro/hu/altalanos-szerzodesi-es-felhasznalasi-feltetelek/"
}

# --- ADATMODELLEK ---
class ChatRequest(BaseModel):
    message: str

class HookRequest(BaseModel):
    url: str
    page_title: str
    visitor_type: str 
    cart_status: str 
    lang: str

# --- SEGÉDFÜGGVÉNYEK ---
def normalize_text(text):
    if not text: return ""
    text = str(text).lower()
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def generate_content_hash(data_string):
    """MD5 hash a változások figyeléséhez"""
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

# --- AUTOMATIZÁLT FRISSÍTŐ MOTOR (V29 SMART SYNC) ---
class AutoUpdater:
    def __init__(self):
        self.api_key_openai = os.getenv("OPENAI_API_KEY")
        self.api_key_pinecone = os.getenv("PINECONE_API_KEY")
        self.client_ai = OpenAI(api_key=self.api_key_openai)
        self.pc = Pinecone(api_key=self.api_key_pinecone)
        self.index = self.pc.Index(INDEX_NAME)

    def scrape_policy(self):
        """Jogi szövegek frissítése"""
        print("🔄 [AUTO] Jogi információk frissítése...")
        full_policy_text = "[TUDÁSBÁZIS AZ ÜGYFÉLSZOLGÁLATHOZ - FRISSÍTVE: MA]\n"
        
        for category, url in POLICY_URLS.items():
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.content, 'html.parser')
                    for script in soup(["script", "style", "nav", "footer", "header"]): script.extract()
                    text = ' '.join(soup.get_text(separator=' ').split())
                    full_policy_text += f"\n--- {category} INFORMÁCIÓK ---\n{text[:4000]}\n"
            except Exception as e:
                print(f"⚠️ Hiba a {category} letöltésekor: {e}")

        try:
            res = self.client_ai.embeddings.create(input="policy", model="text-embedding-3-small")
            self.index.upsert(vectors=[("store_policy", res.data[0].embedding, {"type": "policy", "content": full_policy_text})])
            print("✅ [AUTO] Jogi infók mentve.")
        except Exception as e:
            print(f"❌ [AUTO] Hiba a policy mentéskor: {e}")

    def update_books_from_feed(self):
        """Google Shopping Feed feldolgozása Smart Delta logikával"""
        print(f"🔄 [AUTO] Könyv szinkronizáció innen: {XML_FEED_URL}")
        
        current_sync_ts = int(time.time())
        
        try:
            response = requests.get(XML_FEED_URL, stream=True, timeout=120)
            if response.status_code != 200:
                print(f"❌ [AUTO] Feed hiba: {response.status_code}")
                return

            # XML parse
            try:
                tree = ET.fromstring(response.content)
            except ET.ParseError:
                # Ha a szerver nem szabványos XML-t küld, próbáljuk javítani
                tree = ET.fromstring(response.content.decode('utf-8', 'ignore'))

            # Google Feedben általában <channel><item> van
            items = tree.findall('.//item')
            if not items: items = tree.findall('.//post') # Fallback
            
            print(f"📚 [AUTO] Feed elemszám: {len(items)}")
            
            batch_vectors = []
            uploaded_count = 0
            skipped_hash_count = 0
            
            # Google Namespace
            ns = {'g': 'http://base.google.com/ns/1.0'}

            for item in items:
                try:
                    # 1. KÉSZLET ELLENŐRZÉS (Google Format: g:availability)
                    # Értékek: "in stock", "out of stock", "preorder"
                    avail_node = item.find('g:availability', ns)
                    if avail_node is None: avail_node = item.find('StockStatus') # Fallback
                    
                    availability = safe_str(avail_node.text).lower() if avail_node is not None else "in stock"
                    
                    # Ha "out of stock" vagy "outofstock", akkor ugrunk
                    if "out" in availability:
                        continue

                    # 2. ADATOK
                    id_node = item.find('g:id', ns)
                    if id_node is None: id_node = item.find('ID')
                    if id_node is None or not id_node.text: continue
                    book_id = safe_str(id_node.text)

                    # Cím
                    title_node = item.find('g:title', ns)
                    if title_node is None: title_node = item.find('Title')
                    title = safe_str(title_node.text) if title_node is not None else "Nincs cím"

                    # Leírás (Google feedben gyakran g:description a hosszú)
                    desc_node = item.find('g:description', ns)
                    if desc_node is None: desc_node = item.find('Content')
                    raw_desc = safe_str(desc_node.text) if desc_node is not None else ""
                    clean_desc = clean_html(raw_desc) # Ez a hosszú leírás

                    # Szerző (Próbáljuk kinyerni a leírásból, ha nincs külön mező)
                    author = extract_author(clean_desc[:500]) # Első 500 karakterben szokott lenni

                    # Kategória (g:product_type)
                    cat_node = item.find('g:product_type', ns)
                    if cat_node is None: cat_node = item.find('Productcategories')
                    cat_raw = safe_str(cat_node.text) if cat_node is not None else ""

                    # Link & Kép
                    link_node = item.find('g:link', ns) or item.find('link') or item.find('Permalink')
                    url = safe_str(link_node.text) if link_node is not None else ""

                    img_node = item.find('g:image_link', ns) or item.find('ImageURL')
                    img = safe_str(img_node.text) if img_node is not None else ""

                    # Ár (g:price / g:sale_price - Pl: "2500 HUF")
                    price_node = item.find('g:price', ns) or item.find('Price')
                    sale_node = item.find('g:sale_price', ns) or item.find('SalePrice')
                    
                    regular_price = safe_str(price_node.text) if price_node is not None else "0"
                    sale_price = safe_str(sale_node.text) if sale_node is not None else ""
                    
                    # Google feedben gyakran benne van a pénznem (HUF/RON), azt le kell vágni, ha számolni akarunk,
                    # de megjelenítéshez jó így is. A "final_price" logikához kellhet tisztítás.
                    final_price = sale_price if sale_price else regular_price

                    # 3. FULL TEXT & HASH
                    # Mivel a Google feedben nincs külön "ShortDescription", a clean_desc az elsődleges
                    full_search_text = f"{title} {author} {cat_raw} {clean_desc}"
                    full_search_text = full_search_text[:9500]

                    # Hash generálás (változás figyelés)
                    data_to_hash = f"{book_id}{title}{final_price}{clean_desc[:200]}"
                    content_hash = generate_content_hash(data_to_hash)

                    # SMART DELTA ELLENŐRZÉS
                    # Ha a Pinecone-ban lévő hash egyezik, nem töltjük fel újra.
                    # De a "last_seen" dátumot frissíteni KÉNE, hogy ne törlődjön.
                    # Ezt a Pinecone 'update' metódusával oldjuk meg metadata only? 
                    # Egyszerűbb, ha az "upsert" megtörténik, de a vektorgenerálást (OpenAI) spóroljuk meg.
                    
                    need_embedding = True
                    try:
                        existing = self.index.fetch(ids=[book_id])
                        if existing and 'vectors' in existing and book_id in existing['vectors']:
                            stored_meta = existing['vectors'][book_id].get('metadata', {})
                            if stored_meta.get('content_hash') == content_hash:
                                # Változatlan tartalom -> Megspóroljuk az OpenAI hívást!
                                # Csak a 'last_seen' frissítése miatt visszatöltjük a régi vektort
                                embedding_vector = existing['vectors'][book_id]['values']
                                need_embedding = False
                                skipped_hash_count += 1
                    except: pass

                    if need_embedding:
                        # Ha új vagy változott, kérünk új vektort
                        ai_input = f"CÍM: {title} | SZERZŐ: {author} | KAT: {cat_raw} | TARTALOM: {clean_desc[:1000]}"
                        res = self.client_ai.embeddings.create(input=ai_input[:8000], model="text-embedding-3-small")
                        embedding_vector = res.data[0].embedding

                    metadata = {
                        "title": title,
                        "price": regular_price,
                        "sale_price": sale_price,
                        "url": url,
                        "image_url": img,
                        "lang": "hu", # A feed alapján lehetne detektálni
                        "stock": "instock",
                        "author": author,
                        "category": cat_raw,
                        "short_desc": clean_desc[:500],
                        "full_search_text": full_search_text,
                        "content_hash": content_hash,
                        "last_seen": current_sync_ts # <--- TÜKÖRSZINKRON BÉLYEGZŐ!
                    }
                    
                    batch_vectors.append((book_id, embedding_vector, metadata))
                    uploaded_count += 1

                    if len(batch_vectors) >= 50:
                        self.index.upsert(vectors=batch_vectors)
                        batch_vectors = []

                except Exception as e:
                    continue

            if batch_vectors:
                self.index.upsert(vectors=batch_vectors)
            
            print(f"✅ [AUTO] Feltöltés kész. Frissítve: {uploaded_count}, Változatlan (OpenAI spórolt): {skipped_hash_count}")

            # 4. TAKARÍTÁS (MIRROR SYNC)
            # Törlünk minden olyan könyvet, amit NEM láttunk ebben a körben (tehát last_seen < mostani indítás)
            # Kivéve a 'store_policy'-t!
            print("🧹 [AUTO] Tükörszinkron takarítás...")
            try:
                # Pinecone delete by filter (Metadata szűrés)
                self.index.delete(
                    filter={
                        "last_seen": {"$lt": current_sync_ts},
                        "type": {"$ne": "policy"} # A policy-t ne törölje!
                    }
                )
                print("🧹 [AUTO] Készlethiányos/Törölt könyvek eltávolítva.")
            except Exception as e:
                print(f"⚠️ [AUTO] Törlési hiba (Lehet, hogy a Serverless indexen még nem aktív a filter delete): {e}")

        except Exception as e:
            print(f"❌ [AUTO] Kritikus hiba a frissítés közben: {e}")

    def run_daily_update(self):
        print("⏰ [SCHEDULER] Napi szinkronizáció indul...")
        self.scrape_policy()
        self.update_books_from_feed()

# --- BRAIN (KERESŐ) ---
class BooksyBrain:
    def __init__(self):
        self.updater = AutoUpdater()
        self.client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.index = Pinecone(api_key=os.getenv("PINECONE_API_KEY")).Index(INDEX_NAME)

    def get_dynamic_policy(self):
        try:
            fetch_response = self.index.fetch(ids=["store_policy"])
            if fetch_response and 'store_policy' in fetch_response['vectors']:
                return fetch_response['vectors']['store_policy']['metadata']['content']
            return "Az információk jelenleg nem elérhetőek."
        except: return "Hiba."

    def generate_sales_hook(self, ctx):
        try:
            res = self.client_ai.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role":"system", "content":"Short sales hook."}, {"role":"user", "content":"Hook."}], temperature=0.7)
            return res.choices[0].message.content.strip()
        except: return "Szia!"

    def search_engine_logic(self, query_text, lang_filter):
        try:
            stop_words = ['a', 'az', 'egy', 'es', 'konyv', 'konyvek', 'keresek', 'kiado', 'szerzo', 'cim', 'regeny']
            norm_q = normalize_text(query_text)
            keywords = [w for w in norm_q.split() if w not in stop_words and len(w) > 2]
            clean_q = " ".join(keywords) if keywords else query_text

            res = self.client_ai.embeddings.create(input=clean_q, model="text-embedding-3-small")
            
            filt = {"stock": "instock"}
            if lang_filter in ['hu', 'ro']: filt["lang"] = lang_filter
            
            # Keresés
            raw = self.index.query(vector=res.data[0].embedding, top_k=200, include_metadata=True, filter=filt)
            if not raw.get('matches'): return []

            scored = []
            seen = set()
            for m in raw['matches']:
                meta = m['metadata']
                title = str(meta.get('title', ''))
                if title in seen: continue
                seen.add(title)

                title_n = normalize_text(title)
                auth_n = normalize_text(str(meta.get('author', '')))
                # Itt keressük a full textben is!
                full_n = normalize_text(str(meta.get('full_search_text', '')))
                
                score = 0
                if not keywords: score = m['score'] * 100
                else:
                    matches_cnt = 0
                    for kw in keywords:
                        hit = False
                        if kw in title_n: score += 100; hit = True
                        elif kw in auth_n: score += 80; hit = True
                        elif kw in full_n: score += 20; hit = True # Ha csak a leírásban van, kevesebb pont
                        if hit: matches_cnt += 1
                    if matches_cnt == len(keywords) and len(keywords) > 1: score += 200
                
                if keywords and score < 10: continue
                m['final_relevance'] = score
                scored.append(m)

            scored.sort(key=lambda x: x['final_relevance'], reverse=True)
            return scored[:20]
        except: return []

    def process_message(self, user_input):
        # 1. Intent & Lang
        try:
            res = self.client_ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system", "content":"Detect Language (hu/ro) and Intent (SEARCH/INFO). Output: LANG | INTENT"}, {"role":"user", "content":user_input}], temperature=0.1
            )
            p = res.choices[0].message.content.split('|')
            lang, intent = p[0].strip().lower(), p[1].strip().upper()
        except: lang, intent = 'hu', 'SEARCH'

        # 2. Logic
        if intent == 'INFO':
            policy = self.get_dynamic_policy()
            prompt = f"Te Booksy vagy. Válaszolj EZEK alapján:\n{policy}\nHa nincs válasz, küldd a kapcsolati oldalt."
            instr = "Reply in ROMANIAN." if lang == 'ro' else "Reply in HUNGARIAN."
            ai_res = self.client_ai.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role":"system", "content":prompt}, {"role":"system", "content":instr}, {"role":"user", "content":user_input}], temperature=0.1
            )
            return {"reply": ai_res.choices[0].message.content, "products": []}
        
        # Search
        matches = self.search_engine_logic(user_input, lang)
        
        ctx = ""
        prods = []
        if not matches:
            msg = "Sajnos nem találtam készleten lévő könyvet." if lang == 'hu' else "Nu am găsit."
            tip = "\n\n💡 *Tipp: Ha mindent látni szeretnél, írd hozzá: „minden nyelven”!*"
            return {"reply": msg + tip, "products": []}

        for m in matches:
            meta = m['metadata']
            p = {
                "title": str(meta.get('title')), 
                "price": str(meta.get('final_price') or meta.get('price')), # Végső ár
                "url": str(meta.get('url')), 
                "image": str(meta.get('image_url'))
            }
            prods.append(p)
            ctx += f"- {p['title']} (Szerző: {meta.get('author')}, Ár: {p['price']}, Info: {str(meta.get('short_desc'))[:100]})\n"
            if len(prods) >= 8: break

        instr = "Reply in ROMANIAN." if lang == 'ro' else "Reply in HUNGARIAN."
        sys = "Ajánld a könyveket."
        ai_res = self.client_ai.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role":"system", "content":sys}, {"role":"system", "content":instr}, {"role":"user", "content":f"User: {user_input}\nBooks:\n{ctx}"}], temperature=0.3
        )
        return {"reply": ai_res.choices[0].message.content, "products": prods}

# --- APP SETUP ---
bot = BooksyBrain()
scheduler = BackgroundScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Hajnali 3:00-kor fut az automata frissítés
    scheduler.add_job(bot.updater.run_daily_update, 'cron', hour=3, minute=0)
    scheduler.start()
    print("⏰ Automata Google Feed tükörszinkronizáció beidőzítve (03:00).")
    yield
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def home(): return {"status": "Booksy V29 (Auto Google Sync)"}

@app.post("/hook")
def hook_endpoint(request: HookRequest): return {"hook": bot.generate_sales_hook(request)}

@app.post("/chat")
def chat_endpoint(request: ChatRequest): return bot.process_message(request.message)

@app.post("/force-update")
def force_update(background_tasks: BackgroundTasks):
    background_tasks.add_task(bot.updater.run_daily_update)
    return {"message": "Tükörszinkronizáció elindítva..."}