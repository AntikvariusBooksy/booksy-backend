import os
import time
import requests
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

# A Renderen beállított feed URL-t használjuk
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

def clean_html(raw_html):
    if not raw_html: return ""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    cleantext = cleantext.replace("<![CDATA[", "").replace("]]>", "")
    return " ".join(cleantext.split())

def safe_str(val):
    if val is None: return ""
    return str(val).strip()

def extract_author(short_desc):
    if not short_desc: return ""
    match = re.search(r'(Szerző|Írta):\s*([^<|\n]+)', short_desc, re.IGNORECASE)
    if match: return match.group(2).strip()
    return ""

# --- AUTOMATIZÁLT FRISSÍTŐ MOTOR (V26 MIRROR SYNC) ---
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
                    text = soup.get_text(separator=' ')
                    clean_text = ' '.join(text.split())
                    full_policy_text += f"\n--- {category} INFORMÁCIÓK ---\n{clean_text[:4000]}\n"
            except Exception as e:
                print(f"⚠️ Hiba a {category} letöltésekor: {e}")

        try:
            res = self.client_ai.embeddings.create(input="policy definition", model="text-embedding-3-small")
            self.index.upsert(vectors=[("store_policy", res.data[0].embedding, {"type": "policy", "content": full_policy_text})])
            print("✅ [AUTO] Jogi infók mentve.")
        except Exception as e:
            print(f"❌ [AUTO] Hiba a policy mentéskor: {e}")

    def update_books_from_feed(self):
        """Könyvek tükörszinkronja (Csak az marad, ami a feedben van)"""
        print(f"🔄 [AUTO] Könyv szinkronizáció innen: {XML_FEED_URL}")
        
        current_sync_ts = int(time.time())
        
        try:
            response = requests.get(XML_FEED_URL, stream=True, timeout=120)
            if response.status_code != 200:
                print(f"❌ [AUTO] Feed hiba: {response.status_code}")
                return

            tree = ET.fromstring(response.content)
            items = tree.findall('.//post')
            if not items: items = tree.findall('.//item')
            
            print(f"📚 [AUTO] Feed elemszám: {len(items)}")
            
            batch_vectors = []
            count = 0
            
            for item in items:
                try:
                    ns = {'g': 'http://base.google.com/ns/1.0'}
                    id_tag = item.find('ID')
                    if id_tag is None: id_tag = item.find('g:id', ns)
                    
                    # --- ITT VOLT A HIBA ---
                    # Helyesen: Ha nincs ID vagy üres a szövege, akkor ugrunk
                    if id_tag is None or not id_tag.text: continue 
                    
                    book_id = safe_str(id_tag.text)

                    title = safe_str(item.find('Title').text)
                    content_clean = clean_html(safe_str(item.find('Content').text))
                    short_desc_raw = safe_str(item.find('ShortDescription').text)
                    short_desc_clean = clean_html(short_desc_raw)
                    author = extract_author(short_desc_raw)
                    
                    cat_tag = item.find('Productcategories')
                    lang = "hu"
                    cat_raw = safe_str(cat_tag.text) if cat_tag is not None else ""
                    if "roman" in cat_raw.lower() or "român" in cat_raw.lower(): lang = "ro"
                    
                    url = safe_str(item.find('Permalink').text or item.find('Link').text)
                    img_tag = item.find('ImageURL') or item.find('Image')
                    image = safe_str(img_tag.text) if img_tag is not None else ""
                    
                    price = safe_str(item.find('Price').text)
                    sale = safe_str(item.find('SalePrice').text)
                    final_price = sale if sale else price

                    stock_status = "instock"

                    combined_text = f"CÍM: {title} | SZERZŐ: {author} | KAT: {cat_raw} | {short_desc_clean}"
                    res = self.client_ai.embeddings.create(input=combined_text[:8000], model="text-embedding-3-small")
                    
                    full_search_text = f"{title} {author} {cat_raw} {short_desc_clean} {content_clean}"
                    
                    metadata = {
                        "title": title, "price": final_price, "url": url, "image_url": image,
                        "lang": lang, "stock": stock_status, "author": author, "category": cat_raw,
                        "short_desc": short_desc_clean[:500],
                        "full_search_text": full_search_text[:8000],
                        "last_seen": current_sync_ts 
                    }
                    
                    batch_vectors.append((book_id, res.data[0].embedding, metadata))
                    count += 1

                    if len(batch_vectors) >= 50:
                        self.index.upsert(vectors=batch_vectors)
                        batch_vectors = []

                except Exception as e:
                    continue

            if batch_vectors:
                self.index.upsert(vectors=batch_vectors)
            
            print(f"✅ [AUTO] Feltöltés kész ({count} db). Most jön a takarítás...")

            try:
                self.index.delete(
                    filter={
                        "last_seen": {"$lt": current_sync_ts}
                    }
                )
                print("🧹 [AUTO] Elavult (készlethiányos) könyvek törölve az adatbázisból.")
            except Exception as e:
                print(f"⚠️ [AUTO] Törlési hiba (lehet, hogy Serverless indexen limitált): {e}")

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
                full_n = normalize_text(str(meta.get('full_search_text', '')))
                
                score = 0
                if not keywords: score = m['score'] * 100
                else:
                    matches_cnt = 0
                    for kw in keywords:
                        hit = False
                        if kw in title_n: score += 100; hit = True
                        elif kw in auth_n: score += 80; hit = True
                        elif kw in full_n: score += 20; hit = True
                        if hit: matches_cnt += 1
                    if matches_cnt == len(keywords) and len(keywords) > 1: score += 200
                
                if keywords and score < 10: continue
                m['final_relevance'] = score
                scored.append(m)

            scored.sort(key=lambda x: x['final_relevance'], reverse=True)
            return scored[:20]
        except: return []

    def process_message(self, user_input):
        try:
            res = self.client_ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system", "content":"Detect Language (hu/ro) and Intent (SEARCH/INFO). Output: LANG | INTENT"}, {"role":"user", "content":user_input}], temperature=0.1
            )
            p = res.choices[0].message.content.split('|')
            lang, intent = p[0].strip().lower(), p[1].strip().upper()
        except: lang, intent = 'hu', 'SEARCH'

        if intent == 'INFO':
            policy = self.get_dynamic_policy()
            prompt = f"Te Booksy vagy. Válaszolj EZEK alapján:\n{policy}\nHa nincs válasz, küldd a kapcsolati oldalt."
            instr = "Reply in ROMANIAN." if lang == 'ro' else "Reply in HUNGARIAN."
            ai_res = self.client_ai.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role":"system", "content":prompt}, {"role":"system", "content":instr}, {"role":"user", "content":user_input}], temperature=0.1
            )
            return {"reply": ai_res.choices[0].message.content, "products": []}
        
        matches = self.search_engine_logic(user_input, lang)
        
        ctx = ""
        prods = []
        if not matches:
            msg = "Sajnos nem találtam készleten lévő könyvet." if lang == 'hu' else "Nu am găsit."
            tip = "\n\n💡 *Tipp: Ha mindent látni szeretnél, írd hozzá: „minden nyelven”!*"
            return {"reply": msg + tip, "products": []}

        for m in matches:
            meta = m['metadata']
            p = {"title": str(meta.get('title')), "price": str(meta.get('price')), "url": str(meta.get('url')), "image": str(meta.get('image_url'))}
            prods.append(p)
            ctx += f"- {p['title']} (Szerző: {meta.get('author')}, Ár: {p['price']} RON, Info: {str(meta.get('short_desc'))[:100]})\n"
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
    scheduler.add_job(bot.updater.run_daily_update, 'cron', hour=3, minute=0)
    scheduler.start()
    print("⏰ Automata tükörszinkronizáció beidőzítve (03:00).")
    yield
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def home(): return {"status": "Booksy V26.2 (Syntax Error Fixed)"}

@app.post("/hook")
def hook_endpoint(request: HookRequest): return {"hook": bot.generate_sales_hook(request)}

@app.post("/chat")
def chat_endpoint(request: ChatRequest): return bot.process_message(request.message)

@app.post("/force-update")
def force_update(background_tasks: BackgroundTasks):
    background_tasks.add_task(bot.updater.run_daily_update)
    return {"message": "Tükörszinkronizáció elindítva..."}