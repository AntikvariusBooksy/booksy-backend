import os
import time
import re
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# --- BEÁLLÍTÁSOK ---
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "booksy-index"

# Ezeket az oldalakat olvassa be a jogi válaszokhoz
POLICY_URLS = {
    "FIZETÉS": "https://www.antikvarius.ro/hu/fizetesi-informaciok/",
    "SZÁLLÍTÁS": "https://www.antikvarius.ro/hu/szallitasi-informaciok/",
    "ÁSZF": "https://www.antikvarius.ro/hu/altalanos-szerzodesi-es-felhasznalasi-feltetelek/"
}

# Csatlakozás
if not PINECONE_API_KEY or not OPENAI_API_KEY:
    print("HIBA: Hiányzik a PINECONE_API_KEY vagy OPENAI_API_KEY a .env fájlból!")
    exit()

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    client = OpenAI(api_key=OPENAI_API_KEY)
    index = pc.Index(INDEX_NAME)
except Exception as e:
    print(f"HIBA a csatlakozásnál: {e}")
    exit()

# ==========================================
# 1. RÉSZ: JOGI INFORMÁCIÓK SZKENNELÉSE
# ==========================================
print("\n🌐 1/2. LÉPÉS: Weboldal jogi oldalainak szkennelése...")

def scrape_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Kitisztítjuk a felesleget
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()
            
            # Szöveg kinyerése
            text = soup.get_text(separator=' ')
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = '\n'.join(chunk for chunk in chunks if chunk)
            return clean_text[:4000] 
        else:
            print(f"⚠️ Hiba az URL elérésekor ({response.status_code}): {url}")
            return ""
    except Exception as e:
        print(f"⚠️ Kivétel történt ({url}): {e}")
        return ""

full_policy_text = "[TUDÁSBÁZIS AZ ÜGYFÉLSZOLGÁLATHOZ - FRISSÍTVE: MA]\n"

for category, url in POLICY_URLS.items():
    print(f"   - Olvasás: {category}...")
    content = scrape_text_from_url(url)
    if content:
        full_policy_text += f"\n--- {category} INFORMÁCIÓK ---\n{content}\n"

# Feltöltés a Pinecone-ba "store_policy" ID-val
try:
    res = client.embeddings.create(input="policy definition", model="text-embedding-3-small")
    embedding = res.data[0].embedding
    
    metadata = {
        "type": "policy",
        "content": full_policy_text
    }
    
    index.upsert(vectors=[("store_policy", embedding, metadata)])
    print("✅ Jogi információk sikeresen frissítve!")
except Exception as e:
    print(f"❌ Hiba a jogi infók feltöltésekor: {e}")


# ==========================================
# 2. RÉSZ: KÖNYVEK FELTÖLTÉSE (FULL DATA)
# ==========================================
print("\n📚 2/2. LÉPÉS: Könyvek feltöltése az export.xml-ből...")

try:
    tree = ET.parse('export.xml')
    root = tree.getroot()
except Exception as e:
    print(f"HIBA: Nem találom az export.xml-t! {e}")
    exit()

items = root.findall('.//post')
if not items: items = root.findall('.//item')

batch_size = 50
batch_vectors = []
count = 0
current_ts = int(time.time()) # Időbélyeg a szinkronhoz

# --- Segédfüggvények ---

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

def get_best_price(item):
    price = safe_str(item.find('Price').text if item.find('Price') is not None else "")
    regular = safe_str(item.find('RegularPrice').text if item.find('RegularPrice') is not None else "")
    sale = safe_str(item.find('SalePrice').text if item.find('SalePrice') is not None else "")
    if sale: return sale
    elif regular: return regular
    elif price: return price
    return "0"

def process_categories(cat_string):
    if not cat_string: return "hu", "", ""
    lang = "hu"
    if "roman" in cat_string.lower() or "român" in cat_string.lower(): lang = "ro"
    paths = cat_string.split(',')
    all_cats = set()
    for path in paths:
        parts = path.split('>')
        cleaned_parts = [p.strip() for p in parts]
        for part in cleaned_parts:
            if part.lower() not in ["magyar nyelvű könyvek", "új könyvek", "productcategories", "antikvár könyvek"]:
                all_cats.add(part)
    return lang, " ".join(all_cats), ""

# --- Feldolgozás ---

print(f"Könyvek száma: {len(items)}")
ns = {'g': 'http://base.google.com/ns/1.0'}

for item in items:
    try:
        id_tag = item.find('ID')
        if id_tag is None: id_tag = item.find('g:id', ns)
        if id_tag is None or not id_tag.text: continue 
        book_id = safe_str(id_tag.text)

        title = safe_str(item.find('Title').text if item.find('Title') is not None else "")
        content_raw = safe_str(item.find('Content').text if item.find('Content') is not None else "")
        content_clean = clean_html(content_raw)
        
        short_desc_raw = safe_str(item.find('ShortDescription').text if item.find('ShortDescription') is not None else "")
        short_desc_clean = clean_html(short_desc_raw)
        
        author = extract_author(short_desc_raw)
        
        cat_tag = item.find('Productcategories')
        cat_raw = safe_str(cat_tag.text) if cat_tag is not None else ""
        lang, cat_keywords, main_category = process_categories(cat_raw)

        url = safe_str(item.find('Permalink').text if item.find('Permalink') is not None else "")
        if not url: url = safe_str(item.find('Link').text if item.find('Link') is not None else "")
        
        img_tag = item.find('ImageURL')
        if img_tag is None: img_tag = item.find('Image')
        image = safe_str(img_tag.text) if img_tag is not None else ""

        stock_tag = item.find('StockStatus')
        stock_status = safe_str(stock_tag.text) if stock_tag is not None else "instock"

        final_price = get_best_price(item)

        # AI Embedding (Vektoros kereséshez)
        combined_text_ai = (
            f"CÍM: {title} | SZERZŐ: {author} | KATEGÓRIA: {cat_keywords} | "
            f"RÖVID: {short_desc_clean} | TARTALOM: {content_clean[:800]}"
        )
        res = client.embeddings.create(input=combined_text_ai, model="text-embedding-3-small")
        embedding = res.data[0].embedding

        # Full Text Search Blob (A Pythonos pontos kereséshez)
        full_search_text = f"{title} {author} {cat_keywords} {short_desc_clean} {content_clean}"
        
        metadata = {
            "title": title,
            "price": final_price,
            "url": url,
            "image_url": image,
            "lang": lang,
            "stock": stock_status,
            "author": author,
            "category": cat_keywords,
            "short_desc": short_desc_clean[:500],
            "full_search_text": full_search_text[:9000], # Limit, hogy beférjen
            "last_seen": current_ts # Fontos a szinkronhoz!
        }

        batch_vectors.append((book_id, embedding, metadata))
        count += 1

        if len(batch_vectors) >= batch_size:
            index.upsert(vectors=batch_vectors)
            batch_vectors = []
            print(f"✅ {count} db feldolgozva... ({title})")
            
    except Exception as e:
        continue

if batch_vectors:
    index.upsert(vectors=batch_vectors)
    print(f"✅ Maradék {len(batch_vectors)} db feltöltve.")

print(f"🏁 TELJES FOLYAMAT KÉSZ! Adatbázis frissítve.")