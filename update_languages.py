import os
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from pinecone import Pinecone

# Beállítások
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "booksy-index"

if not PINECONE_API_KEY:
    print("HIBA: Nincs API kulcs!")
    exit()

# Csatlakozás
print("📡 Csatlakozás a Pinecone-hoz...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

print("🌍 NYELVEK FRISSÍTÉSE AZ ADATBÁZISBAN...")
print("📂 export.xml beolvasása...")

try:
    tree = ET.parse('export.xml')
    root = tree.getroot()
except Exception as e:
    print(f"HIBA: Nem találom vagy nem tudom olvasni az export.xml-t! {e}")
    exit()

# Elem keresése (post vagy item)
items = root.findall('.//post')
if not items:
    items = root.findall('.//item')

count = 0
hu_count = 0
ro_count = 0
skipped = 0

print(f"Összesen {len(items)} könyv vizsgálata indul...")

for post in items:
    try:
        # ID keresése
        id_tag = post.find('ID')
        if id_tag is None:
            id_tag = post.find('g:id')
        
        if id_tag is None:
            continue
            
        book_id = id_tag.text
        
        # KATEGÓRIA VIZSGÁLATA
        # Próbáljuk több néven is
        cat_tag = post.find('Productcategories')
        if cat_tag is None:
            cat_tag = post.find('categories')
        
        if cat_tag is None or not cat_tag.text:
            skipped += 1
            continue
            
        categories = cat_tag.text.lower()
        lang_code = "unknown"

        # LOGIKA: Mit keresünk a kategória nevében?
        if "magyar" in categories:
            lang_code = "hu"
            hu_count += 1
        elif "roman" in categories or "român" in categories:
            lang_code = "ro"
            ro_count += 1
        
        if lang_code == "unknown":
            skipped += 1
            continue

        # FRISSÍTÉS A FELHŐBE
        index.update(
            id=book_id,
            set_metadata={"lang": lang_code}
        )
        
        count += 1
        if count % 100 == 0:
            print(f"⏳ {count} db felcímkézve... (HU: {hu_count}, RO: {ro_count})")

    except Exception as e:
        print(f"Hiba egy elemnél: {e}")
        continue

print(f"\n✅ KÉSZ! Nyelvi statisztika:")
print(f"🇭🇺 Magyar könyvek: {hu_count}")
print(f"🇷🇴 Román könyvek: {ro_count}")
print(f"⏩ Kihagyva (nem beazonosítható nyelv): {skipped}")