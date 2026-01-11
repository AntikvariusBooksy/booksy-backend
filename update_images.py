import os
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from pinecone import Pinecone

# 1. Beállítások betöltése
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "booksy-index"

if not PINECONE_API_KEY:
    print("HIBA: Nincs Pinecone API kulcs a .env fájlban!")
    exit()

# 2. Csatlakozás a felhőhöz
print("📡 Csatlakozás a Pinecone-hoz...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# 3. XML feldolgozása
print("📂 export.xml beolvasása...")
try:
    tree = ET.parse('export.xml')
    root = tree.getroot()
except Exception as e:
    print(f"HIBA: Nem tudom olvasni az export.xml-t: {e}")
    exit()

print("🖼️ Képek frissítése indul... (Ez eltarthat pár percig)")

count = 0
updated = 0
skipped = 0

# Végigmegyünk minden <post> elemen
# A te XML-edben a gyökérelem alatt közvetlenül vannak a post-ok? 
# Ha a root maga a lista, akkor így jó. Ha van 'channel' vagy 'channel/item', akkor finomítani kell.
# A snippet alapján feltételezem, hogy a <post> elemeket kell keresni.

items = root.findall('.//post') # Megkeresi bárhol a 'post' elemeket
if not items:
    # Ha nem talál, megpróbáljuk a 'channel/item' logikát, hátha RSS feed
    items = root.findall('.//item') 

print(f"Összesen {len(items)} terméket találtam az XML-ben.")

for post in items:
    try:
        # ID kinyerése (Ez alapján azonosítjuk a könyvet)
        # Próbáljuk az <ID> taget
        id_tag = post.find('ID')
        if id_tag is None:
            continue
        book_id = id_tag.text
        
        # KÉP kinyerése <ImageURL>
        img_tag = post.find('ImageURL')
        if img_tag is None or not img_tag.text:
            skipped += 1
            continue
            
        image_url = img_tag.text

        # 4. KÜLDÉS A FELHŐBE (Update Metadata)
        # Ez a parancs csak a metadata-t frissíti, nem bántja a vektort!
        index.update(
            id=book_id,
            set_metadata={"image_url": image_url}
        )
        
        updated += 1
        count += 1
        
        # Visszajelzés 100-anként
        if count % 100 == 0:
            print(f"⏳ {count} db feldolgozva... (Legutóbbi: ID {book_id})")

    except Exception as e:
        print(f"Hiba az egyik elemnél: {e}")
        continue

print(f"\n✅ KÉSZ! Eredmény:")
print(f"- Frissített könyvek (Kép hozzáadva): {updated} db")
print(f"- Kihagyva (Nincs kép vagy ID): {skipped} db")
print("Most próbáld ki a Chatet, és látnod kell a képeket!")