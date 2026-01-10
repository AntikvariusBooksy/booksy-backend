import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# --- KONFIGURÁCIÓ V3 ---
BATCH_SIZE = 50 
JSON_FILE = "booksy_data.json"
INDEX_NAME = "booksy-index" 

# LIMIT V3: 10.000 karakterre csökkentve.
# Ez garantáltan belefér az OpenAI limitbe (8192 token),
# még akkor is, ha nagyon sűrű a szöveg.
MAX_CHARS = 10000 

load_dotenv()

def main():
    print("--- Booksy AI: Tudásbázis Feltöltő v3 (Stabil) ---")
    
    api_key_openai = os.getenv("OPENAI_API_KEY")
    api_key_pinecone = os.getenv("PINECONE_API_KEY")

    if not api_key_openai or not api_key_pinecone:
        print("[HIBA] Hiányoznak az API kulcsok a .env fájlból!")
        return

    try:
        client_ai = OpenAI(api_key=api_key_openai)
        pc = Pinecone(api_key=api_key_pinecone)
        index = pc.Index(INDEX_NAME)
        stats = index.describe_index_stats()
        print(f"[Kapcsolat OK] Pinecone Index elérhető. Jelenleg feltöltve: {stats['total_vector_count']} db")
    except Exception as e:
        print(f"[HIBA] Kapcsolódási hiba: {e}")
        return

    print(f"[Folyamat] {JSON_FILE} beolvasása...")
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[Siker] {len(data)} könyv betöltve.")
    except Exception as e:
        print(f"[HIBA] JSON olvasási hiba: {e}")
        return

    total_items = len(data)
    print(f"[Start] Feltöltés indul... (Ne állítsd le, ha hibát látsz, a gép megoldja!)")

    # Innen folytatjuk, ha esetleg már vannak fent adatok, de az overwrite miatt most mindegy,
    # az upsert felülírja a meglévőt, nem duplikál.
    
    for i in range(0, total_items, BATCH_SIZE):
        batch = data[i : i + BATCH_SIZE]
        
        vectors_to_upsert = []
        texts_to_embed = []
        valid_indices = []

        # --- ADAT ELŐKÉSZÍTÉS ---
        for idx, item in enumerate(batch):
            # 1. Szöveg vágása (Szigorú 10k limit)
            original_text = str(item.get('text', ''))
            if len(original_text) > MAX_CHARS:
                safe_text = original_text[:MAX_CHARS] + "... [vágva]"
            else:
                safe_text = original_text
            
            texts_to_embed.append(safe_text)
            
            # 2. Metadata tisztítása
            clean_metadata = {}
            # Biztonsági ellenőrzés: ha nincs metadata, üreset adunk
            meta = item.get('metadata', {})
            for key, value in meta.items():
                if value is None:
                    clean_metadata[key] = "" 
                else:
                    clean_metadata[key] = str(value)
            
            # Az eredeti item-et frissítjük
            item['metadata'] = clean_metadata
            valid_indices.append(idx)

        if not texts_to_embed:
            continue

        # --- FELTÖLTÉS ---
        try:
            # A. Vektorizálás
            response = client_ai.embeddings.create(
                input=texts_to_embed,
                model="text-embedding-3-small"
            )
            embeddings = [record.embedding for record in response.data]

            # B. Összepárosítás
            for j, emb in enumerate(embeddings):
                item = batch[valid_indices[j]]
                vectors_to_upsert.append( (str(item['id']), emb, item['metadata']) )

            # C. Feltöltés
            index.upsert(vectors=vectors_to_upsert)

            progress = min(i + BATCH_SIZE, total_items)
            # Kiírjuk százalékosan is
            percent = int((progress / total_items) * 100)
            print(f"[{progress}/{total_items}] Kész ({percent}%)")

        except Exception as e:
            # Itt kezeljük a hibát, hogy ne álljon le a program!
            print(f"⚠️ FIGYELEM: Hiba történt a {i}. csomagnál. Ezt az 50 könyvet átugrom, és folytatom.")
            # Ha nagyon kíváncsi vagy a hibára, kiveheted a kommentet:
            # print(f"Hiba részletei: {e}")
            time.sleep(1) # Pici pihenő és megyünk tovább

    print("--- GRATULÁLOK! A TELJES LISTA FELDOLGOZVA! ---")

if __name__ == "__main__":
    main()