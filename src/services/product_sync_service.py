# Version: 1.2.1 - Booksy Agentic Sync (Strict SKU separation, No Merge Fix)
import os
import logging
import json
from typing import Dict, Any, Optional, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Logger beállítása a meglévő rendszerhez illeszkedve
logger = logging.getLogger(__name__)

class BooksySyncAgent:
    """
    Ez az ágens felelős a könyvek importálásáért és szinkronizálásáért a Booksy v94+ rendszerben.
    
    MŰKÖDÉSI ELV:
    1. Agentic Mode: OpenAI használata az adatok tisztítására (helyesírás, kategória, SEO).
    2. Strict SKU Mode: A termékazonosítás KIZÁRÓLAG a 'sku' mező alapján történik.
       - Ha a SKU nem létezik az adatbázisban -> ÚJ termék létrehozása (Create).
       - Ha a SKU létezik -> Meglévő termék frissítése (Update).
       - Cím/Szerző egyezés esetén SEM vonunk össze termékeket (antikvár jelleg miatt).
    """

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
             logger.error("CRITICAL: OPENAI_API_KEY is missing via env vars.")
        
        # Agentic AI kliens inicializálása
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4-turbo-preview"
        
        logger.info("BooksySyncAgent service initialized (Strict SKU Mode).")

    def _call_ai_agent(self, prompt: str) -> Dict[str, Any]:
        """
        Belső segédfüggvény az AI hívásához JSON módban.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are the Booksy Data Specialist AI. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"AI Agent Error: {e}")
            return {}

    def enhance_book_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Az AI feljavítja a termék leíró adatait.
        FONTOS: A technikai azonosítókat (SKU, ID, Stock) nem módosíthatja.
        """
        if not raw_data.get('title'):
            return raw_data

        # Csak a releváns adatokat küldjük az AI-nak
        ai_input = {
            "title": raw_data.get("title"),
            "author": raw_data.get("author"),
            "description": raw_data.get("description", ""),
            "condition": raw_data.get("condition", "")
        }

        prompt = f"""
        Analyze and enhance this antique book data for the Booksy e-commerce store.
        
        Input Data:
        {json.dumps(ai_input, ensure_ascii=False)}

        Tasks:
        1. Correct typos in Title and Author.
        2. Assign a standard category.
        3. Write a short, professional SEO description (Hungarian).
        4. Extract keywords.

        Output Format (JSON):
        {{
            "title": "Corrected Title",
            "author": "Corrected Author",
            "short_description": "SEO friendly description...",
            "category": "History",
            "tags": ["tag1", "tag2"]
        }}
        """
        
        enhanced_fields = self._call_ai_agent(prompt)
        
        # Összefésülés
        final_data = raw_data.copy()
        if enhanced_fields:
            final_data.update(enhanced_fields)
        
        # BIZTONSÁGI ZÁR: Visszaállítjuk a kritikus üzleti adatokat
        final_data['sku'] = raw_data.get('sku')
        final_data['price'] = raw_data.get('price')
        final_data['stock'] = raw_data.get('stock')
        final_data['id'] = raw_data.get('id')

        return final_data

    def find_product_by_sku(self, sku: str) -> Optional[Dict[str, Any]]:
        """
        Keresés KIZÁRÓLAG SKU alapján az adatbázisban/shopban.
        """
        # IDE JÖN A DB LOGIKA (pl. return db.find_one(...))
        # Jelenleg None-t ad vissza, hogy tesztelhessük az új termék létrehozást
        return None 

    def save_product(self, product_data: Dict[str, Any], is_update: bool = False):
        """ Mentés az adatbázisba. """
        action = "UPDATING" if is_update else "CREATING NEW"
        logger.info(f"{action} Product -> SKU: {product_data.get('sku')} | Title: {product_data.get('title')}")
        # Ide jön a DB insert/update kód

    def sync_book(self, raw_book_data: Dict[str, Any]):
        """
        Vezérlő logika: SKU check -> AI Enhance -> Save
        """
        sku = raw_book_data.get('sku')
        if not sku:
            logger.warning("Skipping book import: Missing SKU.")
            return

        # 1. LÉPÉS: Szigorú SKU ellenőrzés (Nincs cím alapú összevonás!)
        existing_product = self.find_product_by_sku(sku)

        # 2. LÉPÉS: AI adatgazdagítás
        processed_data = self.enhance_book_data(raw_book_data)

        # 3. LÉPÉS: Mentés
        if existing_product:
            processed_data['id'] = existing_product.get('id')
            self.save_product(processed_data, is_update=True)
        else:
            self.save_product(processed_data, is_update=False)

    def run_batch_import(self, books_list: List[Dict[str, Any]]):
        logger.info(f"Starting batch import of {len(books_list)} items with Strict SKU policy.")
        for book in books_list:
            self.sync_book(book)
