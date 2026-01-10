import xml.etree.ElementTree as ET
import json
import re
import os

# --- BEÁLLÍTÁSOK ---
INPUT_FILE = "export.xml"
OUTPUT_FILE = "booksy_data.json"

class BooksyKnowledgeBuilder:
    def __init__(self, xml_file):
        self.xml_file = xml_file
        self.items = []
        self.stats = {"hu": 0, "ro": 0, "unknown": 0, "total": 0}

    def clean_html(self, raw_html):
        if not raw_html: return ""
        text = str(raw_html).replace('</div>', '\n').replace('</p>', '\n').replace('<br>', '\n')
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', text)
        return "\n".join([line.strip() for line in cleantext.splitlines() if line.strip()])

    def determine_language(self, categories_str):
        if not categories_str: return "unknown"
        if "Magyar nyelvű könyvek" in categories_str: return "hu"
        if "Cărți în limba română" in categories_str: return "ro"
        return "unknown"

    def parse_xml(self):
        print(f"--- Booksy AI: '{self.xml_file}' ÚJRA feldolgozása... ---")
        
        try:
            tree = ET.parse(self.xml_file)
            root = tree.getroot()
        except Exception as e:
            print(f"[HIBA] Az XML fájl hibás: {e}")
            return

        posts = root.findall('post')
        if not posts: posts = root.findall('.//item')
        
        print(f"[Info] {len(posts)} termék megtalálva. Készlet infók kinyerése...")

        for post in posts:
            try:
                # 1. KÉSZLET INFÓ KINYERÉSE (EZ A LÉNYEG!)
                # Megnézzük a StockStatus taget. Ha nincs, akkor 'outofstock'.
                stock_raw = post.find('StockStatus')
                if stock_raw is not None and stock_raw.text:
                    stock_status = str(stock_raw.text).lower().strip()
                else:
                    stock_status = "outofstock"

                # Egyéb adatok
                cats = post.find('Productcategories').text if post.find('Productcategories') is not None else ""
                lang = self.determine_language(cats)
                
                if lang == "unknown": 
                    continue

                p_id = post.find('ID').text if post.find('ID') is not None else "N/A"
                title = post.find('Title').text if post.find('Title') is not None else "Névtelen"
                price = post.find('Price').text if post.find('Price') is not None else "0"
                
                desc_raw = post.find('Content').text
                short_desc_raw = post.find('ShortDescription').text
                description = self.clean_html(desc_raw)
                meta_desc = self.clean_html(short_desc_raw)

                image_url = post.find('ImageURL').text if post.find('ImageURL') is not None else ""
                permalink = post.find('Permalink').text if post.find('Permalink') is not None else ""

                # AI szöveg
                ai_text = (
                    f"Cím: {title}\n"
                    f"Nyelv: {lang}\n"
                    f"Ár: {price} RON\n" # Itt is beégetjük a RON-t
                    f"Készlet: {stock_status}\n" # Beleírjuk a szövegbe is, hogy az AI is lássa!
                    f"Infók: {meta_desc}\n"
                    f"Leírás: {description}"
                )

                item = {
                    "id": p_id,
                    "text": ai_text,
                    "metadata": {
                        "title": title, 
                        "url": permalink, 
                        "image": image_url,
                        "price": price, 
                        "lang": lang, 
                        "stock": stock_status  # <--- ITT KELL LENNIE A KULCSNAK!
                    }
                }
                
                self.items.append(item)
                self.stats['total'] += 1

                if self.stats['total'] % 5000 == 0:
                    print(f"... {self.stats['total']} termék feldolgozva ...")

            except Exception as e:
                continue

        # Mentés
        print(f"--- FELDOLGOZÁS KÉSZ! ---")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.items, f, indent=2, ensure_ascii=False)
        print(f"Az új adatbázis (készletinfóval) elmentve: {OUTPUT_FILE}")

if __name__ == "__main__":
    converter = BooksyKnowledgeBuilder(INPUT_FILE)
    converter.parse_xml()