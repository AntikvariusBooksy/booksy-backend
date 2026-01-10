import os
import time
import difflib
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# --- KONFIGURÁCIÓ ---
INDEX_NAME = "booksy-index"
load_dotenv()

class BooksyChat:
    def __init__(self):
        api_key_openai = os.getenv("OPENAI_API_KEY")
        api_key_pinecone = os.getenv("PINECONE_API_KEY")
        
        if not api_key_openai or not api_key_pinecone:
            raise ValueError("Hiányzó API kulcsok a .env fájlban!")

        self.client_ai = OpenAI(api_key=api_key_openai)
        self.pc = Pinecone(api_key=api_key_pinecone)
        self.index = self.pc.Index(INDEX_NAME)

        # --- AZ ÜZLETI TUDÁSBÁZIS ---
        self.store_policy = """
        [SZÁLLÍTÁS / LIVRARE - KRITIKUS FONTOSSÁGÚ!]
        A kézbesítés ideje = FELDOLGOZÁS + SZÁLLÍTÁS.
        
        1. FELDOLGOZÁSI IDŐ (Ami a termék elérhetőségétől függ):
           - "Raktáron" (In Stock): 2-4 munkanap.
           - "Utánrendelhető / Külső raktár" (Backorder): 7-30 nap (beszerzési idő).
        
        2. SZÁLLÍTÁSI IDŐ (Futár):
           - Románia: +24-48 óra.
           - Magyarország: +2-4 munkanap.
           - EU: +3-7 munkanap.

        [KÖLTSÉGEK]
        - Románia: 22 RON.
        - Magyarország: ~3200 HUF.
        - EU: ~23 EUR.

        [EGYÉB INFÓK]
        - Fizetés: Bankkártya (Bárhol), Utánvét (Csak Románia).
        - Kapcsolat: +40 755 583 310, info@antikvarius.ro
        - Visszaküldés: 30 nap.
        """
        
        self.system_prompt = f"""
        Te Booksy vagy, az Antikvarius.ro webshop mesterséges intelligencia alapú értékesítője.
        
        TUDÁSBÁZIS:
        {self.store_policy}

        SZIGORÚ SZABÁLYOK:
        1. NYELV: HU kérdés -> HU válasz. RO kérdés -> RO válasz.
        2. PÉNZNEM: Mindig 'RON'.
        
        3. SZÁLLÍTÁSI IDŐ (NAGYON FONTOS):
           Amikor szállítási időről beszélsz, MINDIG különböztesd meg a két esetet:
           - "Ha a termék raktáron van: 2-4 nap feldolgozás."
           - "Ha utánrendelhető (külső raktár): 7-30 nap feldolgozás."
           - És ehhez add hozzá a futár idejét.
           Soha ne ígérj csak 2-4 napot anélkül, hogy megemlítenéd a külső raktáras lehetőséget!

        KÉT ÜZEMMÓD:
        A) KÖNYAJÁNLÓ (SEARCH): Context alapján. Formátum: [CÍM](URL) - ÁR RON.
        B) ÜGYFÉLSZOLGÁLAT (INFO): Tudásbázis alapján.
        """

    def generate_search_params(self, user_input):
        response = self.client_ai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """
                 Feladat: Elemzed a felhasználó bemenetét.
                 1. Nyelv: 'hu' vagy 'ro'.
                 2. Szándék: 'SEARCH' (könyv) vagy 'INFO' (szállítás, fizetés, kapcsolat).
                 3. Kulcsszó (ha SEARCH).
                 Válasz: "hu | SEARCH | kulcsszavak" vagy "ro | INFO | null"
                 """},
                {"role": "user", "content": user_input}
            ],
            temperature=0.1
        )
        result = response.choices[0].message.content
        try:
            parts = result.split('|')
            return parts[0].strip().lower(), parts[1].strip(), parts[2].strip()
        except:
            return "hu", "SEARCH", user_input

    def search_books(self, query_text, lang_filter):
        response = self.client_ai.embeddings.create(input=query_text, model="text-embedding-3-small")
        query_vector = response.data[0].embedding
        search_results = self.index.query(
            vector=query_vector,
            top_k=20, 
            include_metadata=True,
            filter={"stock": "instock", "lang": lang_filter}
        )
        return search_results

    def chat(self):
        print("\n📚 --- Szia! Booksy v11.0 (Készlet-Tudatos Szállítás) ---")
        print("(Kilépés: 'exit')")
        
        while True:
            user_input = input("\nTe: ")
            
            if user_input.lower() in ["exit", "kilepes", "quit"]:
                print("Booksy: Viszlát! 👋")
                break
            
            print("...(elemzés / analiză)...")
            detected_lang, intent, keywords = self.generate_search_params(user_input)
            
            context_text = ""
            if intent == "SEARCH":
                results = self.search_books(keywords, detected_lang)
                seen_titles = []
                count = 0
                if not results.get('matches'):
                    context_text = "Nincs találat."
                else:
                    for match in results['matches']:
                        meta = match['metadata']
                        title = str(meta.get('title', 'N/A'))
                        is_dup = False
                        for seen in seen_titles:
                            if difflib.SequenceMatcher(None, title.lower(), seen.lower()).ratio() > 0.85:
                                is_dup = True; break
                        if is_dup: continue
                        seen_titles.append(title)
                        context_text += f"- [CÍM: {title}](URL: {meta.get('url')}) - ÁR: {meta.get('price')} RON\n"
                        count += 1
                        if count >= 6: break
            else:
                context_text = "HASZNÁLD A TUDÁSBÁZIST!"

            if detected_lang == 'ro':
                lang_instruction = "IMPORTANT: Reply in ROMANIAN only!"
            else:
                lang_instruction = "IMPORTANT: Reply in HUNGARIAN only!"

            response = self.client_ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "system", "content": lang_instruction},
                    {"role": "user", "content": f"User Question: {user_input}\n\nContext:\n{context_text}"}
                ],
                temperature=0.5
            )
            print(f"Booksy: {response.choices[0].message.content}")

if __name__ == "__main__":
    bot = BooksyChat()
    bot.chat()