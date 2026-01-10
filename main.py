import os
import difflib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# --- KONFIGURÁCIÓ ---
load_dotenv()
INDEX_NAME = "booksy-index"

# Inicializáljuk a Webes Appot (FastAPI)
app = FastAPI()

# CORS beállítások
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Adatmodell
class ChatRequest(BaseModel):
    message: str

# --- BOOKSY LOGIKA (Agy) ---
class BooksyBrain:
    def __init__(self):
        self.api_key_openai = os.getenv("OPENAI_API_KEY")
        self.api_key_pinecone = os.getenv("PINECONE_API_KEY")
        
        if not self.api_key_openai or not self.api_key_pinecone:
            print("HIBA: Hiányzó API kulcsok!")
        
        self.client_ai = OpenAI(api_key=self.api_key_openai)
        self.pc = Pinecone(api_key=self.api_key_pinecone)
        self.index = self.pc.Index(INDEX_NAME)

        # ÜZLETI TUDÁSBÁZIS
        self.store_policy = """
        [SZÁLLÍTÁS / LIVRARE - PROTOKOLL]
        A szállítási időről szóló kérdéseknél MINDIG fel kell sorolnod mindkét lehetőséget.
        
        A KÉPLET: Feldolgozás + Futár = Kézbesítés.

        1. FELDOLGOZÁSI IDŐ (Raktár):
           - "A" ESET: Raktáron lévő termék (In Stock) -> 2-4 munkanap.
           - "B" ESET: Utánrendelhető / Külső raktár (Backorder) -> 7-30 nap (beszerzés).
        
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
        
        3. ANTI-HALLUCINÁCIÓ (KRITIKUS):
           - CSAK ÉS KIZÁRÓLAG a 'Context'-ben kapott könyveket ajánlhatod!
           - SOHA ne találj ki könyveket fejből!
           - Ha a Context üres vagy nem releváns, mondd azt, hogy "Sajnos jelenleg nincs ilyen könyvünk."
        
        4. SZÁLLÍTÁS:
           - Mindig említsd meg a 2-4 napos (raktár) ÉS a 7-30 napos (utánrendelés) opciót is.

        KÉT ÜZEMMÓD:
        A) KÖNYAJÁNLÓ (SEARCH): Context alapján. Formátum: [CÍM](URL) - ÁR RON.
        B) ÜGYFÉLSZOLGÁLAT (INFO): Tudásbázis alapján.
        """

    def generate_search_params(self, user_input):
        try:
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
            parts = result.split('|')
            return parts[0].strip().lower(), parts[1].strip(), parts[2].strip()
        except:
            return "hu", "SEARCH", user_input

    def search_books(self, query_text, lang_filter):
        try:
            response = self.client_ai.embeddings.create(input=query_text, model="text-embedding-3-small")
            query_vector = response.data[0].embedding
            # Keresés
            search_results = self.index.query(
                vector=query_vector,
                top_k=20, 
                include_metadata=True,
                filter={"stock": "instock", "lang": lang_filter}
            )
            return search_results
        except Exception as e:
            print(f"Hiba a keresésnél: {e}")
            return {"matches": []}

    def process_message(self, user_input):
        # 1. Elemzés
        detected_lang, intent, keywords = self.generate_search_params(user_input)
        
        context_text = ""
        has_results = False # Figyelő kapcsoló

        # 2. Adatgyűjtés
        if intent == "SEARCH":
            results = self.search_books(keywords, detected_lang)
            seen_titles = []
            count = 0
            
            # --- BIZTONSÁGI ZÁR: Ha nincs találat a Pinecone-ban ---
            if not results.get('matches') or len(results['matches']) == 0:
                if detected_lang == 'ro':
                    return "Din păcate, nu am găsit nicio carte pe acest subiect în stocul nostru actual. 😞 Poate încercați o altă căutare?"
                else:
                    return "Sajnos jelenleg nem találtam ilyen témájú könyvet a készletünkön. 😞 Esetleg próbáld meg más kulcsszóval!"
            # -------------------------------------------------------

            for match in results['matches']:
                # Opcionális: Szűrhetünk Score alapján is (pl. 0.7 alatt kuka)
                if match['score'] < 0.35: # Ha nagyon gyenge a találat, eldobjuk
                    continue

                meta = match['metadata']
                title = str(meta.get('title', 'N/A'))
                
                # Duplikáció szűrés
                is_dup = False
                for seen in seen_titles:
                    if difflib.SequenceMatcher(None, title.lower(), seen.lower()).ratio() > 0.85:
                        is_dup = True; break
                if is_dup: continue
                seen_titles.append(title)
                
                context_text += f"- [CÍM: {title}](URL: {meta.get('url')}) - ÁR: {meta.get('price')} RON\n"
                count += 1
                has_results = True
                if count >= 6: break
            
            # --- BIZTONSÁGI ZÁR 2: Ha volt találat, de mindet kiszűrtük (duplikáció vagy gyenge score miatt) ---
            if not has_results:
                 if detected_lang == 'ro':
                    return "Din păcate, nu am găsit nicio carte relevantă în stoc. 😞"
                 else:
                    return "Sajnos nem találtam releváns könyvet a készleten. 😞"
            # --------------------------------------------------------------------------------------------------

        else:
            # INFO MÓD (Itt engedjük az AI-t beszélni a Policy alapján)
            context_text = "HASZNÁLD A TUDÁSBÁZIST!"

        # 3. Válasz generálás
        if detected_lang == 'ro':
            lang_instruction = "IMPORTANT: Reply in ROMANIAN only! STRICTLY NO HALLUCINATIONS. Only recommend books from the Context list."
        else:
            lang_instruction = "IMPORTANT: Reply in HUNGARIAN only! STRICTLY NO HALLUCINATIONS. Only recommend books from the Context list."

        response = self.client_ai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "system", "content": lang_instruction},
                {"role": "user", "content": f"User Question: {user_input}\n\nContext:\n{context_text}"}
            ],
            temperature=0.3 # Visszavettem a kreativitásból (0.5 -> 0.3)
        )
        return response.choices[0].message.content

# Példányosítjuk az agyat
bot = BooksyBrain()

# --- API VÉGPONTOK ---
@app.get("/")
def home():
    return {"status": "Booksy AI Server is Running", "version": "1.2 Secure"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    user_message = request.message
    if not user_message:
        raise HTTPException(status_code=400, detail="Üres üzenet")
    
    response_text = bot.process_message(user_message)
    return {"reply": response_text}