import os, time, json
from google import genai
from google.genai import types
import anthropic 
from openai import OpenAI
from dotenv import load_dotenv

from database import DBHandler, log_event, STORE_POLICIES_FILE

load_dotenv()

# Inicializáljuk a klienseket
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# A lassú, de okos modell a beszélgetéshez, és a villámgyors a felugró ablakokhoz
CLAUDE_MODEL = "claude-3-5-sonnet-latest" 
OPENAI_MODEL = "gpt-4o-mini"

class BooksyProactiveAgent:
    def __init__(self, db: DBHandler):
        self.db = db

    def _get_policies(self):
        if os.path.exists(STORE_POLICIES_FILE):
            with open(STORE_POLICIES_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("policies", "")
        return "Céges szabályzat nem elérhető."

    def _intent_routing(self, msg: str) -> dict:
        system_prompt = (
            "Te egy e-kereskedelmi router vagy. Elemezd a bejövő üzenetet. Válaszolj KIZÁRÓLAG JSON formátumban!\n"
            "Lehetséges 'intent' értékek: 'policy' (szállítás, fizetés, kapcsolat, árak, cégadatok), 'search' (konkrét könyv vagy téma keresése), 'general' (egyéb csevegés).\n"
            "Ha 'search', akkor generálj egy 'expanded_query' mezőt, ami 3-5 szemantikus kulcsszóval bővíti a keresést.\n"
            "Minta JSON: {\"intent\": \"search\", \"expanded_query\": \"eredeti szó, szinonima1, szinonima2\"}"
        )
        try:
            if not msg:
                return {"intent": "search", "expanded_query": ""}
                
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": msg}
                ],
                response_format={ "type": "json_object" },
                max_tokens=150
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            log_event(f"⚠️ Intent Routing Hiba: {e}")
            return {"intent": "search", "expanded_query": msg} 

    def _vector_search(self, query: str, limit: int = 4, ui_lang: str = "ro") -> list:
        if not query:
            return []
            
        try:
            vec_req = gemini_client.models.embed_content(
                model="gemini-embedding-001", 
                contents=query, 
                config=types.EmbedContentConfig(output_dimensionality=768)
            )
            vec = vec_req.embeddings[0].values
            
            # Kérünk rengeteg találatot (20 db), hogy a szigorú nyelvi szűrés után is maradjon könyv
            db_res = self.db.collection.query(
                query_embeddings=[vec], 
                n_results=20, 
                where={"$and": [{"stock": "instock"}, {"type": "book"}]}
            )
            
            if db_res['ids'] and db_res['ids'][0]:
                products = db_res['metadatas'][0]
                seen_titles = set()
                final_products = []
                
                for p in products:
                    if len(final_products) >= limit:
                        break
                        
                    url = p.get('url', '').lower()
                    
                    # --- GOLYÓÁLLÓ NYELVI ÉS KATEGÓRIA URL SZŰRŐ ---
                    if ui_lang == 'hu':
                        # Ha a látogató a magyar felületen van, TILOS román kategóriát ajánlani!
                        if 'carti-in-limba-romana' in url: 
                            continue 
                        # Szigorítva: Csak akkor fogadja el, ha benne van a magyar slug vagy a /hu/ tag
                        if 'magyar-nyelvu-konyvek' not in url and '/hu/' not in url:
                            continue
                    else: 
                        # Ha a látogató a román felületen van, TILOS magyar könyvet / kategóriát ajánlani!
                        if 'magyar-nyelvu-konyvek' in url or '/hu/' in url: 
                            continue 
                        # Szigorítva: Csak a román kategóriát fogadja el
                        if 'carti-in-limba-romana' not in url:
                            continue

                    if 'image_url' in p and 'image' not in p: p['image'] = p['image_url']
                    clean_title = p.get('title', '').strip().lower()
                    
                    if clean_title not in seen_titles:
                        seen_titles.add(clean_title)
                        final_products.append(p)
                return final_products
            return []
        except Exception as e:
            log_event(f"⚠️ Vektor Keresés Hiba: {e}")
            return []

    def _generate_response(self, user_msg: str, intent_data: dict, products: list, is_proactive: bool = False, trigger_context: str = "", ui_lang: str = "ro", user_mode: str = "felfedezo") -> str:
        policy_text = self._get_policies()
        
        # Szigorú nyelvi elválasztás a Prompt számára
        if ui_lang == "hu":
            lang_instruction = "MAGYARUL (Hungarian)"
            persona_style = "Művelt, tapasztalt, rendkívül segítőkész antikvárius szakértő vagy."
            context_text = "Nem találtam megfelelő könyvet a raktárban."
            if products:
                context_text = "\n".join([f"Könyv: {p['title']} - {p.get('author','')} - Ár: {p.get('price','')}. Infó: {p.get('text_preview','')}" for p in products])
        else:
            lang_instruction = "ROMÂNĂ (Romanian - în limba română)"
            persona_style = "Ești un anticar expert, cultivat, pasionat de cărți și foarte amabil."
            context_text = "Nu am găsit cărți potrivite în stoc."
            if products:
                context_text = "\n".join([f"Titlu: {p['title']} - Autor: {p.get('author','')} - Preț: {p.get('price','')}. Descriere: {p.get('text_preview','')}" for p in products])

        if user_mode == "vadasz":
            mode_instruction = "A látogató céltudatos (vadász). Légy lényegretörő, pontos, fókuszálj az árakra és a raktárkészletre!" if ui_lang == "hu" else "Vizitatorul este hotărât. Fii precis, axează-te pe preț și stoc!"
        else:
            mode_instruction = "A látogató böngészik (felfedező). Adj kulturális kontextust, mesélj a könyvek hangulatáról!" if ui_lang == "hu" else "Vizitatorul explorează. Oferă context cultural, povestește despre atmosfera cărților!"

        system_prompt = (
            f"Te Booksy vagy, az antikvarius.ro prémium antikváriumának szaktanácsadója. {persona_style}\n"
            f"Vásárlói profil: {mode_instruction}\n\n"
            f"Céges tudásbázisod:\n<company_policies>\n{policy_text}\n</company_policies>\n\n"
            f"SÉRTHETETLEN SZABÁLYOK:\n"
            f"1. A szállítási díj zónánként FIX! SOHA NICS INGYENES SZÁLLÍTÁS!\n"
            f"2. Utánvétes fizetés KIZÁRÓLAG Románián belül lehetséges!\n"
            f"3. A VÁLASZT KÖTELEZŐEN ÉS KIZÁRÓLAG {lang_instruction} FOGALMAZD MEG!\n"
            f"4. Formázás: ZÉRÓ HTML címke!\n"
        )

        user_content = f"Üzenet / Message: '{user_msg}'\n\nTalálatok / Results:\n{context_text}"

        try:
            if is_proactive:
                # SEBESSÉG OPTIMALIZÁLÁS: A proaktív megkeresések a villámgyors GPT-4o-mini-n futnak (1-2 másodperc)
                system_prompt += (
                    f"\nFIGYELEM: Ez egy PROAKTÍV megszólítás. A helyzet: {trigger_context}. "
                    f"Légy nagyon rövid (max 2-3 mondat), természetes, udvarias, de ne légy tolakodó! "
                    f"Írj KIZÁRÓLAG {lang_instruction}!"
                )
                res = openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Scrie mesajul de întâmpinare / Írd meg a megszólítást."}
                    ],
                    max_tokens=250,
                    temperature=0.7
                )
                return res.choices[0].message.content.strip()
            else:
                # Normál Chat logikához megmarad a Claude a mély és okos beszélgetéshez
                res = claude_client.messages.create(
                    model=CLAUDE_MODEL, 
                    max_tokens=1000, 
                    system=system_prompt, 
                    messages=[{"role": "user", "content": user_content}]
                )
                return res.content[0].text.strip()
                
        except Exception as e:
            log_event(f"⚠️ Válaszgenerálási Hiba: {e}")
            return "Eroare tehnică. Te rog încearcă mai târziu." if ui_lang == "ro" else "Sajnos technikai hiba történt. Kérlek, próbáld újra később!"

    def process_chat(self, msg: str, ui_lang: str = "ro", user_mode: str = "felfedezo") -> dict:
        intent_data = self._intent_routing(msg)
        final_products = []
        if intent_data['intent'] == 'search':
            final_products = self._vector_search(intent_data.get('expanded_query', msg), limit=4, ui_lang=ui_lang)
        
        reply_text = self._generate_response(
            msg, intent_data, final_products, is_proactive=False, ui_lang=ui_lang, user_mode=user_mode
        )
        return {
            "reply": reply_text, 
            "products": final_products, 
            "zero_match_flag": (intent_data['intent'] == 'search' and len(final_products) == 0)
        }

    def process_proactive_trigger(self, trigger_type: str, session_data: dict) -> dict:
        trigger_context = ""
        search_query = ""
        ui_lang = session_data.get("ui_lang", "ro")
        user_mode = session_data.get("user_mode", "felfedezo")
        book_title = session_data.get("last_book_title", "")

        # Nyelvi kontextus a modellnek (Itt kapja meg a pontos instrukciót a választott nyelven!)
        if ui_lang == "ro":
            if trigger_type == "cart_abandonment":
                trigger_context = f"Atenție: clientul este în coș și vrea să plece. Amintește-i politicos că taxa de livrare este fixă. Ultima carte vizionată: '{book_title}'."
                search_query = book_title or "clasic"
            elif trigger_type == "product_exit_intent":
                trigger_context = f"Clientul părăsește pagina produsului '{book_title}'. Atrage-i atenția politicos că exemplarele noastre anticare sunt unice și se vând repede."
                search_query = book_title or "raritate"
            elif trigger_type == "general_exit_intent":
                trigger_context = "Clientul părăsește prima pagină. Salută-l scurt și oferă-i ajutorul tău de anticar."
                search_query = "beletristică"
            elif trigger_type == "zero_match_search":
                search_query = session_data.get("failed_search_term", "")
                trigger_context = f"Căutarea a eșuat pentru: '{search_query}'. Oferă-i cărți similare din stoc."
            elif trigger_type == "checkout_hesitation":
                 trigger_context = "Clientul ezită la checkout. Amintește de plata ramburs (doar în RO)."
                 search_query = ""
        else: # hu
            if trigger_type == "cart_abandonment":
                trigger_context = f"A látogató a kosár oldalon van, de el akarja hagyni az oldalt. Emlékeztesd, hogy a szállítási díj fix, így érdemes telepakolni a dobozt. Utolsó könyv: '{book_title}'."
                search_query = book_title or "klasszikus"
            elif trigger_type == "product_exit_intent":
                trigger_context = f"A látogató kilépne a '{book_title}' oldaláról. Hívd fel a figyelmét az egyedi példányokra, és hogy gyorsan elkelnek."
                search_query = book_title or "ritkaság"
            elif trigger_type == "general_exit_intent":
                trigger_context = "A látogató kilép a főoldalról. Köszöntsd röviden, és ajánld a segítséged."
                search_query = "klasszikus"
            elif trigger_type == "zero_match_search":
                search_query = session_data.get("failed_search_term", "")
                trigger_context = f"A kereső nem adott találatot erre: '{search_query}'. Segíts neki hasonló kötetekkel."
            elif trigger_type == "checkout_hesitation":
                 trigger_context = "A látogató a pénztárnál elakadt. Emlékeztesd az utánvétes fizetési lehetőségre Románián belül."
                 search_query = ""

        final_products = []
        if search_query:
            # Rögtön keresünk, kihagyva az intent_routing lassítását proaktív módban!
            final_products = self._vector_search(search_query, limit=3, ui_lang=ui_lang)
            
        reply_text = self._generate_response(
            user_msg="", 
            intent_data={"intent": "proactive"}, 
            products=final_products, 
            is_proactive=True, 
            trigger_context=trigger_context,
            ui_lang=ui_lang,
            user_mode=user_mode
        )
        
        return {
            "reply": reply_text,
            "products": final_products,
            "trigger_handled": True
        }