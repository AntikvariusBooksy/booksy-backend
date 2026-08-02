import os, time, json
import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import anthropic
from openai import OpenAI
from dotenv import load_dotenv

from database import DBHandler, log_event, get_store_policies, ADMIN_EMAILS

load_dotenv()

# Inicializáljuk a klienseket
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# HIVATALOS 2026-OS CLAUDE 5 API AZONOSÍTÓK
CLAUDE_MODEL = "claude-sonnet-5" 
OPENAI_MODEL = "gpt-4o-mini"

class BooksyProactiveAgent:
    def __init__(self, db: DBHandler):
        self.db = db

    def _intent_routing(self, msg: str) -> dict:
        system_prompt = (
            "Te egy e-kereskedelmi router vagy. Elemezd a bejövő üzenetet. Válaszolj KIZÁRÓLAG JSON formátumban!\n"
            "Lehetséges 'intent' értékek: 'policy' (szállítás, fizetés, contact, tarife, árak, szabályzat, retur), 'search' (konkrét könyv vagy téma keresése), 'general' (egyéb csevegés, üdvözlés).\n"
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
                    
                    # --- GOLYÓÁLLÓ URL KATEGÓRIA SZŰRŐ (CSAK A MEGFELELŐ NYELVŰ KÖNYVEK JÖHETNEK) ---
                    if ui_lang == 'hu':
                        # Magyar felület: Ha a "magyar-nyelvu-konyvek" NINCS az URL-ben, akkor elvetjük!
                        if 'magyar-nyelvu-konyvek' not in url:
                            continue
                    else: 
                        # Román felület: Ha a "carti-in-limba-romana" NINCS az URL-ben, akkor elvetjük!
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
        
        # Szigorú nyelvi elválasztás a Prompt számára
        if ui_lang == "hu":
            lang_instruction = "MAGYARUL (Hungarian)"
            persona_style = "Művelt, tapasztalt, rendkívül segítőkész antikvárius szakértő vagy."
            context_text = "Nem találtam megfelelő könyvet a raktárban."
        else:
            lang_instruction = "ROMÂNĂ (Romanian - în limba română)"
            persona_style = "Ești un anticar expert, cultivat, pasionat de cărți și foarte amabil."
            context_text = "Nu am găsit cărți potrivite în stoc."

        # Termék adatok előkészítése
        if intent_data.get('intent') == 'search' and products:
            if ui_lang == "hu":
                context_text = "\n".join([f"Könyv: {p['title']} - {p.get('author','')} - Ár: {p.get('price','')}. Infó: {p.get('text_preview','')}" for p in products])
            else:
                context_text = "\n".join([f"Titlu: {p['title']} - Autor: {p.get('author','')} - Preț: {p.get('price','')}. Descriere: {p.get('text_preview','')}" for p in products])

        if user_mode == "vadasz":
            mode_instruction = "A látogató céltudatos (vadász). Légy lényegretörő, fókuszálj az árakra și a tényekre!" if ui_lang == "hu" else "Vizitatorul este hotărât. Fii precis, axează-te pe preț și fapte!"
        else:
            mode_instruction = "A látogató böngészik (felfedező). Adj kulturális kontextust, mesélj a könyvek hangulatáról!" if ui_lang == "hu" else "Vizitatorul explorează. Oferă context cultural, povestește despre atmosfera cărților!"

        system_prompt = (
            f"Te Booksy vagy, az antikvarius.ro prémium antikváriumának szaktanácsadója. {persona_style}\n"
            f"Vásárlói profil: {mode_instruction}\n\n"
            f"SÉRTHETETLEN SZABÁLYOK:\n"
            f"1. A szállítási díj zónánként FIX! SOHA NICS INGYENES SZÁLLÍTÁS!\n"
            f"2. Utánvétes fizetés KIZÁRÓLAG Románián belül lehetséges!\n"
            f"3. A VÁLASZT KÖTELEZŐEN ÉS KIZÁRÓLAG {lang_instruction} FOGALMAZD MEG!\n"
            f"4. Formázás: ZÉRÓ HTML címke! Csak markdown (félkövér, listák).\n"
        )

        # Ha a szándék szabályzat (policy), betöltjük a memóriából a lementett pontos cégadatokat
        if intent_data.get('intent') == 'policy' and not is_proactive:
            system_prompt += (
                f"\n\nAZ ANTIKVARIUS.RO HIVATALOS SZABÁLYZATA (Fizetés, Szállítás, Kapcsolat):\n<policy>\n{trigger_context}\n</policy>\n"
                f"Ezek a hivatalos informații. Használd ezeket a válaszodban! Légy pontos a díjakkal și időtartamokkal kapcsolatban!"
            )

        user_content = f"Üzenet / Message: '{user_msg}'\n\nTalálatok / Results:\n{context_text}"

        try:
            if is_proactive:
                system_prompt += (
                    f"\nFIGYELEM: Ez egy PROAKTÍV megszólítás. A helyzet: {trigger_context}. "
                    f"Légy nagyon rövid (max 2-3 mondat), természetes, udvarias, dar ne légy tolakodó! "
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
                # NORMÁL CHAT - CLAUDE ELSŐDLEGES MOTOR (CLAUDE 5)
                try:
                    res = claude_client.messages.create(
                        model=CLAUDE_MODEL, 
                        max_tokens=1000, 
                        system=system_prompt, 
                        messages=[{"role": "user", "content": user_content}]
                    )
                    
                    # Biztonságos szövegkinyerés: Kiszűrjük a "ThinkingBlock" (gondolkodás) elemeket!
                    final_text = ""
                    for block in res.content:
                        if getattr(block, 'type', '') == 'text':
                            final_text += block.text
                            
                    return final_text.strip()
                except Exception as claude_err:
                    log_event(f"⚠️ Claude modell hiba, átkapcsolás GPT-4o-mini-re: {claude_err}")
                    # B-TERV: Ha a Claude elszáll, a GPT azonnal átveszi a munkát, így nincs fagyás!
                    res_fallback = openai_client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        max_tokens=1000,
                        temperature=0.7
                    )
                    return res_fallback.choices[0].message.content.strip()
                
        except Exception as e:
            log_event(f"⚠️ Válaszgenerálási Hiba (Mindkét modell elszállt): {e}")
            return "Eroare tehnică. Te rog încearcă mai târziu." if ui_lang == "ro" else "Sajnos technikai hiba történt. Kérlek, próbáld újra később!"

    def process_chat(self, msg: str, ui_lang: str = "ro", user_mode: str = "felfedezo") -> dict:
        intent_data = self._intent_routing(msg)
        final_products = []
        policy_context = ""
        
        if intent_data['intent'] == 'search':
            final_products = self._vector_search(intent_data.get('expanded_query', msg), limit=4, ui_lang=ui_lang)
        elif intent_data['intent'] == 'policy':
            # Beolvassuk a memóriából az éjszaka/indításkor lekapart pontos szabályzatot
            policy_context = get_store_policies()
            
        reply_text = self._generate_response(
            msg, intent_data, final_products, is_proactive=False, trigger_context=policy_context, ui_lang=ui_lang, user_mode=user_mode
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

class BooksyAnalyticsReporter:
    def __init__(self, analytics_db):
        self.db = analytics_db

    def generate_and_send_daily_report(self):
        try:
            # 1. Lekérdezzük az előző napi logokat
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            logs = self.db.get_logs_for_date(yesterday)
            
            # Ha ma teszteljük és tegnap nem volt log, akkor a mai napot is belerakjuk a teszt kedvéért
            if not logs:
                today = datetime.now().strftime("%Y-%m-%d")
                logs = self.db.get_logs_for_date(today)
                yesterday = f"{today} (Mai teszt adatok)"
                
            if not logs:
                log_event("Nincs elég adat az analitikához, e-mail küldés kihagyva.")
                return {"status": "skipped", "message": "Nincs log"}

            log_summary = f"Dátum: {yesterday}\nÖsszes interakció: {len(logs)}\n\n"
            for log in logs:
                log_summary += f"- Esemény: {log.get('trigger_type', 'manual')} | Nyelv: {log.get('ui_language', 'ro')} | Üzenet: '{log.get('user_msg', '')[:100]}' | Nincs találat flag: {log.get('zero_match_flag', False)}\n"
            
            # 2. AI Elemzés (GPT-4o-mini)
            system_prompt = (
                "Te egy vezetői adatelemző vagy az Antikvarius.ro-nál. Elemezd a webáruház előző napi chat logjait. "
                "Készíts egy profi, vizuálisan vonzó, mobilbarát HTML e-mail jelentést magyar nyelven! "
                "Tartalmazzon: 1. Napi összefoglalót (interakciók száma). 2. Miket kerestek a legtöbbet. "
                "3. 'Nincs találat' (zero-match) elemzést - mik azok a könyvek/témák, amiket kerestek, de nincsenek. "
                "4. Javaslatokat beszerzésre vagy UX javításra. "
                "KIZÁRÓLAG érvényes HTML kódot adj vissza <html> és <body> tagekkel, inline CSS formázással, "
                "sötétkék/arany színvilággal. Ne tegyél markdown ```html blokkokat a kimenetbe!"
            )
            
            res = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Itt vannak a napi logok:\n{log_summary}"}
                ],
                max_tokens=2500,
                temperature=0.5
            )
            
            html_content = res.choices[0].message.content.strip()
            if html_content.startswith("```html"):
                html_content = html_content[7:-3].strip()
            
            # 3. Mentsük el az adatbázisba a riportot
            self.db.save_report("daily_analytics", yesterday, html_content)
            
            # 4. E-mail küldés
            sender = os.getenv("SMTP_SENDER")
            password = os.getenv("SMTP_PASSWORD")
            server_url = os.getenv("SMTP_SERVER", "mail.antikvarius.ro")
            
            if not sender or not password:
                log_event("⚠️ Nincs SMTP beállítva, az analitikai e-mail küldés elmaradt.")
                return {"status": "error", "message": "Nincs SMTP beállítva"}
                
            server = smtplib.SMTP(server_url, 26, timeout=15)
            server.starttls()
            server.login(sender, password)
            
            for admin in ADMIN_EMAILS:
                msg = MIMEMultipart()
                msg['Subject'] = f"📊 Booksy AI Napi Analitika ({yesterday})"
                msg['From'] = sender
                msg['To'] = admin
                msg.attach(MIMEText(html_content, 'html'))
                server.send_message(msg)
                
            server.quit()
            log_event(f"✅ Napi AI Analitika sikeresen elküldve a vezetőségnek ({yesterday}).")
            return {"status": "success", "message": "Jelentés elküldve"}
            
        except Exception as e:
            log_event(f"❌ Napi Analitika Hiba: {e}")
            return {"status": "error", "message": str(e)}