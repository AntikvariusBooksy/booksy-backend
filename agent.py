import os
import time
import json
import requests
import smtplib
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import anthropic
from openai import OpenAI
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Környezeti változók betöltése
load_dotenv()

# Kliensek inicializálása
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# API AZONOSÍTÓK - 2026 Claude frissítés
CLAUDE_MODEL = "claude-sonnet-5" 
OPENAI_MODEL = "gpt-4o-mini"

from database import DBHandler, log_event, get_store_policies, ADMIN_EMAILS

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
                    
                    if ui_lang == 'hu':
                        if 'magyar-nyelvu-konyvek' not in url: continue
                    else: 
                        if 'carti-in-limba-romana' not in url: continue

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
        
        if ui_lang == "hu":
            lang_instruction = "MAGYARUL (Hungarian)"
            persona_style = "Művelt, tapasztalt, rendkívül segítőkész antikvárius szakértő vagy."
            context_text = "Nem találtam megfelelő könyvet a raktárban."
        else:
            lang_instruction = "ROMÂNĂ (Romanian - în limba română)"
            persona_style = "Ești un anticar expert, cultivat, pasionat de cărți și foarte amabil."
            context_text = "Nu am găsit cărți potrivite în stoc."

        if intent_data.get('intent') == 'search' and products:
            if ui_lang == "hu":
                context_text = "\n".join([f"Könyv: {p['title']} - {p.get('author','')} - Ár: {p.get('price','')}. Infó: {p.get('text_preview','')}" for p in products])
            else:
                context_text = "\n".join([f"Titlu: {p['title']} - Autor: {p.get('author','')} - Preț: {p.get('price','')}. Descriere: {p.get('text_preview','')}" for p in products])

        if user_mode == "vadasz":
            mode_instruction = "A látogató céltudatos (vadász). Légy lényegretörő, fókuszálj az árakra!" if ui_lang == "hu" else "Vizitatorul este hotărât. Fii precis, axează-te pe preț!"
        else:
            mode_instruction = "A látogató böngészik (felfedező). Adj kulturális kontextust!" if ui_lang == "hu" else "Vizitatorul explorează. Oferă context cultural!"

        system_prompt = (
            f"Te Booksy vagy, az antikvarius.ro prémium antikváriumának szaktanácsadója. {persona_style}\n"
            f"Vásárlói profil: {mode_instruction}\n\n"
            f"SÉRTHETETLEN SZABÁLYOK:\n"
            f"1. A szállítási díj zónánként FIX! SOHA NICS INGYENES SZÁLLÍTÁS!\n"
            f"2. Utánvétes fizetés KIZÁRÓLAG Románián belül lehetséges!\n"
            f"3. A VÁLASZT KÖTELEZŐEN ÉS KIZÁRÓLAG {lang_instruction} FOGALMAZD MEG!\n"
            f"4. Formázás: ZÉRÓ HTML címke! Csak markdown (félkövér, listák).\n"
        )

        if intent_data.get('intent') == 'policy' and not is_proactive:
            system_prompt += f"\n\nAZ ANTIKVARIUS.RO HIVATALOS SZABÁLYZATA:\n<policy>\n{trigger_context}\n</policy>\nHasználd ezeket az információkat!"

        user_content = f"Üzenet / Message: '{user_msg}'\n\nTalálatok / Results:\n{context_text}"

        try:
            if is_proactive:
                system_prompt += f"\nFIGYELEM: Ez egy PROAKTÍV megszólítás. A helyzet: {trigger_context}. Légy nagyon rövid (max 2-3 mondat)!"
                res = openai_client.chat.completions.create(
                    model=OPENAI_MODEL, 
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": "Scrie mesajul."}], 
                    max_tokens=250
                )
                return res.choices[0].message.content.strip()
            else:
                try:
                    res = claude_client.messages.create(
                        model=CLAUDE_MODEL, 
                        max_tokens=1000, 
                        system=system_prompt, 
                        messages=[{"role": "user", "content": user_content}]
                    )
                    final_text = ""
                    for block in res.content:
                        if getattr(block, 'type', '') == 'text':
                            final_text += block.text
                    return final_text.strip()
                except Exception as e:
                    log_event(f"⚠️ Claude hiba, átkapcsolás GPT-4o-mini-re: {e}")
                    res_f = openai_client.chat.completions.create(
                        model=OPENAI_MODEL, 
                        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}], 
                        max_tokens=1000
                    )
                    return res_f.choices[0].message.content.strip()
        except Exception as e:
            log_event(f"❌ Hiba: {e}")
            return "Eroare tehnică."

    def process_chat(self, msg: str, ui_lang: str = "ro", user_mode: str = "felfedezo") -> dict:
        intent_data = self._intent_routing(msg)
        final_products = []
        policy_context = ""
        if intent_data['intent'] == 'search':
            final_products = self._vector_search(intent_data.get('expanded_query', msg), limit=4, ui_lang=ui_lang)
        elif intent_data['intent'] == 'policy':
            policy_context = get_store_policies()
        reply_text = self._generate_response(msg, intent_data, final_products, is_proactive=False, trigger_context=policy_context, ui_lang=ui_lang, user_mode=user_mode)
        return {"reply": reply_text, "products": final_products, "zero_match_flag": (intent_data['intent'] == 'search' and len(final_products) == 0)}

    def process_proactive_trigger(self, trigger_type: str, session_data: dict) -> dict:
        trigger_context = ""; search_query = ""
        ui_lang = session_data.get("ui_lang", "ro")
        
        if trigger_type == 'cart_abandonment':
            trigger_context = "A látogató kilépne a kosárból. Emlékeztesd, hogy a szállítási díj fix, így érdemes még körülnézni!"
        elif trigger_type == 'checkout_hesitation':
            trigger_context = "A látogató elakadt a pénztárnál. Emlékeztesd, hogy Románián belül utánvéttel is fizethet!"
        elif trigger_type == 'zero_match_search':
            trigger_context = f"A látogató keresett valamit ({session_data.get('failed_search_term')}), de nem talált. Ajánlj fel segítséget!"
            
        reply_text = self._generate_response("", {"intent": "proactive"}, [], is_proactive=True, trigger_context=trigger_context, ui_lang=ui_lang)
        return {"reply": reply_text, "products": [], "trigger_handled": True}


class BooksyAnalyticsReporter:
    def __init__(self, analytics_db):
        self.db = analytics_db

    def generate_and_send_daily_report(self):
        try:
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            logs = self.db.get_logs_for_date(yesterday)
            
            if not logs: 
                log_event("Nincs log tegnapról, analitika kihagyva.")
                return
            
            log_summary = f"Dátum: {yesterday}\nÖsszesen {len(logs)} interakció.\n{json.dumps(logs, indent=2)}"
            
            market_context = (
                "2026-os erdélyi és romániai (főleg Marosvásárhely és környéke) könyvpiaci trendek: "
                "A Bookfest 2026 alapján kiemelkedő a kereslet a pszichológiai thrillerek (pl. Freida McFadden, Anders & Anette de la Motte), "
                "valamint a skandináv krimik iránt. Matt Haig 'feel-good' regényei stabilan vezetik az eladásokat. "
                "A helyi, marosvásárhelyi közönség kifejezetten keresi a minőségi magyar nyelvű kortárs irodalmat és a ritkaságokat, "
                "valamint a helytörténeti köteteket. Gyerekkönyveknél a Harry Potter sorozat kötetei hiánycikkek."
            )

            system_prompt = f"""Te az Antikvarius.ro (Marosvásárhely) Vezetői Adatelemzője és E-kereskedelmi Stratégája vagy.
PIACI KONTEXTUS 2026: {market_context}

FELADAT: Elemezd a kapott chatbot logokat, és készíts egy vizuálisan lenyűgöző, vezetői HTML jelentést.

KÖTELEZŐ TARTALOM ÉS STRUKTÚRA:
1. Átfogó Összefoglaló: KPI-ok (Interakciók száma, Nyelvi megoszlás HU/RO, Eszközök). Szöveges értékelés a forgalom minőségéről.
2. 'Nincs találat' (Zero-Match) Elemzés: Pontosan miket kerestek a userek, amit az AI nem talált a raktárban? Listázd a konkrét keresési kifejezéseket! Értékeld a kiesett bevételt.
3. UX és Súrlódási Elemzés: Hányszor kellett a botnak beavatkoznia (cart_abandonment, checkout_hesitation)? Milyen UX hibákra utal ez a weboldalon (pl. szállítási díj nem tiszta)?
4. Beszerzési és Marketing Javaslatok: A "nincs találat" szavak és a 2026-os marosvásárhelyi trendek alapján MILYEN KÖNYVEKET kell sürgősen beszerezni a raktárba?

FORMAI KÖVETELMÉNYEK:
- A válaszod KIZÁRÓLAG egy érvényes, tiszta HTML kód lehet, a <!DOCTYPE html><html>...</html> tagekkel bezárólag.
- NE írj semmilyen bevezető vagy lezáró szöveget a HTML-en kívül!
- Használj beágyazott CSS-t a <style> tagben. A dizájn legyen tiszta: fehér háttér (#ffffff), sötétkék fejlécek (#0b57d0), világosszürke árnyékolt dobozok a KPI-oknak, és jól olvasható fekete szöveg (Arial/Helvetica).
"""
            
            res = claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"Itt vannak a tegnapi logok az elemzéshez:\n{log_summary}"}
                ]
            )
            
            html_content = ""
            for block in res.content:
                if getattr(block, 'type', '') == 'text':
                    html_content += block.text

            # GOLYÓÁLLÓ HTML KINYERÉS ÉS TISZTÍTÁS (Bulletproof parser)
            clean_html = html_content.strip()
            
            # 1. Eltávolítjuk a markdown kódblokkokat, ha a modell mégis használta volna
            if "```html" in clean_html:
                clean_html = clean_html.split("```html")[1].split("```")[0].strip()
            elif "```" in clean_html:
                clean_html = clean_html.split("```")[1].split("```")[0].strip()
            
            # 2. Ellenőrizzük, hogy van-e benne HTML struktúra. Ha nincs, becsomagoljuk.
            if "<html" not in clean_html.lower() and "<body" not in clean_html.lower():
                log_event("⚠️ A Claude nem rakott HTML tageket a válaszba, automatikus becsomagolás indul.")
                clean_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
body {{ font-family: Arial, sans-serif; color: #333; line-height: 1.6; padding: 20px; background-color: #ffffff; }}
h1, h2, h3 {{ color: #0b57d0; }}
.card {{ background: #f8f9fa; border: 1px solid #e0e0e0; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
ul {{ padding-left: 20px; }}
</style>
</head>
<body>
    <h2>Antikvarius.ro Vezetői Elemzés (Automatikus fallback formátum)</h2>
    <div>{clean_html.replace(chr(10), '<br>')}</div>
</body>
</html>"""

            # Ha teljesen üres maradt valami oknál fogva
            if not clean_html:
                raise ValueError("Az AI által generált HTML tartalom teljesen üres lett tisztítás után!")

            log_event(f"✅ Riport generálva. HTML hossza: {len(clean_html)} karakter.")
            self.db.save_report("daily_analytics", yesterday, clean_html)
            
            # TÉNYLEGES E-MAIL KÜLDÉS
            smtp_server = os.getenv("SMTP_SERVER")
            smtp_sender = os.getenv("SMTP_SENDER")
            smtp_pass = os.getenv("SMTP_PASSWORD")
            
            if not all([smtp_server, smtp_sender, smtp_pass]):
                log_event("❌ Hiba: Hiányzó SMTP beállítások, analitika e-mail nem lett elküldve.")
                return

            msg = MIMEMultipart()
            msg['Subject'] = f"📊 Antikvarius.ro Vezetői AI Analitika - {yesterday}"
            msg['From'] = smtp_sender
            msg['To'] = ", ".join(ADMIN_EMAILS)
            msg.attach(MIMEText(clean_html, 'html'))
            
            try:
                port = int(os.getenv("SMTP_PORT", 587))
                with smtplib.SMTP(smtp_server, port, timeout=15) as server:
                    server.starttls()
                    server.login(smtp_sender, smtp_pass)
                    server.send_message(msg)
                
                log_event("✅ Napi AI Analitika sikeresen elküldve a vezetőségnek.")
            except Exception as smtp_err:
                log_event(f"❌ SMTP Kapcsolódási Hiba: {smtp_err}")

        except Exception as e:
            log_event(f"❌ Napi Analitika Generálási Hiba: {e}")