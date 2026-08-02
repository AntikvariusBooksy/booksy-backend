import os
import time
import json
import requests
import smtplib
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formatdate, make_msgid
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

# API AZONOSÍTÓK - 2026
CLAUDE_MODEL = "claude-sonnet-5" 
OPENAI_MODEL = "gpt-4o-mini"
# A Google hivatalos 2026-os API frissítése alapján az új modell:
GEMINI_MODEL = "gemini-3.6-flash"

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
                system_prompt += f"\nFIGYELEM: Ez Ez egy PROAKTÍV megszólítás. A helyzet: {trigger_context}. Légy nagyon rövid (max 2-3 mondat)!"
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
                        elif hasattr(block, 'text'):
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

    def get_real_time_market_trends(self):
        """Gemini-vel lekéri az aktuális Google trendeket. Interactions API használata, golyóálló attribútumkereséssel."""
        log_event(f"🔍 Valós idejű webes trendelemzés indítása a Gemini-vel (Modell: {GEMINI_MODEL})...")
        try:
            prompt = (
                "Készíts egy nagyon rövid, 3 pontos webes kutatást az aktuális romániai és erdélyi "
                "könyveladási trendekről. Fókuszálj arra, hogy milyen könyveket és írókat keresnek a legtöbben."
            )
            
            interaction = gemini_client.interactions.create(
                model=GEMINI_MODEL,
                input=prompt,
                tools=[{"type": "google_search"}],
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 1500
                }
            )
            
            # Hivatalos dokumentáció szerinti 2026-os API kinyerés a Gemini-től
            if hasattr(interaction, 'output_text') and interaction.output_text:
                log_event("✅ Valós idejű trendadatok sikeresen lekérve.")
                return str(interaction.output_text)
            elif hasattr(interaction, 'text') and interaction.text:
                log_event("✅ Valós idejű trendadatok lekérve (fallback text attribútumon).")
                return str(interaction.text)
            else:
                raw_str = str(interaction)
                if len(raw_str) > 20:
                    log_event("✅ Valós idejű trendadatok kinyerve string castolással.")
                    return raw_str
                
                log_event("⚠️ A Gemini API nem adott vissza értelmezhető adatot.")
                return "Jelenleg nem állnak rendelkezésre valós idejű trendadatok a webről."
                
        except Exception as e:
            log_event(f"⚠️ Hiba a valós idejű trendek lekérésekor: {e}")
            return (
                "INTERNETES ADAT NEM ELÉRHETŐ. Kérlek, KIZÁRÓLAG a mellékelt belső "
                "logokból és a saját tudásodból vonj le következtetéseket az esetleges trendekről."
            )

    def generate_and_send_daily_report(self):
        try:
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            logs = self.db.get_logs_for_date(yesterday)
            
            if not logs: 
                log_event("Nincs log tegnapról, analitika kihagyva.")
                return
            
            log_summary = f"Dátum: {yesterday}\nÖsszesen {len(logs)} interakció.\n{json.dumps(logs, indent=2)}"
            
            # 1. Trendek lekérése a Google-től a javított API-val
            real_time_trends = self.get_real_time_market_trends()

            # 2. Claude Prompt - Tiszta Szöveges (Markdown) Kényszerítés Dedikált Struktúrával
            system_prompt = (
                "Te egy Vezetői Adatelemző vagy az Antikvarius.ro-nál.\n\n"
                f"PIACI KONTEXTUS / TRENDEK (A Webről):\n{real_time_trends}\n\n"
                "SÉRTHETETLEN SZABÁLYOK:\n"
                "1. Készíts egy professzionális, jól tagolt VEZETŐI JELENTÉST tiszta Markdown formátumban!\n"
                "2. SOHA NE HASZNÁLJ HTML TAGEKET! (Semmilyen <html>, <div>, stb. címkét ne írj bele).\n"
                "3. KÖTELEZŐ SZINTÉZIS: Vesd össze a PIACI KONTEXTUS-t a BELSŐ LOGOKKAL! Keresd meg, hogy a weben pörgő trendek megjelennek-e a saját kereséseink között, és ha nem, emeld ki ezt mint üzleti hiányosságot (opportunity gap)!\n\n"
                "A jelentés KÖTELEZŐ szerkezete (használj '# ' címsorokat):\n"
                "# 1. Piaci Környezet és Webes Trendek (A kapott webről származó adatok összefoglalása)\n"
                "# 2. Átfogó Összefoglaló (Belső logok: Interakciók, arányok, eszközök, hibák)\n"
                "# 3. Keresési Elemzés és Szintézis (Zero-match elemzés, hiányosságok, ÉS a webes trendek összevetése a belső keresésekkel)\n"
                "# 4. UX és Vásárlói Súrlódások (Elakadások, technikai hibák, kosárelhagyások)\n"
                "# 5. Vezetői Stratégia (Konkrét beszerzési, technikai és marketing javaslatok a belső adatok és a külső trendek alapján)\n"
            )
            
            log_event("🧠 Claude elemző processz indítása (Tiszta Szöveg Mód, Megnövelt Token Limittel)...")
            res = claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=8000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"Elemezd a következő logokat és készítsd el a Markdown jelentést:\n\n{log_summary}"}
                ]
            )
            
            # Golyóálló kinyerés
            raw_content = ""
            if hasattr(res, 'content'):
                for block in res.content:
                    if hasattr(block, 'type') and block.type == 'text' and hasattr(block, 'text'):
                        raw_content += block.text
                    elif hasattr(block, 'text') and not hasattr(block, 'type'):
                        raw_content += block.text
            
            # Végső fallback, ha a Claude teljesen kifordult magából
            if not raw_content:
                raw_content = str(res)
                
            raw_content = raw_content.strip()
            
            # Tisztítás a Python string reprezentációk ellen (Ha mégis nyers Message object maradt volna)
            raw_content = re.sub(r"^Message\(.*?content=\[TextBlock\(text='", "", raw_content)
            raw_content = re.sub(r"^\[TextBlock\(text='", "", raw_content)
            raw_content = re.sub(r"', type='text'\)\]\)$", "", raw_content)
            raw_content = re.sub(r"', type='text'\)\]$", "", raw_content)
            raw_content = raw_content.replace('\\n', '\n')
            
            # Fallback ellenőrzés és vészhelyzeti e-mail
            if len(raw_content) < 200:
                log_event(f"⚠️ A Claude válasza gyanúsan rövid ({len(raw_content)} karakter). Vészhelyzeti HTML generálása.")
                final_email_html = f"""<!DOCTYPE html>
                <html lang="hu">
                <head><meta charset="UTF-8"></head>
                <body style="font-family: Arial, sans-serif; background-color: #fce4e4; padding: 20px;">
                    <div style="background: white; padding: 30px; border-left: 10px solid #d9534f; border-radius: 8px;">
                        <h2 style="color: #d9534f;">⚠️ Rendszer Figyelmeztetés: AI Elemzési Hiba</h2>
                        <p>Az elemző modul a várt HTML riport helyett a következő váratlan és rövid választ adta:</p>
                        <div style="background: #f5f5f5; padding: 15px; border: 1px solid #ddd; font-family: monospace;">
                            {raw_content}
                        </div>
                        <p style="margin-top: 20px;"><strong>Napi feldolgozott interakciók száma:</strong> {len(logs)} db</p>
                        <p style="font-size: 11px; color: #888;">Ezt az e-mailt a Booksy AI biztonsági modulja generálta, hogy megelőzze az üres e-mailek kiküldését.</p>
                    </div>
                </body>
                </html>"""
            else:
                # 3. Python Konverzió (Markdown -> Szép HTML)
                html_content = raw_content.replace('\n', '<br>')
                html_content = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color: #333;">\1</strong>', html_content)
                html_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html_content)
                html_content = re.sub(r'# (.*?)<br>', r'<h2 style="color: #0b57d0; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px; margin-top: 25px;">\1</h2>', html_content)
                html_content = re.sub(r'## (.*?)<br>', r'<h3 style="color: #1a73e8; margin-top: 20px;">\1</h3>', html_content)
                
                final_email_html = f"""<!DOCTYPE html>
                <html lang="hu">
                <head><meta charset="UTF-8"></head>
                <body style="font-family: 'Segoe UI', Helvetica, Arial, sans-serif; background-color: #f4f7f6; padding: 20px; color: #444; line-height: 1.6;">
                    <div style="background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); max-width: 800px; margin: auto;">
                        <div style="text-align: center; margin-bottom: 30px;">
                            <h1 style="color: #0b57d0; margin-bottom: 5px;">📊 Vezetői AI Analitika</h1>
                            <span style="background: #e8f0fe; color: #1a73e8; padding: 5px 15px; border-radius: 20px; font-weight: bold; font-size: 14px;">{yesterday}</span>
                        </div>
                        <div style="font-size: 15px;">
                            {html_content}
                        </div>
                        <hr style="border: 0; border-top: 1px solid #eee; margin: 40px 0 20px 0;">
                        <p style="font-size: 12px; color: #999; text-align: center;">Ezt a jelentést a Booksy AI Generálta tiszta szöveges (Markdown) alapon.</p>
                    </div>
                </body>
                </html>"""

            log_event(f"✅ Riport generálva. Kinyert szöveg hossza: {len(raw_content)} karakter.")
            self.db.save_report("daily_analytics", yesterday, final_email_html)
            
            smtp_server = os.getenv("SMTP_SERVER")
            smtp_sender = os.getenv("SMTP_SENDER")
            smtp_pass = os.getenv("SMTP_PASSWORD")
            
            if not all([smtp_server, smtp_sender, smtp_pass]):
                log_event("❌ Hiba: Hiányzó SMTP beállítások, analitika e-mail nem lett elküldve.")
                return

            try:
                # Multipart mód a tökéletes kódolásért
                msg = MIMEMultipart('alternative')
                msg['Subject'] = f"📊 Antikvarius.ro Vezetői AI Analitika - {yesterday}"
                msg['From'] = smtp_sender
                msg['To'] = ", ".join(ADMIN_EMAILS)
                msg['Date'] = formatdate(localtime=True)
                msg['Message-ID'] = make_msgid()
                
                text_part = MIMEText(raw_content, 'plain', 'utf-8')
                msg.attach(text_part)
                
                html_part = MIMEText(final_email_html, 'html', 'utf-8')
                msg.attach(html_part)
                
                port = int(os.getenv("SMTP_PORT", 587))
                with smtplib.SMTP(smtp_server, port, timeout=20) as server:
                    server.starttls()
                    server.login(smtp_sender, smtp_pass)
                    server.sendmail(smtp_sender, ADMIN_EMAILS, msg.as_string())
                
                log_event("✅ Napi AI Analitika sikeresen elküldve a vezetőségnek.")
            except Exception as smtp_err:
                log_event(f"❌ SMTP Kapcsolódási/Küldési Hiba: {smtp_err}")

        except Exception as e:
            log_event(f"❌ Napi Analitika Generálási Hiba: {e}")