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

gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# API AZONOSÍTÓK - 2026
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
                    # Golyóálló kinyerés: csak a text attribútummal rendelkező blokkokat használjuk fel
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
        """Gemini-vel lekéri az aktuális Google trendeket. Golyóálló hibakezeléssel."""
        log_event("🔍 Valós idejű webes trendelemzés indítása a Gemini-vel...")
        try:
            prompt = (
                "Keress rá a weben a legfrissebb könyveladási és olvasási trendekre Romániában, "
                "kifejezetten fókuszálva Erdélyre és Marosvásárhely környékére, valamint a magyar nyelvű olvasókra. "
                "Milyen műfajok, szerzők vagy könyvek pörögnek most a legjobban a könyvesboltokban vagy a Google keresésekben? "
                "Foglald össze röviden, 3-4 pontban, tényekre támaszkodva."
            )
            
            # Megpróbáljuk a legfrissebb modellek egyikével
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}]
                )
            )
            
            if response and response.text:
                log_event("✅ Valós idejű trendadatok sikeresen lekérve.")
                return response.text
            else:
                raise Exception("Üres választ adott a Gemini.")
                
        except Exception as e:
            # GOLYÓÁLLÓ VÉDELEM: Nem szemeteljük tele a Claude promptját a hibaüzenettel.
            # Ehelyett egy határozott, tiszta instrukciót adunk vissza.
            log_event(f"⚠️ Hiba a valós idejű trendek lekérésekor: {e}")
            return (
                "KÜLSŐ ADAT JELENLEG NEM ELÉRHETŐ. SÉRTHETETLEN SZABÁLY: "
                "Mivel a Google Trends adatok most nem elérhetők, KIZÁRÓLAG az alább átadott 'Belső Logok' alapján vonj le következtetéseket. "
                "Használd a saját általános piaci tudásodat az erdélyi és román könyvpiacról a logokban szereplő keresőszavak értékeléséhez."
            )

    def generate_and_send_daily_report(self):
        try:
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            logs = self.db.get_logs_for_date(yesterday)
            
            if not logs: 
                log_event("Nincs log tegnapról, analitika kihagyva.")
                return
            
            log_summary = f"Dátum: {yesterday}\nÖsszesen {len(logs)} interakció.\n{json.dumps(logs, indent=2)}"
            
            # 1. Valós Google Trends (vagy tiszta fallback utasítás)
            real_time_trends = self.get_real_time_market_trends()

            # 2. Szigorú Claude Prompt HTML vázzal
            system_prompt = (
                "Te egy Vezetői Adatelemző és E-kereskedelmi Stratéga vagy az Antikvarius.ro-nál.\n\n"
                f"PIACI KONTEXTUS / TRENDEK:\n{real_time_trends}\n\n"
                "SÉRTHETETLEN SZABÁLY: Bármi is történik, a válaszodnak KIZÁRÓLAG egy teljes, vizuálisan formázott HTML dokumentumnak kell lennie (<html>-től </html>-ig). "
                "Tilos markdown kódblokkokat (```html) vagy bevezető mondatokat (pl. 'Íme a jelentés') használnod. Csak a nyers HTML kód jöhet!\n\n"
                "A jelentés KÖTELEZŐ elemei:\n"
                "1. Átfogó Összefoglaló: KPI kártyák (Összes interakció, RO/HU arány, Készüléktípusok).\n"
                "2. Zero-Match (Nincs találat) Elemzés: Pontosan mik voltak a sikertelen keresések a logok alapján?\n"
                "3. UX/Súrlódás Elemzés: Mikor indult be a kosárelhagyás vagy fizetési hezitálás trigger, és sikeres volt-e?\n"
                "4. Vezetői Stratégia: Milyen könyveket/kategóriákat szerezzünk be a keresések és a trendek alapján?\n\n"
                "KÖTELEZŐ HTML FORMÁZÁS: Használj modern CSS-t (fehér háttér, sötétkék címsorok #0b57d0, szürke árnyékos dobozok a KPI-oknak). "
                "A <head> részbe KÖTELEZŐEN tedd be a <meta charset=\"UTF-8\"> taget!"
            )
            
            log_event("🧠 Claude elemző processz indítása...")
            res = claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"Kérlek, elemezd az alábbi belső logokat:\n\n{log_summary}"}
                ]
            )
            
            # Golyóálló parser: ThinkingBlock és TextBlock kezelése
            raw_content = ""
            if res.content:
                for block in res.content:
                    # Kinyerjük a szöveget, bárhol is legyen
                    if getattr(block, 'type', '') == 'text':
                        raw_content += block.text
                    elif hasattr(block, 'text'):
                        raw_content += block.text
            
            raw_content = raw_content.strip()

            clean_html = ""
            
            # Biztonsági ellenőrzés: Megtagadta az AI a feladatot (pl. hibaüzenetet adott HTML helyett)?
            if len(raw_content) < 200:
                log_event(f"⚠️ A Claude válasza gyanúsan rövid ({len(raw_content)} karakter). Vészhelyzeti HTML generálása.")
                clean_html = f"""<!DOCTYPE html>
                <html lang="hu">
                <head><meta charset="UTF-8"></head>
                <body style="font-family: Arial, sans-serif; background-color: #fce4e4; padding: 20px; color: #333;">
                    <div style="background: white; padding: 20px; border-radius: 8px; border-left: 5px solid #d9534f; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        <h2 style="color: #d9534f; margin-top: 0;">⚠️ Rendszer Figyelmeztetés: AI Elemzési Hiba</h2>
                        <p>Az elemző modul a várt HTML riport helyett a következő váratlan és rövid választ adta:</p>
                        <pre style="background: #f4f4f4; padding: 15px; border: 1px solid #ccc; white-space: pre-wrap;">{raw_content}</pre>
                        <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;">
                        <p><strong>Napi feldolgozott interakciók száma:</strong> {len(logs)} db</p>
                        <p style="font-size: 12px; color: #777;">Ezt az e-mailt a Booksy AI biztonsági modulja generálta, hogy megelőzze az üres e-mailek kiküldését.</p>
                    </div>
                </body>
                </html>"""
            else:
                # Kikeressük az érvényes HTML-t Regex-el
                html_match = re.search(r'(?i)<html[\s\S]*</html>', raw_content)
                if html_match:
                    clean_html = html_match.group(0)
                else:
                    log_event("⚠️ A Claude nem rakott <html> taget a válaszba, regexes tisztítás aktiválva.")
                    clean_text = re.sub(r"^```(html)?\s*", "", raw_content, flags=re.IGNORECASE|re.MULTILINE)
                    clean_text = re.sub(r"```\s*$", "", clean_text, flags=re.IGNORECASE|re.MULTILINE)
                    clean_html = f"""<!DOCTYPE html>
                    <html lang="hu">
                    <head><meta charset="UTF-8"></head>
                    <body style="font-family: 'Segoe UI', Arial, sans-serif; background-color: #f4f7f6; padding: 20px;">
                        <div style="background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                            {clean_text}
                        </div>
                    </body>
                    </html>"""

            log_event(f"✅ Riport generálva. Végső HTML hossza: {len(clean_html)} karakter.")
            self.db.save_report("daily_analytics", yesterday, clean_html)
            
            smtp_server = os.getenv("SMTP_SERVER")
            smtp_sender = os.getenv("SMTP_SENDER")
            smtp_pass = os.getenv("SMTP_PASSWORD")
            
            if not all([smtp_server, smtp_sender, smtp_pass]):
                log_event("❌ Hiba: Hiányzó SMTP beállítások, analitika e-mail nem lett elküldve.")
                return

            try:
                msg = MIMEMultipart('alternative')
                msg['Subject'] = f"📊 Antikvarius.ro Vezetői AI Analitika - {yesterday}"
                msg['From'] = smtp_sender
                msg['To'] = ", ".join(ADMIN_EMAILS)
                
                # Kötelező levélszemét-szűrő elleni fejlécek
                msg['Date'] = formatdate(localtime=True)
                msg['Message-ID'] = make_msgid()
                
                # KIFEJEZETT UTF-8 kódolás a magyar és román karakterek miatt
                html_part = MIMEText(clean_html, 'html', 'utf-8')
                msg.attach(html_part)
                
                port = int(os.getenv("SMTP_PORT", 587))
                with smtplib.SMTP(smtp_server, port, timeout=20) as server:
                    server.starttls()
                    server.login(smtp_sender, smtp_pass)
                    # Szigorú Envelope Addressing a To fejléc helyett
                    server.sendmail(smtp_sender, ADMIN_EMAILS, msg.as_string())
                
                log_event("✅ Napi AI Analitika sikeresen elküldve a vezetőségnek.")
            except Exception as smtp_err:
                log_event(f"❌ SMTP Kapcsolódási/Küldési Hiba: {smtp_err}")

        except Exception as e:
            log_event(f"❌ Napi Analitika Generálási Hiba: {e}")