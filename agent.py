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
                    # Golyóálló kinyerés Claude 5 esetére
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

    def get_real_time_market_trends(self):
        """Gemini-vel lekéri az aktuális Google trendeket Marosvásárhelyre és Romániára vonatkozóan."""
        log_event("🔍 Valós idejű webes trendelemzés indítása a Gemini-vel...")
        try:
            prompt = (
                "Keress rá a weben a legfrissebb könyveladási és olvasási trendekre Romániában, "
                "kifejezetten fókuszálva Erdélyre és Marosvásárhely környékére, valamint a magyar nyelvű olvasókra. "
                "Milyen műfajok, szerzők vagy könyvek pörögnek most a legjobban a könyvesboltokban vagy a Google keresésekben? "
                "Foglald össze röviden, 3-4 pontban, tényekre támaszkodva."
            )
            
            # A Gemini új generációs API hívása beépített Google Search groundinggal
            response = gemini_client.models.generate_content(
                model='gemini-1.5-flash',
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
            log_event(f"⚠️ Hiba a valós idejű trendek lekérésekor: {e}")
            return "Az élő internetes trendadatok lekérése nem sikerült. Kérlek, támaszkodj az általános könyvpiaci tudásodra a jelentésnél."

    def generate_and_send_daily_report(self):
        try:
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            logs = self.db.get_logs_for_date(yesterday)
            
            if not logs: 
                log_event("Nincs log tegnapról, analitika kihagyva.")
                return
            
            log_summary = f"Dátum: {yesterday}\nÖsszesen {len(logs)} interakció.\n{json.dumps(logs, indent=2)}"
            
            # 1. Lekérjük a valós Google Trends adatokat a Gemini-vel
            real_time_trends = self.get_real_time_market_trends()

            # 2. Átadjuk az adatokat és a fix HTML sablont a Claude-nak
            html_skeleton = """
            <!DOCTYPE html>
            <html lang="hu">
            <head>
                <meta charset="UTF-8">
                <style>
                    body { font-family: 'Segoe UI', Arial, sans-serif; background-color: #f4f7f6; color: #333; margin: 0; padding: 20px; }
                    .container { max-width: 800px; margin: auto; background: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
                    h1 { color: #0b57d0; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; font-size: 24px; }
                    h2 { color: #1a73e8; margin-top: 30px; font-size: 18px; border-bottom: 1px solid #eee; padding-bottom: 5px; }
                    h3 { color: #333; font-size: 16px; margin-top: 20px; }
                    .kpi-container { display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; }
                    .kpi-card { flex: 1; background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 6px; padding: 15px; text-align: center; min-width: 150px; }
                    .kpi-value { font-size: 24px; font-weight: bold; color: #0b57d0; }
                    .kpi-label { font-size: 12px; color: #666; text-transform: uppercase; }
                    .highlight { background-color: #e8f0fe; padding: 15px; border-left: 4px solid #1a73e8; margin: 15px 0; border-radius: 4px; }
                    .warning { background-color: #fef0f0; border-left: 4px solid #d93025; padding: 15px; margin: 15px 0; border-radius: 4px;}
                    ul { padding-left: 20px; }
                    li { margin-bottom: 8px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📊 Antikvarius.ro – Vezetői AI Analitika</h1>
                    <p><strong>Dátum:</strong> {yesterday}</p>
                    
                    <!-- KPI SZEKCIÓ -->
                    <div class="kpi-container">
                        <!-- Töltsd ki a KPI kártyákat az interakciókkal és arányokkal -->
                    </div>

                    <h2>🌍 1. Piaci Trendek (Google Keresések alapján)</h2>
                    <div class="highlight">
                        <!-- Írd le a valós idejű trendeket és elemzésüket itt -->
                    </div>

                    <h2>🔍 2. Keresési Elemzés és 'Nincs Találat' (Zero-Match)</h2>
                    <div class="warning">
                        <!-- Emeled ki, mit kerestek sikertelenül, és miért probléma ez -->
                    </div>
                    
                    <h2>📉 3. UX és Súrlódás (Kosárelhagyás, Fizetés)</h2>
                    <!-- Emezd ki a proaktív triggereket (cart abandonment, stb) -->

                    <h2>💡 4. Beszerzési Javaslatok a Vezetőségnek</h2>
                    <!-- A Zero-match és a Piaci trendek összevetése -->
                </div>
            </body>
            </html>
            """

            system_prompt = (
                f"Te egy Vezetői Adatelemző és E-kereskedelmi Stratéga vagy az Antikvarius.ro-nál. "
                "A feladatod egy profitábilis, vezetői jelentés megírása a mellékelt HTML sablon (Skeleton) alapján.\n\n"
                f"1. VALÓS IDEJŰ WEBES TRENDEK (Ezt a Google biztosította most, használd fel az elemzésben!):\n{real_time_trends}\n\n"
                "UTASÍTÁSOK:\n"
                "- Fogd a megadott HTML vázat, és TÖLTSD KI a megfelelő adatokkal és a te elemző, C-level szövegeddel a megjegyzések (<!-- ... -->) helyén.\n"
                "- A kimeneted KIZÁRÓLAG az érvényes, kiegészített HTML kód lehet! Szigorúan a <html> taggel kell kezdődnie és a </html> taggel végződnie.\n"
                "- TILOS bármilyen markdown kódblokkot (pl. ```html) vagy magyarázó szöveget tenni a HTML elé vagy mögé!\n"
                "- Végezz mély elemzést: Keresd meg a logokban a hiánycikkeket (Zero-match), és vesd össze azokat a megadott valós webes trendekkel. Tegyél javaslatot a beszerzésre!\n"
                "- Értékeld a kosárelhagyási és fizetési akadályokat (friction)."
            )
            
            log_event("🧠 Claude elemző processz indítása...")
            res = claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"Itt vannak a tegnapi logok. Kérlek, add vissza a kitöltött HTML-t.\n\nLogok:\n{log_summary}\n\nHTML Sablon:\n{html_skeleton}"}
                ]
            )
            
            # Golyóálló kinyerés a Claude válaszból
            raw_content = ""
            if res.content:
                raw_content = res.content[0].text.strip()
            
            # Reguláris kifejezéssel levadásszuk a tiszta HTML-t, kizárva a markdown szemetet
            clean_html = ""
            html_match = re.search(r'(?i)<html[\s\S]*</html>', raw_content)
            
            if html_match:
                clean_html = html_match.group(0)
            else:
                # Ha a Claude mégis "elfelejtette" a html taget, letisztítjuk és mi csomagoljuk be
                log_event("⚠️ A Claude nem rakott <html> taget a válaszba, kényszerített csomagolás aktiválva.")
                clean_text = re.sub(r"^```(html)?\s*", "", raw_content, flags=re.IGNORECASE|re.MULTILINE)
                clean_text = re.sub(r"```\s*$", "", clean_text, flags=re.IGNORECASE|re.MULTILINE)
                clean_html = f"<!DOCTYPE html>\n<html lang=\"hu\">\n<head>\n<meta charset=\"UTF-8\">\n<style>body {{ font-family: Arial, sans-serif; padding: 20px; }}</style>\n</head>\n<body>\n{clean_text}\n</body>\n</html>"

            if len(clean_html) < 100:
                log_event(f"❌ A megtisztított HTML túl rövid (Hossz: {len(clean_html)}). Generálás megszakítva.")
                return

            log_event(f"✅ Riport generálva. HTML hossza: {len(clean_html)} karakter.")
            self.db.save_report("daily_analytics", yesterday, clean_html)
            
            # 3. SMTP E-mail Küldés Szigorú UTF-8 Kódolással és Sendmail formátummal
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
                
                # Kötelező anti-spam fejlécek
                msg['Date'] = formatdate(localtime=True)
                msg['Message-ID'] = make_msgid()
                
                # KIFEJEZETT UTF-8 kódolás (A Spam szűrők és ékezetek miatt KÖTELEZŐ)
                html_part = MIMEText(clean_html, 'html', 'utf-8')
                msg.attach(html_part)
                
                port = int(os.getenv("SMTP_PORT", 587))
                with smtplib.SMTP(smtp_server, port, timeout=20) as server:
                    # Titkosítás bekapcsolása (A TLS KÖTELEZŐ a modern szervereknél)
                    server.starttls()
                    server.login(smtp_sender, smtp_pass)
                    
                    # SZIGORÚ boríték szintű címzés a fejléc helyett (Envelope Addressing)
                    server.sendmail(smtp_sender, ADMIN_EMAILS, msg.as_string())
                
                log_event("✅ Napi AI Analitika sikeresen elküldve a vezetőségnek.")
            except Exception as smtp_err:
                log_event(f"❌ SMTP Kapcsolódási/Küldési Hiba: {smtp_err}")

        except Exception as e:
            log_event(f"❌ Napi Analitika Generálási Hiba: {e}")