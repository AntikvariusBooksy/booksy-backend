import os
import time
import requests
import unicodedata
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from bs4 import BeautifulSoup
from apscheduler.schedulers.background import BackgroundScheduler

# --- KONFIGURÁCIÓ ---
load_dotenv()
INDEX_NAME = "booksy-index"

# A Renderen beállított feed URL-t használjuk
XML_FEED_URL = os.getenv("XML_FEED_URL", "https://www.antikvarius.ro/wp-content/uploads/woo-feed/google/xml/booksyfullfeed.xml")

POLICY_URLS = {
    "FIZETÉS": "https://www.antikvarius.ro/hu/fizetesi-informaciok/",
    "SZÁLLÍTÁS": "https://www.antikvarius.ro/hu/szallitasi-informaciok/",
    "ÁSZF": "https://www.antikvarius.ro/hu/altalanos-szerzodesi-es-felhasznalasi-feltetelek/"
}

# --- ADATMODELLEK (EZT HAGYTAM KI AZ ELŐBB) ---
class ChatRequest(BaseModel):
    message: str

class HookRequest(BaseModel):
    url: str
    page_title: str
    visitor_type: str 
    cart_status: str 
    lang: str

# --- SEGÉDFÜGGVÉNYEK ---
def normalize_text(text):
    if not text: return ""
    text = str(text).lower()
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def clean_html(raw_html):
    if not raw_html: return ""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    cleantext = cleantext.replace("<![CDATA[", "").replace("]]>", "")
    return " ".join(cleantext.split())

def safe_str(val):
    if val is None: return ""
    return str(val).strip()

def extract_author(short_desc):
    if not short_desc: return ""
    match = re.search(r'(Szerző|Írta):\s*([^<|\n]+)', short_desc, re.IGNORECASE)
    if match: return match.group(2).strip()
    return ""

# --- AUTOMATIZÁLT FRISSÍTŐ MOTOR (V26 MIRROR SYNC) ---
class AutoUpdater:
    def __init__(self):
        self.api_key_openai = os.getenv("OPENAI_API_KEY")
        self.api_key_pinecone = os.getenv("PINECONE_API_KEY")
        self.client_ai = OpenAI(api_key=self.api_key_openai)
        self.pc = Pinecone(api_key=self.api_key_pinecone)
        self.index = self.pc.Index(INDEX_NAME)

    def scrape_policy(self):
        """Jogi szövegek frissítése"""
        print("🔄 [AUTO] Jogi információk frissítése...")
        full_policy_text = "[TUDÁSBÁZIS AZ ÜGYFÉLSZOLGÁLATHOZ - FRISSÍTVE: MA]\n"
        
        for category, url in POLICY_URLS.items():
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.content, 'html.parser')
                    for script in soup(["script", "style", "nav", "footer", "header"]): script.extract()
                    text = soup.get_text(separator=' ')
                    clean_text = ' '.join(text.split())
                    full_policy_text += f"\n--- {category} INFORMÁCIÓK ---\n{clean_text[:4000]}\n"
            except Exception as e:
                print(f"⚠️ Hiba a {category} letöltésekor: {e}")

        try:
            res = self.client_ai.embeddings.create(input="policy definition", model="text-embedding-3-small")
            self.index.upsert(vectors=[("store_policy", res.data[0].embedding, {"type": "policy", "content": full_policy_text})])
            print("✅ [AUTO] Jogi infók mentve.")
        except Exception as e:
            print(f"❌ [AUTO] Hiba a policy mentéskor: {e}")

    def update_books_from_feed(self):
        """Könyvek tükörszinkronja (Csak az marad, ami a feedben van)"""
        print(f"🔄 [AUTO] Könyv szinkronizáció innen: {XML_FEED_URL}")
        
        current_sync_ts = int(time.time())
        
        try:
            response = requests.get(XML_FEED_URL, stream=True, timeout=120)
            if response.status_code != 200:
                print(f"❌ [AUTO] Feed hiba: {response.status_code}")
                return

            tree = ET.fromstring(response.content)
            items = tree.findall('.//post')
            if not items: items = tree.findall('.//item')
            
            print(f"📚 [AUTO] Feed elemszám: {len(items)}")
            
            batch_vectors = []
            count = 0
            
            for item in items:
                try:
                    ns = {'g': 'http://base.google.com/ns/1.0'}
                    id_tag = item.find('ID')
                    if id_tag is None: id_tag = item.find('g:id', ns)
                    if id_tag is None or not id