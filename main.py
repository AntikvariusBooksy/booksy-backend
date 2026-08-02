__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from fastapi import BackgroundTasks
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from database import DBHandler, AnalyticsDB, clean_pii, get_geo_from_ip, update_store_policies
from agent import BooksyProactiveAgent, BooksyAnalyticsReporter

db_handler = DBHandler()
analytics_db = AnalyticsDB()
agent = BooksyProactiveAgent(db_handler)
reporter = BooksyAnalyticsReporter(analytics_db)
LOCAL_TZ = pytz.timezone('Europe/Bucharest')

scheduler = BackgroundScheduler()
# Minden reggel 8:00-kor lefut az AI analitika generálása és kiküldése
scheduler.add_job(reporter.generate_and_send_daily_report, CronTrigger(hour=8, minute=0, timezone=LOCAL_TZ))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Szerver indulásakor betöltjük a legfrissebb szabályzatokat a memóriába (RAG)
    update_store_policies()
    scheduler.start()
    yield
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_headers=["*"], allow_methods=["*"])

class ChatRequest(BaseModel): 
    message: str; context_url: Optional[str] = ""; session_id: Optional[str] = ""
    device_type: Optional[str] = "Desktop"; ui_lang: Optional[str] = "hu"
    chat_lang: Optional[str] = "hu"; target_catalog: Optional[str] = "mixed"
    user_mode: Optional[str] = "felfedezo"

class ProactiveRequest(BaseModel):
    trigger_type: str
    session_id: str
    context_url: Optional[str] = ""
    failed_search_term: Optional[str] = ""
    last_book_title: Optional[str] = ""
    ui_lang: Optional[str] = "hu"
    device_type: Optional[str] = "Desktop"
    user_mode: Optional[str] = "felfedezo"

class InitRequest(BaseModel): url: str; session_id: str; ui_lang: str = "hu"

@app.get("/")
def home(): return {"status": "V259 Online (Proactive Agent - Claude Fixed, Policy RAG)", "project": "Booksy"}

@app.post("/chat")
def chat(req: ChatRequest, request: Request): 
    start_time = time.time()
    bot_response = agent.process_chat(req.message, req.ui_lang, req.user_mode)
    latency = int((time.time() - start_time) * 1000)
    
    forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip = forwarded_for.split(",")[0].strip() if forwarded_for else (request.client.host if request.client else None)
    geo_country, geo_region = get_geo_from_ip(client_ip)
    
    log_data = {
        "session_id": req.session_id, "user_msg": clean_pii(req.message), "bot_reply": bot_response.get("reply", "")[:200],
        "context_url": req.context_url, "geo_country": geo_country, "geo_region": geo_region,
        "ui_language": req.ui_lang, "chat_language": req.chat_lang, "target_catalog": req.target_catalog,
        "offered_book_ids": ",".join([p.get("id", "") for p in bot_response.get("products", [])]) if bot_response.get("products") else "", 
        "zero_match_flag": bot_response.get("zero_match_flag", False),
        "latency_ms": latency, "device_type": req.device_type, "trigger_type": "manual"
    }
    analytics_db.log_chat(log_data)
    return bot_response

@app.post("/proactive-hook")
def proactive_hook(req: ProactiveRequest, request: Request):
    start_time = time.time()
    session_data = {
        "failed_search_term": req.failed_search_term,
        "last_book_title": req.last_book_title,
        "ui_lang": req.ui_lang,
        "user_mode": req.user_mode
    }
    
    bot_response = agent.process_proactive_trigger(req.trigger_type, session_data)
    latency = int((time.time() - start_time) * 1000)
    
    forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip = forwarded_for.split(",")[0].strip() if forwarded_for else (request.client.host if request.client else None)
    geo_country, geo_region = get_geo_from_ip(client_ip)
    
    log_data = {
        "session_id": req.session_id, "user_msg": f"[TRIGGER]: {req.trigger_type}", "bot_reply": bot_response.get("reply", "")[:200],
        "context_url": req.context_url, "geo_country": geo_country, "geo_region": geo_region,
        "ui_language": req.ui_lang, "chat_language": req.ui_lang, "target_catalog": "mixed",
        "offered_book_ids": ",".join([p.get("id", "") for p in bot_response.get("products", [])]) if bot_response.get("products") else "", 
        "zero_match_flag": False,
        "latency_ms": latency, "device_type": req.device_type, "trigger_type": req.trigger_type
    }
    analytics_db.log_chat(log_data)
    return bot_response

@app.post("/init-chat")
def init_chat(req: InitRequest): 
    if req.ui_lang == "hu":
        return {"ui_lang": "hu", "bubble_text": "Miben segíthetek?", "placeholder": "Kérdezz bármit..."}
    else:
        return {"ui_lang": "ro", "bubble_text": "Cu ce te pot ajuta?", "placeholder": "Întreabă orice..."}

@app.post("/test-analytics")
def test_analytics(bt: BackgroundTasks):
    # A háttérben elindítja a folyamatot, így a hívó nem fagy le amíg a GPT és az SMTP dolgozik
    bt.add_task(reporter.generate_and_send_daily_report)
    return {"status": "Triggered", "message": "AI Analitika generálása és e-mail küldés elindult a háttérben!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)