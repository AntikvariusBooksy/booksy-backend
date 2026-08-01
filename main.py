__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os, time
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from typing import Optional

from database import DBHandler, AnalyticsDB, log_event, clean_pii, get_geo_from_ip, LOCAL_TZ
from agent import BooksyProactiveAgent, AutoUpdater, AIAnalyticsAgent

db_handler = DBHandler()
analytics_db = AnalyticsDB()
updater = AutoUpdater(db_handler)
agent = BooksyProactiveAgent(db_handler)
analytics_agent = AIAnalyticsAgent()
scheduler = BackgroundScheduler()

/* STREAMING_CHUNK:Scheduler routines... */
def master_morning_routine():
    log_event("🌅 Master Láncreakció Indítása (V256)")
    updater.fetch_store_policies()
    try:
        sync_success = updater.run_daily_update()
        if not sync_success: log_event("⚠️ Szinkronizációs hiba, korábbi adatok használata.")
    except Exception as e: log_event(f"⚠️ Váratlan hiba a szinkronnál: {e}")

def daily_analytics_job():
    try: analytics_agent.generate_daily_report()
    except Exception as e: log_event(f"⚠️ Napi Analitika Hiba: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(master_morning_routine, CronTrigger(hour=7, minute=0, timezone=LOCAL_TZ))
    scheduler.add_job(daily_analytics_job, CronTrigger(hour=8, minute=0, timezone=LOCAL_TZ))
    scheduler.start(); yield; scheduler.shutdown()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_headers=["*"], allow_methods=["*"])

/* STREAMING_CHUNK:Request payload models... */
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

/* STREAMING_CHUNK:API endpoints... */
@app.get("/")
def home(): return {"status": "V256 Online (Intelligent Expert Agent)", "project": "Booksy"}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)