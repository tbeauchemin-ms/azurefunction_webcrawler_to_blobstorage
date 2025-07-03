# function_app.py

import os
import json
import logging

import azure.functions as func

from crawler import start_crawl, visited, failed

app = func.FunctionApp()

# ── Timer trigger for daily crawl ────────────────────────────────────────────
@app.schedule(
    schedule=os.getenv("CRAWL_CRON", "0 0 2 * * *"),  # default: 02:00 UTC daily
    arg_name="mytimer",
    run_on_startup=True,
    use_monitor=True
)
@app.function_name(name="WebCrawlTimerFunction")
def web_crawl_timer(mytimer: func.TimerRequest) -> None:
    logging.info("🔄 Timer trigger fired; starting crawl")
    start_crawl()  # your existing function
    logging.info("✅ Crawl finished; visited=%d failed=%d", len(visited), len(failed))

# ── HTTP trigger for health check ────────────────────────────────────────────
@app.route(route="ping", auth_level=func.AuthLevel.ANONYMOUS)
@app.function_name(name="PingHttpTrigger")
def ping(req: func.HttpRequest) -> func.HttpResponse:
    status = {
        "status": "ok",
        "visited": len(visited),
        "failed": len(failed)
    }
    logging.info("🩺 Health check: %s", status)
    return func.HttpResponse(
        body=json.dumps(status),
        status_code=200,
        mimetype="application/json"
    )
