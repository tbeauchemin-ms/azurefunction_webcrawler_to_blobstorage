import os
import re
import time
import uuid
import json
import logging
import requests
import hashlib
from io import BytesIO
from urllib.parse import urlparse, urljoin, urlunparse
from urllib import robotparser
from threading import Lock
from xml.etree import ElementTree as ET
from readability import Document
from bs4 import BeautifulSoup
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from PyPDF2 import PdfReader
from dateutil import parser as date_parser, tz
from applicationinsights import TelemetryClient
from docx import Document as DocxDocument
import pandas as pd

# ----------- Logging and Env Config -----------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
os_env = os.getenv
STORAGE_ACCOUNT = os.environ["STORAGE_ACCOUNT_NAME"]
CONTENT_CONT = os_env("CONTAINER_NAME", "content")
LOGS_CONT = os_env("LOG_CONTAINER_NAME", "logs")
ALLOW_DOMAINS = {d.strip().lower() for d in os_env("ALLOW_DOMAINS", "").split(";") if d.strip()}
SKIP_PATTERNS = [re.compile(p) for p in os_env("SKIP_REGEXES", "").split(";") if p]
MAX_DEPTH = int(os_env("MAX_DEPTH", "3"))
REQUEST_DELAY = float(os_env("REQUEST_DELAY", "0.5"))
PAGE_TIMEOUT_MS = int(os_env("PAGE_TIMEOUT_MS", "45000"))
NETWORK_IDLE_WAIT_MS = int(os_env("NETWORK_IDLE_WAIT_MS", "0"))
MAX_CONTENT_CHARS = int(os_env("MAX_CONTENT_CHARS", "500000"))
INCLUDE_PDFS = os_env("INCLUDE_PDFS", "true").lower() == "true"
PDF_DOWNLOAD_TIMEOUT_MS = int(os_env("PDF_DOWNLOAD_TIMEOUT_MS", "60000"))
PDF_CHUNK_CHARS = int(os_env("PDF_CHUNK_CHARS", "4000"))
PDF_CHUNK_OVERLAP = int(os_env("PDF_CHUNK_OVERLAP", "300"))
MAX_PDF_BYTES = int(os_env("MAX_PDF_BYTES", "20000000"))
RESPECT_ROBOTS = os_env("RESPECT_ROBOTS", "true").lower() == "true"
SAVE_404 = os_env("SAVE_404", "false").lower() == "true"
USER_AGENT = os_env("USER_AGENT", "Mozilla/5.0 (GenericCrawler/1.0)")
INSIGHTS_KEY = os_env("APPINSIGHTS_INSTRUMENTATIONKEY", "")
RETRY_COUNT = int(os_env("RETRY_COUNT", "3"))
RETRY_BACKOFF = float(os_env("RETRY_BACKOFF_FACTOR", "2"))

cred = DefaultAzureCredential()
blob_service = BlobServiceClient(
    account_url=f"https://{STORAGE_ACCOUNT}.blob.core.windows.net",
    credential=cred
)
content_container = blob_service.get_container_client(CONTENT_CONT)
logs_container = blob_service.get_container_client(LOGS_CONT)
tc = TelemetryClient(INSIGHTS_KEY) if INSIGHTS_KEY else None

visited = set()
documents = []
sitemap_urls = []
failed = []
robots_cache = {}
collection_lock = Lock()

# ----------- Utility Functions -----------

def sanitize(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()

def normalize_url(url):
    """Lowercase scheme/netloc; DO NOT lowercase path!"""
    p = urlparse(url)
    scheme = "https"
    netloc = p.netloc.lower().replace("www.", "")
    path = p.path.rstrip("/")
    return urlunparse((scheme, netloc, path, "", "", ""))

def blob_name_for_url(url, chunk_index=None):
    norm_url = normalize_url(url)
    url_hash = hashlib.md5(norm_url.encode("utf-8")).hexdigest()
    parsed = urlparse(norm_url)
    base = sanitize(parsed.path or "root")
    if chunk_index is not None:
        return f"{base}_{chunk_index}_{url_hash}.json"
    else:
        return f"{base}_{url_hash}.json"

def robot_allows(url, agent):
    if not RESPECT_ROBOTS:
        return True
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    if base not in robots_cache:
        rp = robotparser.RobotFileParser()
        rp.set_url(f"{base}/robots.txt")
        try:
            rp.read()
            robots_cache[base] = rp
        except:
            robots_cache[base] = None
    rp = robots_cache.get(base)
    return rp is None or rp.can_fetch(agent, url)

def should_skip(url):
    parsed = urlparse(url)
    return parsed.scheme not in ("http", "https") or any(p.search(url) for p in SKIP_PATTERNS)

def is_allowed_link(link):
    host = urlparse(link).netloc.lower().replace("www.", "")
    return any(host == d or host.endswith(f".{d}") for d in ALLOW_DOMAINS)

def upload_json(data, blob_name, container, last_modified=None):
    try:
        meta = {}
        if last_modified:
            meta["last_modified"] = last_modified
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode()
        cs = ContentSettings(content_type="application/json")
        blob_client = container.get_blob_client(blob_name)
        if blob_client.exists():
            props = blob_client.get_blob_properties()
            blob_lastmod = props.metadata.get("last_modified") if props.metadata else None
            if last_modified and blob_lastmod == last_modified:
                logging.info(f"Skipping unchanged content: {blob_name}")
                return False
        blob_client.upload_blob(
            data=payload,
            overwrite=True,
            content_settings=cs,
            metadata=meta
        )
        if tc:
            tc.track_event("Upload", {"blob": blob_name})
        return True
    except Exception:
        logging.exception(f"Upload failed: {blob_name}")
        if tc:
            tc.track_exception()
        return False

def parse_sitemap(url, seen=None):
    seen = seen or set()
    try:
        res = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        if res.status_code != 200:
            logging.warning("Sitemap fetch failed: %s", url)
            return []
        root = ET.fromstring(res.content)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        if root.tag.endswith("sitemapindex"):
            urls = []
            for loc in root.findall(".//sm:loc", ns):
                link = loc.text
                if link and link not in seen:
                    seen.add(link)
                    urls += parse_sitemap(link, seen)
            return urls
        return [
            loc.text for loc in root.findall(".//sm:loc", ns)
            if loc.text and not should_skip(loc.text)
        ]
    except Exception as e:
        logging.warning("Error parsing sitemap %s: %s", url, e)
        return []

def extract_main_content(html):
    try:
        doc = Document(html)
        main_html = doc.summary()
        soup = BeautifulSoup(main_html, "html.parser")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
    for sel in [
        "[id*=lang]", "[class*=lang]", "[id*=translat]", "[class*=translat]",
        "[id*=goog]", "[class*=goog]", ".goog-te-banner-frame", ".goog-te-menu-frame",
        "[id*=cookie]", "[class*=cookie]", ".header", ".footer", ".sidebar",
        "header", "footer", "nav", "aside", "form", "script", "style"
    ]:
        for el in soup.select(sel):
            el.decompose()
    main = soup.find("main") or soup.find(id="main-content") or soup.find("div", class_="content") or soup.body
    text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)
    lines = text.splitlines()
    filtered = []
    in_lang_block = False
    for line in lines:
        if 'select language' in line.lower() or (len(line) > 40 and sum(1 for w in line.split() if len(w) > 5) > 4):
            in_lang_block = True
            continue
        if in_lang_block:
            if 'powered by translate' in line.lower() or not line.strip():
                in_lang_block = False
            continue
        filtered.append(line)
    cleaned_text = "\n".join(filtered).strip()
    return cleaned_text if len(cleaned_text) > 200 else text

# ----------- JavaScript HREF Extraction -----------

JS_OPEN_RE = re.compile(r'''javascript:window\.open\((["'])(.+?\.(pdf|docx|xlsx|doc|xls))\1''', re.IGNORECASE)

def extract_real_link(base_url, href):
    if href.lower().startswith("javascript:"):
        m = JS_OPEN_RE.search(href)
        if m:
            relative_url = m.group(2)
            # Ignore .doc and .xls
            if relative_url.lower().endswith((".doc", ".xls")):
                return None
            return normalize_url(urljoin(base_url, relative_url))
        else:
            return None
    else:
        return normalize_url(urljoin(base_url, href))

# ----------- Handlers for Files/Pages ------------

def handle_docx(url: str):
    norm_url = normalize_url(url)
    logging.info("Visiting DOCX: %s", norm_url)
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=PDF_DOWNLOAD_TIMEOUT_MS/1000)
        r.raise_for_status()
        doc = DocxDocument(BytesIO(r.content))
        content = "\n".join(para.text for para in doc.paragraphs)
        if not content.strip():
            return
        last_mod = r.headers.get("Last-Modified", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        rec = {
            "id": uuid.uuid4().hex,
            "url": norm_url,
            "title": sanitize(urlparse(norm_url).path),
            "content": content,
            "last_modified": last_mod,
        }
        filename = blob_name_for_url(norm_url)
        upload_json(rec, filename, content_container, last_modified=last_mod)
        with collection_lock:
            documents.append(rec)
            sitemap_urls.append(norm_url)
    except Exception as e:
        logging.warning(f"Error handling docx: {url} {e}")
        with collection_lock:
            failed.append({"url": url, "reason": f"DOCX parse error: {e}"})

def handle_xlsx(url: str):
    norm_url = normalize_url(url)
    logging.info("Visiting XLSX: %s", norm_url)
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=PDF_DOWNLOAD_TIMEOUT_MS/1000)
        r.raise_for_status()
        xls = pd.ExcelFile(BytesIO(r.content))
        text_chunks = []
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            text_chunks.append(f"Sheet: {sheet}\n{df.to_string(index=False)}")
        content = "\n\n".join(text_chunks)
        if not content.strip():
            return
        last_mod = r.headers.get("Last-Modified", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        rec = {
            "id": uuid.uuid4().hex,
            "url": norm_url,
            "title": sanitize(urlparse(norm_url).path),
            "content": content,
            "last_modified": last_mod,
        }
        filename = blob_name_for_url(norm_url)
        upload_json(rec, filename, content_container, last_modified=last_mod)
        with collection_lock:
            documents.append(rec)
            sitemap_urls.append(norm_url)
    except Exception as e:
        logging.warning(f"Error handling xlsx: {url} {e}")
        with collection_lock:
            failed.append({"url": url, "reason": f"XLSX parse error: {e}"})

def handle_pdf(playwright_ctx, url: str):
    norm_url = normalize_url(url)
    logging.info("Visiting PDF: %s", norm_url)
    if not INCLUDE_PDFS:
        return []
    for attempt in range(RETRY_COUNT):
        try:
            h = requests.head(url, headers={"User-Agent": USER_AGENT}, timeout=5)
            size = int(h.headers.get("Content-Length", 0))
            if size > MAX_PDF_BYTES:
                logging.warning("Skipping large PDF %d bytes: %s", size, norm_url)
                return []
            break
        except Exception:
            time.sleep(REQUEST_DELAY * (RETRY_BACKOFF**attempt))
    content_bytes, headers = None, {}
    for attempt in range(RETRY_COUNT):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=PDF_DOWNLOAD_TIMEOUT_MS/1000)
            if r.status_code == 403:
                pg = playwright_ctx.new_page()
                pg.set_extra_http_headers({"User-Agent": USER_AGENT})
                pr = pg.goto(url, timeout=PAGE_TIMEOUT_MS, wait_until="domcontentloaded")
                if NETWORK_IDLE_WAIT_MS:
                    pg.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_WAIT_MS)
                content_bytes = pr.body()
                headers = pr.headers
                pg.close()
            else:
                r.raise_for_status()
                content_bytes = r.content
                headers = r.headers
            break
        except (requests.RequestException, PlaywrightTimeoutError) as e:
            logging.warning("PDF fetch attempt %d failed: %s", attempt+1, e)
            time.sleep(REQUEST_DELAY * (RETRY_BACKOFF**attempt))
    else:
        with collection_lock:
            failed.append({"url": url, "reason": "PDF fetch failed"})
        return []
    if not content_bytes or b"%PDF" not in content_bytes[:4]:
        with collection_lock:
            failed.append({"url": url, "reason": "Invalid PDF"})
        return []
    try:
        reader = PdfReader(BytesIO(content_bytes))
    except Exception as e:
        with collection_lock:
            failed.append({"url": url, "reason": f"PDF parse error: {e}"})
        return []
    raw = "\n\n".join((p.extract_text() or "") for p in reader.pages)
    if not raw.strip():
        return []
    chunks, i = [], 0
    while i < len(raw):
        chunks.append(raw[i : i + PDF_CHUNK_CHARS])
        i += PDF_CHUNK_CHARS - PDF_CHUNK_OVERLAP
    lm_hdr = headers.get("Last-Modified") if headers else None
    if lm_hdr:
        dt = date_parser.parse(lm_hdr)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz.UTC)
        last_mod = dt.astimezone(tz.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        last_mod = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if tc:
        tc.track_metric("PDFChunks", len(chunks))
    with collection_lock:
        for idx, chunk in enumerate(chunks, 1):
            rec = {
                "id": uuid.uuid4().hex,
                "url": norm_url,
                "title": sanitize(urlparse(norm_url).path),
                "chunk_index": idx,
                "chunk_total": len(chunks),
                "content": chunk,
                "last_modified": last_mod,
            }
            filename = blob_name_for_url(norm_url, chunk_index=idx)
            upload_json(rec, filename, content_container, last_modified=last_mod)
            documents.append(rec)
            sitemap_urls.append(norm_url)
    return []

# ----------- Main HTML Handler with Logging -----------

def enforce_trailing_slash_if_directory(url):
    return url

def handle_page(playwright_ctx, url: str):
    url = enforce_trailing_slash_if_directory(url)
    norm_url = normalize_url(url)
    logging.info("Visiting HTML: %s", norm_url)
    if RESPECT_ROBOTS and not robot_allows(norm_url, USER_AGENT):
        return []
    pg = playwright_ctx.new_page()
    try:
        resp = None
        raw_links = []
        html = None
        soup = None
        headers = {}
        for attempt in range(RETRY_COUNT):
            try:
                resp = pg.goto(url, timeout=PAGE_TIMEOUT_MS, wait_until="domcontentloaded")
                if NETWORK_IDLE_WAIT_MS:
                    pg.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_WAIT_MS)
                if resp.status == 429:
                    time.sleep(REQUEST_DELAY * (RETRY_BACKOFF ** attempt))
                    continue
                if resp.status >= 500:
                    raise PlaywrightTimeoutError(f"{resp.status}")
                html = pg.content()
                headers = resp.headers
                soup = BeautifulSoup(html, "html.parser")
                # Get all hrefs from rendered DOM (JavaScript)
                raw_links = pg.eval_on_selector_all(
                    "a[href]",
                    "els => els.map(e => e.getAttribute('href'))"
                )
                break
            except PlaywrightTimeoutError as e:
                logging.warning("HTML fetch attempt %d failed: %s", attempt+1, e)
                time.sleep(REQUEST_DELAY * (RETRY_BACKOFF ** attempt))
            except Exception as e:
                logging.warning("HTML fetch attempt %d failed: %s", attempt+1, e)
                time.sleep(REQUEST_DELAY * (RETRY_BACKOFF ** attempt))
        if soup is None:
            with collection_lock:
                failed.append({"url": url, "reason": "All fetch attempts failed or invalid HTML"})
            return []

        # PATCH: Backup extraction using BeautifulSoup for any missed links (safety net!)
        soup_links = [a.get("href") for a in soup.find_all("a", href=True)]
        all_raw_links = set(raw_links) | set(soup_links)  # union, remove dups

        # PATCH: Always resolve links relative to current page!
        absolute_links = []
        for l in all_raw_links:
            real = extract_real_link(url, l)
            if real:
                absolute_links.append(real)
        logging.info(f"All absolute links found on {url}: {absolute_links}")

        # Log filtering
        for l in absolute_links:
            logging.info(f"Filter: {l} | allowed: {is_allowed_link(l)}, skip: {should_skip(l)}, visited: {l in visited}")

        if not resp:
            with collection_lock:
                failed.append({"url": url, "reason": "No response from server"})
            return []
        if resp.status == 404:
            if SAVE_404:
                rec404 = {
                    "id": uuid.uuid4().hex,
                    "url": norm_url,
                    "content": "",
                    "status": 404
                }
                filename = f"404_{sanitize(urlparse(norm_url).path)}_{rec404['id']}.json"
                upload_json(rec404, filename, content_container)
            return []
        if resp.status >= 400:
            with collection_lock:
                failed.append({"url": url, "reason": f"HTTP {resp.status}"})
            return []

        main_text = extract_main_content(html)[:MAX_CONTENT_CHARS]
        last_mod = headers.get("Last-Modified", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        rec = {
            "id": uuid.uuid4().hex,
            "url": norm_url,
            "title": sanitize(urlparse(norm_url).path),
            "content": main_text,
            "last_modified": last_mod
        }
        filename = blob_name_for_url(norm_url)
        upload_json(rec, filename, content_container, last_modified=last_mod)
        with collection_lock:
            documents.append(rec)
            sitemap_urls.append(norm_url)
        filtered_links = [
            l for l in set(absolute_links)
            if is_allowed_link(l) and not should_skip(l)
        ]
        logging.info(f"Filtered to {len(filtered_links)} crawlable links on {url}: {filtered_links}")
        return filtered_links
    except Exception as e:
        logging.warning(f"General error handling HTML {url}: {e}")
        with collection_lock:
            failed.append({"url": url, "reason": f"HTML error: {e}"})
        return []
    finally:
        pg.close()

# ----------- Crawler Recursion/Walk -----------

def walk(playwright_ctx, url: str, depth=0):
    norm_url = normalize_url(url)
    logging.info("Visiting: %s (depth %d)", norm_url, depth)
    with collection_lock:
        if (norm_url in visited
            or depth >= MAX_DEPTH
            or should_skip(norm_url)
            or (RESPECT_ROBOTS and not robot_allows(norm_url, USER_AGENT))):
            if norm_url in visited:
                logging.info(f"Already visited: {norm_url}")
            if depth >= MAX_DEPTH:
                logging.info(f"Max depth reached: {norm_url}")
            if should_skip(norm_url):
                logging.info(f"Skipped by regex: {norm_url}")
            if (RESPECT_ROBOTS and not robot_allows(norm_url, USER_AGENT)):
                logging.info(f"Blocked by robots.txt: {norm_url}")
            return
        visited.add(norm_url)

    try:
        if norm_url.endswith(".pdf"):
            handle_pdf(playwright_ctx, url)
        elif norm_url.endswith(".docx"):
            handle_docx(url)
        elif norm_url.endswith(".xlsx"):
            handle_xlsx(url)
        elif norm_url.endswith(".doc"):
            logging.info("Skipping .doc (not supported reliably in Python)")
            return
        elif norm_url.endswith(".xls"):
            logging.info("Skipping .xls (old Excel format, add xlrd if needed)")
            return
        else:
            links = handle_page(playwright_ctx, url)
            logging.info(f"Following {len(links)} child links from {url}")
            for link in links:
                time.sleep(REQUEST_DELAY)
                walk(playwright_ctx, link, depth + 1)
    except Exception as e:
        logging.warning(f"walk error on {url}: {e}")
        with collection_lock:
            failed.append({"url": url, "reason": f"Walk error: {e}"})

# ----------- Entry Point ------------

def start_crawl():
    try:
        seeds = [u.strip() for u in os_env("BASE_URLS", "").split(";") if u.strip()]
        for s in seeds:
            ALLOW_DOMAINS.add(urlparse(s).netloc.lower().replace("www.", ""))
        urls = []
        for seed in seeds:
            sm = seed.rstrip("/") + "/sitemap.xml"
            entries = parse_sitemap(sm)
            if entries:
                logging.info("✅ Using sitemap for %s (%d URLs)", seed, len(entries))
                urls.extend(u for u in entries if normalize_url(u) not in visited)
            else:
                logging.info("⚠️  No sitemap for %s; falling back to seed", seed)
                urls.append(seed)
        logging.info("🚀 Starting crawl with %d starting URLs", len(urls))
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
            context_opts = {
                "user_agent": USER_AGENT,
                "locale": "en-US",
                "accept_downloads": True,
                "java_script_enabled": True,
            }
            ctx = browser.new_context(**context_opts)
            for url in urls:
                walk(ctx, url)
            browser.close()
    except Exception as e:
        logging.error(f"Crawl failed: {e}", exc_info=True)
    finally:
        if failed:
            upload_json(failed, "failed_url.json", logs_container)
        if sitemap_urls:
            upload_json(sitemap_urls, "sitemap.json", logs_container)
        if documents:
            upload_json(documents, "docs.json", logs_container)
        logging.info(f"Crawl complete, output files written. Visited={len(visited)} Failed={len(failed)}")

if __name__ == "__main__":
    start_crawl()
