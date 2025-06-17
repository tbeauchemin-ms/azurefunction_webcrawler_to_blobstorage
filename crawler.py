import os
import re
import time
import uuid
import json
import logging
import requests
import hashlib

from collections import deque
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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
MAX_CONTENT_CHARS = int(os_env("MAX_CONTENT_CHARS", "50000"))

INCLUDE_PDFS = os_env("INCLUDE_PDFS", "true").lower() == "true"
INCLUDE_DOCX = os_env("INCLUDE_DOCX", "true").lower() == "true"
INCLUDE_XLSX = os_env("INCLUDE_XLSX", "true").lower() == "true"

PDF_DOWNLOAD_TIMEOUT_MS = int(os_env("PDF_DOWNLOAD_TIMEOUT_MS", "60000"))
CHUNK_CHARS = int(os_env("CHUNK_CHARS", "4000"))
CHUNK_OVERLAP = int(os_env("CHUNK_OVERLAP", "300"))
MAX_PDF_BYTES = int(os_env("MAX_PDF_BYTES", "20000000"))

RESPECT_ROBOTS = os_env("RESPECT_ROBOTS", "true").lower() == "true"
USER_AGENT = os_env("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

INSIGHTS_KEY = os_env("APPINSIGHTS_INSTRUMENTATIONKEY", "")
RETRY_COUNT = int(os_env("RETRY_COUNT", "3"))
RETRY_BACKOFF = float(os_env("RETRY_BACKOFF_FACTOR", "2"))
SAVE_404 = os_env("SAVE_404", "false").lower() == "true"

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

# Dedup cache: (url, chunk_index)
seen = set()
try:
    with open("docs.json", "r") as f:
        for d in json.load(f):
            seen.add((d["url"], d.get("chunk_index", 1)))
except FileNotFoundError:
    pass

session = requests.Session()
session.headers.update({
    "User-Agent": USER_AGENT,
    "Accept": "application/pdf,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.google.com",
})
session.mount('https://', HTTPAdapter(max_retries=Retry(
    total=RETRY_COUNT, backoff_factor=RETRY_BACKOFF,
    status_forcelist=[403, 429, 500, 502, 503, 504]))
)

# ----------- Utility Functions -----------

def get_domain_session(base_url: str) -> requests.Session:
    """
    Return a copy of the global session, with its headers cloned,
    ready for per‐domain customization.
    """
    dom_sess = requests.Session()
    # copy adapters from the global session (so you keep your retry logic)
    for prefix, adapter in session.adapters.items():
        dom_sess.mount(prefix, adapter)

    # copy over any default headers you want
    dom_sess.headers.update(session.headers)

    # now you can update per-domain headers as needed
    return dom_sess


def sanitize(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()

def normalize_url(url):
    """
    Lowercase the scheme and netloc, strip 'www.', remove trailing slash.
    Ensures that URLs differing only by case or slash (and fragment) are treated identically.
    """
    p = urlparse(url)
    scheme = "https"
    netloc = p.netloc.lower().replace("www.", "")
    path = p.path.rstrip("/")
    return urlunparse((scheme, netloc, path, "", "", ""))

def blob_name_for_url(url, chunk_index=None):
    """
    Generate a unique blob name based on the normalized URL and optional chunk_index.
    """
    # include fragment in hash so that tab panes get unique names
    parsed = urlparse(url)
    fragment = parsed.fragment or ""
    base_url = urlunparse((parsed.scheme, parsed.netloc.lower().replace("www.", ""), parsed.path.rstrip("/"), "", "", ""))
    url_hash = hashlib.md5((base_url + "#" + fragment).encode("utf-8")).hexdigest()
    base = sanitize(parsed.path or "root")
    if fragment:
        base = f"{base}_{sanitize(fragment)}"
    if chunk_index is not None:
        return f"{base}_{chunk_index}_{url_hash}.json"
    else:
        return f"{base}_{url_hash}.json"

def robot_allows(url, agent):
    """
    Cache robots.txt parsing per host. Return True if allowed, False otherwise.
    """
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
    """
    Skip URLs whose scheme is not http/https or that match any SKIP_PATTERN.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return True
    for p in SKIP_PATTERNS:
        if p.search(url):
            return True
    return False

def is_allowed_link(link):
    """
    Return True if link's host is in ALLOW_DOMAINS (or a permitted subdomain).
    """
    host = urlparse(link).netloc.lower().replace("www.", "")
    return any(host == d or host.endswith(f".{d}") for d in ALLOW_DOMAINS)

def upload_json(data, blob_name, container, last_modified=None):
    """
    Upload a JSON blob. If the blob already exists with the same last_modified metadata, skip.
    """
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
                logging.debug(f"Skipping unchanged content: {blob_name}")
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
    """
    Recursively parse sitemap.xml. Return a list of all URLs under that sitemap, excluding ones that should_skip.
    """
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
    """
    Use Readability to extract main article HTML, then strip out unwanted elements
    (headers, footers, sidebars, language‐selector blocks). Return cleaned text.
    """
    try:
        doc = Document(html)
        main_html = doc.summary()
        soup = BeautifulSoup(main_html, "html.parser")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    # Remove common non‐content selectors
    for sel in [
        "[id*=lang]", "[class*=lang]", "[id*=translat]", "[class*=translat]",
        "[id*=goog]", "[class*=goog]", ".goog-te-banner-frame", ".goog-te-menu-frame",
        "[id*=cookie]", "[class*=cookie]", ".header", ".footer", ".sidebar",
        "header", "footer", "nav", "aside", "form", "script", "style"
    ]:
        for el in soup.select(sel):
            el.decompose()

    main = (
        soup.find("main") or
        soup.find(id="main-content") or
        soup.find("div", class_="content") or
        soup.body
    )
    text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)

    # Filter out leftover language‐selector lines
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

# Regular expression to detect JS opens of file links
JS_OPEN_RE = re.compile(r'''javascript:window\.open\((["'])(.+?\.(pdf|docx|xlsx|doc|xls))\1''', re.IGNORECASE)


# ----------- Link Extraction and Normalization ----------- 
def extract_real_link(base_url, href):
    """
    If href is a JavaScript window.open to a known file extension, extract the URL.
    If href starts with '#', return base_url + that fragment (so #fragment remains).
    Otherwise, resolve href relative to base_url and normalize it.
    """
    if not href:
        return None

    href = href.strip()
    # 1) Handle javascript:window.open(...)
    if href.lower().startswith("javascript:"):
        m = JS_OPEN_RE.search(href)
        if m:
            relative_url = m.group(2)
            if relative_url.lower().endswith((".doc", ".xls")):
                return None
            return urljoin(base_url, relative_url)
        else:
            return None

    # 2) If href is just a "#fragment", keep the fragment
    if href.startswith("#"):
        return urljoin(base_url, href)

    # 3) Otherwise, resolve and normalize
    abs_url = urljoin(base_url, href)
    return normalize_url(abs_url)

# ----------- Chunking and Uploading -----------

def emit_chunks(url, text, last_mod):
    # sliding-window chunking
    windows = []
    if len(text) > MAX_CONTENT_CHARS:
        i = 0
        while i < len(text):
            windows.append(text[i : i + CHUNK_CHARS])
            i += CHUNK_CHARS - CHUNK_OVERLAP
    else:
        windows = [text]

    for idx, chunk in enumerate(windows, 1):
        rec = {
            "id": uuid.uuid4().hex,
            "url": url,
            "title": sanitize(urlparse(url).path),
            "chunk_index": idx,
            "chunk_total": len(windows),
            "content": chunk,
            "last_modified": last_mod,
        }
        filename = blob_name_for_url(url, chunk_index=idx)
        key = (url, idx)
        if key not in seen and upload_json(rec, filename, content_container, last_modified=last_mod):
            with collection_lock:
                documents.append(rec)
                sitemap_urls.append(url)
            seen.add(key)


# ----------- Handlers for Files ----------

def handle_docx(url: str):
    norm_url = normalize_url(url)
    logging.info("Visiting DOCX: %s", norm_url)
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=PDF_DOWNLOAD_TIMEOUT_MS / 1000)
        r.raise_for_status()
        doc = DocxDocument(BytesIO(r.content))
        content = "\n".join(para.text for para in doc.paragraphs)
        if not content.strip():
            return
        last_mod = r.headers.get("Last-Modified", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        # If small, single‐shot; else sliding‐window chunk
        emit_chunks(norm_url, content, last_mod)


    except Exception as e:
        logging.warning(f"Error handling docx: {url} {e}")
        with collection_lock:
            failed.append({"url": url, "reason": f"DOCX parse error: {e}"})

def handle_xlsx(url: str):
    norm_url = normalize_url(url)
    logging.info("Visiting XLSX: %s", norm_url)
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=PDF_DOWNLOAD_TIMEOUT_MS / 1000)
        r.raise_for_status()
        xls = pd.ExcelFile(BytesIO(r.content))
        text_chunks = []
        for sheet in xls.sheet_names:
            df = xls.parse(sheet, dtype=str, na_filter=False)
            # Concatenate all cells into plain text
            rows = df.astype(str).values.tolist()
            for row in rows:
                text_chunks.append(" ".join(row))
        content = "\n".join(text_chunks)
        if not content.strip():
            return
        last_mod = r.headers.get("Last-Modified", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        # If small, single‐shot; else sliding‐window chunk
        emit_chunks(norm_url, content, last_mod)

    except Exception as e:
        logging.warning(f"Error handling xlsx: {url} {e}")
        with collection_lock:
            failed.append({"url": url, "reason": f"XLSX parse error: {e}"})

def handle_pdf(playwright_ctx, url: str):
    norm_url = normalize_url(url)
    logging.info("Visiting PDF: %s", norm_url)

    # Dynamically generate headers
    parsed   = urlparse(url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    referer  = f"{base_url}/"

    pdf_sess = get_domain_session(base_url)
    pdf_sess.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/pdf,*/*;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": referer,
        "Origin": base_url
    })

    retries = Retry(
        total=RETRY_COUNT,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=[403, 429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"]
    )

    pdf_sess.mount("https://", HTTPAdapter(max_retries=retries))
    pdf_sess.mount("http://", HTTPAdapter(max_retries=retries))

    try:
        # Initial visit to base domain to establish cookies if necessary
        pdf_sess.get(base_url, timeout=10).raise_for_status()

        # Check PDF size with HEAD request
        h = pdf_sess.head(url, allow_redirects=True, timeout=10)
        if h.status_code == 200:
            size = int(h.headers.get("Content-Length", 0))
            if size > MAX_PDF_BYTES:
                logging.warning("Skipping large PDF: %s (%d bytes)", norm_url, size)
                return
        else:
            logging.warning("HEAD request returned %d for %s; proceeding to GET", h.status_code, url)

    except Exception as e:
        logging.warning("HEAD error for PDF %s: %s; proceeding to GET", url, e)

    success = False
    for attempt in range(RETRY_COUNT):
        try:
            r = pdf_sess.get(url, allow_redirects=True, timeout=(10, PDF_DOWNLOAD_TIMEOUT_MS / 1000))
            if r.status_code == 200 and "pdf" in r.headers.get("Content-Type", "").lower():
                success = True
                break
            else:
                with collection_lock:
                    failed.append({"url": url, "reason": f"PDF HTTP {r.status_code} (GET)"})
        except Exception as e:
            with collection_lock:
                failed.append({"url": url, "reason": f"PDF GET error: {e}"})
        time.sleep(REQUEST_DELAY * (RETRY_BACKOFF ** attempt))

    if success:
        with collection_lock:
            failed[:] = [f for f in failed if not (f["url"] == url and "(GET)" in f["reason"])]
        reader = PdfReader(BytesIO(r.content))
        raw = "".join(p.extract_text() or "" for p in reader.pages)
        lm = r.headers.get("Last-Modified", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        emit_chunks(norm_url, raw, lm)
        return

    logging.warning("All GET attempts failed; falling back to Playwright download")

    # Fallback Playwright
    # ─── Fallback: use Playwright’s browser download ────────────────────────────
    try:
        # 1) Open a real browser page and establish cookies/JS context
        page = playwright_ctx.new_page()
        page.goto(base_url, timeout=30000, wait_until="networkidle")

        # 2) Trigger the PDF download
        with page.expect_download() as dl:
            page.goto(url, timeout=60000)
        download = dl.value

        # 3) Read the downloaded bytes
        buffer = download.content()
        reader = PdfReader(BytesIO(buffer))
        raw = "".join(p.extract_text() or "" for p in reader.pages)

        # 4) Emit chunks just like in the GET path
        lm = download.suggested_filename  # or grab headers from download if available
        emit_chunks(norm_url, raw, lm)

        page.close()
        return

    except Exception as e:
        logging.warning("Playwright browser download for %s failed: %s", url, e)
        with collection_lock:
            failed.append({"url": url, "reason": f"Playwright download error: {e}"})
        try:
            page.close()
        except:
            pass
        return


def handle_page(playwright_ctx, url: str):
    """
    Visit an HTML page via Playwright, simulate any fragment‐triggered click,
    wait for full network‐idle, strip out any <noscript>, then extract:
      1) All <div id="fragment"> panes for every href="#fragment" found
      2) The main page content itself
    Upload each as its own JSON blob (with fragment in blob name),
    and return filtered links (including full URLs + fragment ones).
    """
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

        # ─── 1) Fetch with retries ─────────────────────────────────────────────────
        for attempt in range(RETRY_COUNT):
            try:
                # Navigate to URL (with fragment if present) and wait for all JS to finish
                resp = pg.goto(url, timeout=PAGE_TIMEOUT_MS, wait_until="networkidle")

                # If there's a fragment, attempt a click to reveal any hidden pane
                parsed = urlparse(url)
                fragment = parsed.fragment
                if fragment:
                    tab_selector = f"a[href='#{fragment}'], #{fragment}"
                    try:
                        pg.click(tab_selector, timeout=5000)
                        if NETWORK_IDLE_WAIT_MS:
                            pg.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_WAIT_MS)
                    except:
                        pass

                if resp.status == 429:
                    time.sleep(REQUEST_DELAY * (RETRY_BACKOFF ** attempt))
                    continue
                if resp.status >= 500:
                    raise PlaywrightTimeoutError(f"{resp.status}")

                html = pg.content()
                headers = resp.headers
                soup = BeautifulSoup(html, "html.parser")

                # Remove <noscript> blocks entirely
                for nos in soup.select("noscript"):
                    nos.decompose()

                # Gather all hrefs from JS-rendered + static DOM
                raw_links = pg.eval_on_selector_all(
                    "a[href]",
                    "els => els.map(e => e.getAttribute('href'))"
                )
                break

            except PlaywrightTimeoutError as e:
                logging.warning("HTML fetch attempt %d failed: %s", attempt + 1, e)
                time.sleep(REQUEST_DELAY * (RETRY_BACKOFF ** attempt))
            except Exception as e:
                logging.warning("HTML fetch attempt %d failed: %s", attempt + 1, e)
                time.sleep(REQUEST_DELAY * (RETRY_BACKOFF ** attempt))

        if soup is None:
            with collection_lock:
                failed.append({"url": url, "reason": "All fetch attempts failed or invalid HTML"})
            return []

        # ─── 2) Build set of all candidate links ─────────────────────────────────
        soup_links = [a.get("href") for a in soup.find_all("a", href=True)]
        all_raw_links = set(raw_links) | set(soup_links)

        # Include any pure fragment hrefs (e.g. "#tab2")
        # They’ll be turned into full URLs via extract_real_link.
        # Now resolve + filter them:
        absolute_links = []
        for l in all_raw_links:
            real = extract_real_link(url, l)
            if real:
                absolute_links.append(real)

        filtered_links = [
            l for l in set(absolute_links)
            if is_allowed_link(l) and not should_skip(l)
        ]
        logging.info(f"Filtered to {len(filtered_links)} crawlable links on {url}")

        if not resp:
            with collection_lock:
                failed.append({"url": url, "reason": "No response from server"})
            pg.close()
            return []

        # ─── 3) Handle HTTP statuses ─────────────────────────────────────────────
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
            pg.close()
            return []

        if resp.status >= 400:
            with collection_lock:
                failed.append({"url": url, "reason": f"HTTP {resp.status}"})
            pg.close()
            return []

        # ─── 4) Extract every fragment-pane's text (for href="#fragment" links) ─
        # Find all unique fragments from filtered_links
        fragments = set()
        for link in filtered_links:
            parsed = urlparse(link)
            if parsed.fragment:
                fragments.add(parsed.fragment)

        for fragment in fragments:
            pane = soup.find(id=fragment)
            if pane:
                pane_text = pane.get_text(separator="\n", strip=True)
                if pane_text.strip():
                    fragment_url = urljoin(url, f"#{fragment}")
                    last_mod = headers.get(
                        "Last-Modified",
                        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    )
                    rec = {
                        "id": uuid.uuid4().hex,
                        "url": fragment_url,
                        "title": sanitize(urlparse(fragment_url).path + fragment),
                        "chunk_index": 1, 
                        "chunk_total": 1,
                        "content": pane_text,
                        "last_modified": last_mod
                    }
                    filename = blob_name_for_url(fragment_url, chunk_index=1)
                    key = (fragment_url, 1)
                    if key not in seen and upload_json(rec, filename, content_container, last_modified=last_mod):
                        with collection_lock:
                            documents.append(rec)
                            sitemap_urls.append(norm_url)
                        seen.add(key)


        # ─── 5) Extract and truncate main page content ───────────────────────────
        rendered_html = str(soup)
        full_text = extract_main_content(rendered_html)

        if len(full_text) > MAX_CONTENT_CHARS:
            snippet = full_text[:MAX_CONTENT_CHARS]
            last = max(
                snippet.rfind("."),
                snippet.rfind("\n"),
                snippet.rfind(" ")
            )
            if last > 0:
                main_text = snippet[: last + 1].strip()
            else:
                main_text = snippet
        else:
            main_text = full_text

        # emit the main page exactly like PDF/DOCX/XLSX
        last_mod = headers.get("Last-Modified", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        emit_chunks(norm_url, main_text, last_mod)


        pg.close()
        return filtered_links

    except Exception as e:
        logging.warning(f"General error handling HTML {url}: {e}")
        with collection_lock:
            failed.append({"url": url, "reason": f"HTML error: {e}"})
        pg.close()
        return []

def enforce_trailing_slash_if_directory(url):
    parsed = urlparse(url)
    # If path has no “.” in last segment, treat it as a directory
    if "." not in os.path.basename(parsed.path):
        if not parsed.path.endswith("/"):
            return url.rstrip("/") + "/"
    return url

# ----------- BFS Crawler (Queue-based, No Duplication) -----------

def bfs_crawl(seed_urls):
    global visited
    queue = deque()

    # Initialize queue with (url, depth) tuples
    for seed_url in seed_urls:
        norm_seed = normalize_url(seed_url)
        if norm_seed not in visited:
            queue.append((seed_url, 0))
            visited.add(norm_seed)

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

        while queue:
            url, depth = queue.popleft()
            norm_url = normalize_url(url)

            if depth > MAX_DEPTH:
                continue
            if should_skip(norm_url) or (RESPECT_ROBOTS and not robot_allows(norm_url, USER_AGENT)):
                logging.debug(f"Skipping {norm_url} by filter or robots.txt")
                continue

            # Dispatch based on file extension
            if norm_url.endswith(".pdf") and INCLUDE_PDFS:
                handle_pdf(ctx, url)
            elif norm_url.endswith(".docx") and INCLUDE_DOCX:
                handle_docx(url)
            elif norm_url.endswith(".xlsx") and INCLUDE_XLSX:
                handle_xlsx(url)
            elif norm_url.endswith(".doc") or norm_url.endswith(".xls"):
                logging.debug(f"Skipping unsupported Office format: {norm_url}")
            else:
                links = handle_page(ctx, url)
                logging.info(f"Crawled: {norm_url} (depth={depth}), found {len(links)} links")

                # Enqueue new, unvisited links by normalized URL
                for link in links:
                    norm_link = normalize_url(link)  # strip fragment for visited-check
                    if norm_link not in visited:
                        visited.add(norm_link)
                        queue.append((link, depth + 1))  # enqueue full link (with fragment)

            time.sleep(REQUEST_DELAY)

        browser.close()

# ----------- Entry Point ------------

def start_crawl():
    try:
        seeds = [u.strip() for u in os_env("BASE_URLS", "").split(";") if u.strip()]
        
        # Dynamically add domains from seeds to ALLOW_DOMAINS
        for s in seeds:
            ALLOW_DOMAINS.add(urlparse(s).netloc.lower().replace("www.", ""))

        urls = []
        for seed in seeds:
            sm_url = seed.rstrip("/") + "/sitemap.xml"
            entries = parse_sitemap(sm_url)
            if entries:
                logging.info("✅ Using sitemap for %s (%d URLs)", seed, len(entries))
                for u in entries:
                    norm_u = normalize_url(u)
                    if norm_u not in visited:
                        urls.append(u)
            else:
                logging.info("⚠️ No sitemap for %s; falling back to seed URL", seed)
                urls.append(seed)

        logging.info("🚀 Starting BFS crawl with %d URLs", len(urls))
        bfs_crawl(urls)

    except Exception as e:
        logging.error(f"Crawl failed: {e}", exc_info=True)

    finally:
        # Save any failures and sitemap URLs to blob storage
        if failed:
            upload_json(failed, "failed_urls.json", logs_container)
        if sitemap_urls:
            upload_json(sitemap_urls, "sitemap.json", logs_container)
        if documents:
            upload_json(documents, "docs.json", logs_container)

        logging.info(
            "Crawl complete. Total URLs Visited: %d, Failed: %d",
            len(visited),
            len(failed)
        )

if __name__ == "__main__":
    start_crawl()

