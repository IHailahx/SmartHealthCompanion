##HealthCompanion

#telegram bot

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

TELEGRAM_TOKEN = "8527483344:AAHg69NPJie_ZHJCrs2bGVoGDCtl7tf84iQ"


# ---------- Telegram handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 أهلاً! اكتب سؤالاً صحياً (مثل: ما أعراض ارتفاع ضغط الدم؟) وسأبحث في قاعدة المعرفة."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = (update.message.text or "").strip()
    if not query:
        return

    await update.message.reply_text("⏳ جارٍ البحث عن إجابة من المصادر المسجَّلة...")

    try:
        results = rag_query(query, collection, model, reranker, top_k=5, min_score=0.25)
    except Exception as e:
        print("RAG query failed:", e)
        await update.message.reply_text("❌ حدث خطأ أثناء معالجة سؤالك.")
        return

    if not results:
        await update.message.reply_text("لم أجد معلومات كافية لسؤالك. جرّبي صياغة ثانية 🙏")
        return

    parts = []
    for score, doc, meta in results[:3]:
        snippet = doc.strip()
        if len(snippet) > 600:
            snippet = snippet[:600] + "..."

        src = ""
        if isinstance(meta, dict):
            source = meta.get("source")
            page = meta.get("page")
            if source:
                src += f"\n🔎 المصدر: {source}"
            if page:
                src += f" (صفحة {page})"

        parts.append(f"📌 {snippet}{src}")

    reply = "\n\n────────\n\n".join(parts)
    await update.message.reply_text(reply)


def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 RAG Telegram bot running...")
    app.run_polling() 





# OCR Reader for PDF with unreadable text
import easyocr
from PIL import Image
print("🔍 Initializing EasyOCR (ar + en)...")
easyocr_reader = easyocr.Reader(['ar', 'en'], gpu=True)


# --------------------------------------------------------
# Models and Vector DB setup
# --------------------------------------------------------
import numpy as np
import pandas as pd
import uuid
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
import time
from sentence_transformers import CrossEncoder
import shutil


print(torch.cuda.is_available())
print("CUDA available? ", torch.cuda.is_available())
print("GPU count: ", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

#Hailah  --  if not GPU use CPU
# Embedding model
def load_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    
    return SentenceTransformer("intfloat/multilingual-e5-small", device=device)

# Reranker model
def load_reranker_model(device="cuda"):
    reranker = CrossEncoder(
        "BAAI/bge-reranker-base",
        device=device,
        trust_remote_code=True
    )
    return reranker

# Initialize the ChromaDB
import chromadb


def init_chroma_collection(path=r"C:\Users\haila\OneDrive\Desktop\Project\HealthCompanionBot\data", name="healthcare_rag"):
  client = chromadb.PersistentClient(path=path)
  collection = client.get_or_create_collection(
      name=name, metadata={"hnsw:space":"cosine"} # ideal for normalized embeddings
   )
  print(f"📦 ChromaDB collection ready → {name}")
  print (collection.count())
  return collection



# Store the chunks in ChromaDB
def store_chunks_in_chroma(chunks, collection, embed_model, batch_size=128):

    print(f"\n🚀 Storing {len(chunks)} chunks with batch size = {batch_size}")

    texts = [c["content"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metas = [c["metadata"] for c in chunks]


    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_metas = metas[i:i+batch_size]
        ## Empedding
        embeddings = embed_model.encode(batch_texts, show_progress_bar=False)

        collection.add(
            embeddings=np.array(embeddings).tolist(),
            documents=batch_texts,
            metadatas=batch_metas,
            ids=batch_ids
        )

        print(f"   → Stored {len(batch_ids)} chunks")

# Save chunks in JSONL format
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n✅ Saved {len(data)} chunks → {path}")


    # --------------------------------------------------------
# Preprocessing Phase
# --------------------------------------------------------

import re
import unicodedata

# Arabic Base
ARABIC_BASE = (
    r"\u0600-\u06FF"   # Arabic core (letters, digits, punctuation)
    r"\u0750-\u077F"   # Arabic Supplement
    r"\u08A0-\u08FF"   # Arabic Extended
)

# Arabic diacritics only
RE_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)

# Tatweel
RE_TATWEEL = re.compile(r"\u0640")

# All character not Arabic or basic punctuation or whitespace
RE_NON_ARABIC = re.compile(
    fr"[^{ARABIC_BASE}\s\.,؛،؟?!\-\/]"
)

# Fix excessive repeated chars (3+ → 1) — strictly for OCR noise
RE_REPEAT_FIX = re.compile(r"(.)\1{2,}")

# Collapse multiple spaces
RE_MULTI_SPACE = re.compile(r"\s+")


def clean_pdf_text(text: str) -> str:
    text = text.replace("\n", " ") # replace new lines with space
    text = re.sub(r"\s{2,}", " ", text) # remove extra spaces
    return text.strip()


def normalize_presentation_forms(text: str) -> str:
    """
    Converts Arabic contextual forms (ﻟﻼ, ﺍ, ﻫ, ﻲ…)
    into their canonical Unicode letters without changing meaning.
    Handle the issue in Arabic characters in PDFs and URLs
    """
    return unicodedata.normalize("NFKC", text)

def normalize_arabic_text(text: str) -> str:
    """
    STRICT Arabic text normalization.
    - Removes noise
    - Preserves meaning
    - Avoids destructive letter substitutions
    """

    # 1) Normalize contextual forms (no letter identity change)
    text = normalize_presentation_forms(text)

    # 2) Remove diacritics (tashkeel) ONLY
    text = RE_DIACRITICS.sub("", text)

    # 3) Remove tatweel (ـ)
    text = RE_TATWEEL.sub("", text)

    # 4) Remove all non-Arabic characters (foreign letters, noise)
    text = RE_NON_ARABIC.sub(" ", text)

    # 5) Fix repeated glyphs (OCR artifacts)
    text = RE_REPEAT_FIX.sub(r"\1", text)

    # 6) Normalize whitespace
    text = RE_MULTI_SPACE.sub(" ", text).strip()

    return text

# Fix issues in URL Content
def clean_url_arabic(text):
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text) #remove control charecters
    text = re.sub(r"\b\d{7,}\b", " ", text) #remove long numbers
    text = re.sub(r"\s{2,}", " ", text) #remove multiple spaces
    return text.strip() #remove the spaces in starter and end 

# --------------------------------------------------------
# PDF Loader
# --------------------------------------------------------

def extract_pdf_page_text(page, ocr_reader=easyocr_reader, dpi=300):
    """
    1) Try PyMuPDF text extraction first.
    2) If page contains no text or PyMuPDF fails → do OCR using EasyOCR.
    """

    # ---------- Strategy 1: PyMuPDF text ----------
    try:
        txt1 = page.get_text("plain") or ""
        txt2 = page.get_text("blocks") or ""
        txt3 = page.get_text("layout") or ""

        # Combine these
        combined = " ".join([
            txt1,
            " ".join(b[4] for b in txt2) if isinstance(txt2, list) else "",
            txt3
        ]).strip()

        # If PyMuPDF finds real text, return it
        if combined and len(re.findall(r"[ء-ي]", combined)) > 5:
            return combined

    except Exception:
        pass  # PyMuPDF failed → will fallback to OCR


    # ---------- Strategy 2: OCR Fallback ----------
    try:
        # Render page as high-resolution image
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Convert to array
        img_np = np.array(img)

        # Run OCR
        ocr_result = ocr_reader.readtext(img_np, detail=0)

        ocr_text = " ".join(ocr_result).strip()

        if ocr_text:
            return ocr_text

    except Exception as e:
        print(f"⚠️ OCR failed on page {page.number+1}: {e}")
        return ""

    return ""

# Semantic chunking for PDFs
def semantic_chunk_text(text, chunk_size=600, overlap=150):
    """
    Splits text into semantic chunks using sentence boundaries + overlap.
    """

    # Split by major Arabic punctuation
    sentences = re.split(r"[\.؟!\n]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) < chunk_size:
            current += " " + sent
        else:
            chunks.append(current.strip())
            # add overlap
            current = current[-overlap:] + " " + sent

    if current.strip():
        chunks.append(current.strip())

    return chunks

# PDF Main load function
def chunk_pdf_pages(
    input_pdf_path,
    output_jsonl_path=None,
    normalize=True,
    min_chars=15,
    chunk_size=600,
    overlap=150
):
    pdf = fitz.open(input_pdf_path)
    chunks = []
    doc_name = Path(input_pdf_path).stem

    print(f"\n📘 Processing PDF: {doc_name}")
    print(f"Total pages: {len(pdf)}")
    #page by page
    for i in range(len(pdf)):
        page = pdf[i]

        # Extract via hybrid method (text or OCR) 
        raw_text = extract_pdf_page_text(page)

        if not raw_text or len(raw_text.strip()) < min_chars:
            print(f"⚠️ Page {i+1}: No extractable text → Skipped")
            continue

        # Clean before normalize
        cleaned = clean_pdf_text(raw_text)

        if normalize:
            cleaned = normalize_arabic_text(cleaned)

        arabic_chars = re.findall(r"[ء-ي]", cleaned)
        if len(arabic_chars) < 10:
            print(f"⚠️ Page {i+1}: Insufficient Arabic content → Skipped")
            continue

        # ---------- Semantic Chunking ----------
        page_chunks = semantic_chunk_text(
            cleaned,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for j, ch_text in enumerate(page_chunks, start=1):
            cid = f"{doc_name}_page_{i+1}_chunk_{j}"

            chunks.append({
                "id": cid,
                "page_number": i+1,
                "chunk_index": j,
                "content": ch_text,
                "metadata": {
                    "source": doc_name,
                    "type": "PDF_page",
                    "page": i+1
                }
            })

        print(f"✅ Page {i+1}: {len(page_chunks)} semantic chunks created")

    # Save JSONL if requested
    if output_jsonl_path:
        save_jsonl(chunks, output_jsonl_path)

    print(f"\n📦 Completed PDF extraction: {len(chunks)} chunks generated")
    return chunks

# ============================================================
#  MAQA Loader
# ============================================================
#Hailah ---- اشوف انه افضل نشيله لان chunk_pdf_pages تعوض عنها 
def chunk_maqa_dataset(input_xlsx_path, output_jsonl_path, normalize=True):
    df = pd.read_excel(input_xlsx_path)

    required_cols = ["q_body", "a_body", "category"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column missing: {col}")

    df = df.dropna(subset=required_cols)
    chunks = []

    for idx, row in df.iterrows():
        q = row["q_body"]
        a = row["a_body"]
        c = row["category"]

        if normalize:
            q = normalize_arabic_text(q)
            a = normalize_arabic_text(a)
            c = normalize_arabic_text(c)

        chunk_text = f"سؤال: {q}\nالتصنيف: {c}\nالإجابة: {a}"

        chunks.append({
            "id": f"maqa_{idx}",
            "question": q,
            "answer": a,
            "category": c,
            "content": chunk_text,
            "metadata": {"source": "MAQA", "type": "Q&A"}
        })

    save_jsonl(chunks, output_jsonl_path)
    return chunks

# ============================================================
#  URL Loader
# ============================================================

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import tldextract
import re
from collections import deque
from pathlib import Path
import fitz
import time
import json
import hashlib


# CONFIGURATION

def make_uid(url: str, extra: str = ""):
    base = url + extra
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    return f"{h}"

BASE_DOMAIN = "moh.gov.sa"
BASE_ROOT = "https://www.moh.gov.sa/awarenessplateform/"

EXCLUDE_PREFIXES = [
    "/awarenessplateform/EducationalSeries",
    "/awarenessplateform/PublishingImages",
    "/_layouts/",
    "/Style%20Library/",
    "/SiteAssets/",
    "/Images/",
]

VALID_CONTENT_TYPES = ["text/html", "application/xhtml+xml"]

PDF_EXT = (".pdf",)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Linux; X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/120.0"
}


# URL NORMALIZATION + FILTERING

def canonicalize_url(url):
    """Normalize and keep ?PageIndex; remove anchors."""
    parsed = urlparse(url)
    query = f"?{parsed.query}" if parsed.query else ""
    clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}{query}"
    clean = clean.split("#")[0]
    return clean


def is_valid_internal(url: str) -> bool:
    """Filter internal AwarenessPlatform URLs except excluded ones."""
    parsed = urlparse(url)

    # Must be inside moh.gov.sa domain
    if parsed.netloc.lower() not in [BASE_DOMAIN, f"www.{BASE_DOMAIN}"]:
        return False

    path = parsed.path.lower()

    # Must be inside awareness platform
    if "/awarenessplateform/" not in path:
        return False

    # Exclusions
    for ex in EXCLUDE_PREFIXES:
        if path.startswith(ex.lower()):
            return False

    return True


def is_pdf(url):
    return url.lower().endswith(".pdf")


# SAFE HTML TEXT EXTRACTION

def clean_text(t):
    t = re.sub(r"[\x00-\x1F\x7F]", " ", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()

#Hailah - english and arabic should we make it only for arabic
##BeautifulSoup
def extract_arabic_text(html):
    soup = BeautifulSoup(html, "html.parser")

    for bad in ["script", "style", "svg", "iframe"]:
        for t in soup.find_all(bad):
            t.decompose()

    txt = soup.get_text(" ", strip=True)
    txt = clean_text(txt)

    if len(txt) < 40:
        return ""

    return txt

def extract_main_content(html):
    soup = BeautifulSoup(html, "html.parser")

    # 1) Remove all irrelevant sections
    for bad in ["script", "style", "header", "footer", "nav", "svg", "iframe"]:
        for t in soup.find_all(bad):
            t.decompose()

    # 2) Try known SharePoint content blocks
    candidates = [
        # Most MOH pages use this
        soup.select_one(".ms-rtestate-field"),

        # Alternative block
        soup.select_one(".article__body"),

        # ID used in some categories
        soup.select_one("#ctl00_PlaceHolderMain_ContentMain"),

        # Fallback main container for some templates
        soup.select_one(".contentPage"),

        # Some pages wrap content inside article tags
        soup.find("article"),
    ]

    for c in candidates:
        if c and c.get_text(strip=True):
            text = c.get_text(" ", strip=True)
            return clean_text(text)

    # 3) Fallback: extract <main> tag if exists
    main_tag = soup.find("main")
    if main_tag:
        return clean_text(main_tag.get_text(" ", strip=True))

    # 4) Final fallback (worst case)
    return clean_text(soup.get_text(" ", strip=True))

# PDF EXTRACTOR

def extract_pdf(url):
    print(f"📄 PDF → {url}")

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except:
        print("❌ PDF download failed")
        return []

    try:
        pdf = fitz.open(stream=resp.content, filetype="pdf")
    except:
        print("❌ PDF parse failed")
        return []

    out = []
    name = Path(urlparse(url).path).stem

    for i, page in enumerate(pdf):
        try:
            text = page.get_text()
        except:
            continue

        if not text or len(text.strip()) < 20:
            continue

        out.append({
		"id": make_uid(url, f"_pdf_{i+1}"),
		"content": text.strip(),
		"metadata": {
		"type": "PDF_page",
        "source": name,
        "page": i + 1,
        "page_url": url}})

    return out


# SHAREPOINT API FIX — ChronicDisease Hidden Pages

PROCESS_QUERY_URL = (
    "https://www.moh.gov.sa/awarenessplateform/ChronicDisease/_vti_bin/client.svc/ProcessQuery"
)

SHAREPOINT_LIST_ID = "{35500486-8cc5-4e1d-bcbe-8551f0fc78c1}"


def fetch_chronic_disease_hidden_pages():
    """Fetch JS-rendered pages not visible in HTML."""
    print("\n🔍 Fetching SharePoint ChronicDisease hidden pages...")

    body = f"""
<Request AddExpandoFieldTypeSuffix="true"
         SchemaVersion="15.0.0.0"
         LibraryVersion="16.0.0.0"
         ApplicationName="PythonCrawler"
         xmlns="http://schemas.microsoft.com/sharepoint/clientquery/2009">
  <Actions>
    <ObjectPath Id="1" ObjectPathId="0" />
    <Query Id="2" ObjectPathId="0">
      <Query SelectAllProperties="true">
        <Properties />
      </Query>
      <ChildItemQuery SelectAllProperties="true">
        <Properties />
      </ChildItemQuery>
    </Query>
  </Actions>
  <ObjectPaths>
    <List Id="0" ParentId="3" ObjectPathId="4" TypeId="{SHAREPOINT_LIST_ID}" />
    <StaticMethod TypeId="3" Name="Current" Id="5" />
  </ObjectPaths>
</Request>
"""

    try:
        resp = requests.post(PROCESS_QUERY_URL, data=body, headers={"Content-Type": "text/xml"})
        raw = resp.text
    except:
        print("❌ SharePoint API failed")
        return []

    matches = re.findall(
        r"(/awarenessplateform/ChronicDisease/Pages/[A-Za-z0-9\-_]+\.aspx)",
        raw
    )

    urls = [urljoin("https://www.moh.gov.sa", u) for u in matches]
    urls = sorted(set(urls))

    print(f"✅ SharePoint extra pages: {len(urls)} found")
    return urls


# DYNAMIC LINK EXTRACTOR

def extract_links(soup, base):
    found = set()

    # Normal <a>, <img>, <iframe>
    for tag in soup.find_all(["a", "iframe"]):
        href = tag.get("href")
        if not href:
            continue
        absolute = urljoin(base, href)
        absolute = canonicalize_url(absolute)

        if is_valid_internal(absolute):
            found.add(absolute)

    # Inline JS patterns
    html = str(soup)
    js_links = re.findall(r"'(https?://[^']+)'", html)
    for link in js_links:
        link = canonicalize_url(link)
        if is_valid_internal(link):
            found.add(link)

    # Capture DVWP href entries (SharePoint)
    dvwp_links = re.findall(r'href="([^"]+)"', html)
    for link in dvwp_links:
        absolute = canonicalize_url(urljoin(base, link))
        if is_valid_internal(absolute):
            found.add(absolute)

    return found


# MAIN CRAWLER

def crawl_all_awareness(max_pages=1500, delay=0.15):
    visited = set()
    queue = deque()

    # Start from top
    queue.append(BASE_ROOT)

    # Add hidden ChronicDisease pages
    for url in fetch_chronic_disease_hidden_pages():
        queue.append(url)

    pages = []

    print("\n🌍 Starting AwarenessPlatform crawl...\n")

    while queue and len(visited) < max_pages:
        url = canonicalize_url(queue.popleft())

        if url in visited:
            continue
        visited.add(url)

        print(f"\n🔎 Visiting: {url}")

        # PDF
        if is_pdf(url):
            pdf_chunks = extract_pdf(url)
            pages.extend(pdf_chunks)
            continue

        # Fetch HTML
        try:
            resp = requests.get(url, headers=HEADERS, timeout=12)
        except:
            print("⚠ Request failed")
            continue

        ctype = resp.headers.get("Content-Type", "").lower()

        if not any(t in ctype for t in VALID_CONTENT_TYPES):
            print("⚠ Not HTML, skipped")
            continue

        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        # Extract text
        text = extract_main_content(html)
        if text:
            pages.append({
			"id": make_uid(url),
			"url": url,
			"content": text,
			"metadata": {
			"type": "web_page",
			"source": "MOH_AwarenessPlateform",
			"page_url": url}})
            print("📚 Stored text")
        else:
            print("⚠ Low-content page, skipped")

        # Extract links
        new_links = extract_links(soup, url)

        print(f"🔗 Found {len(new_links)} links")

        for link in new_links:
            if link not in visited:
                queue.append(link)

        time.sleep(delay)

    print(f"\n🎉 Crawl completed → {len(pages)} items extracted.\n")
    return pages

# ============================================================
#  RAG Query
# ============================================================

def rag_query(query, collection, model, reranker, top_k=10, min_score=0.25):
    """
    Returns only high-relevance chunks with reranking and hallucination filtering.
    """

    # STEP 1 — Embed query
    query_emb = model.encode([query], normalize_embeddings=True).tolist()[0]

    # STEP 2 — Vector search (gets rough candidates)
    search = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k * 3,   # get extra candidates to rerank
        include=["metadatas", "documents"]
    )

    docs = search["documents"][0]
    metas = search["metadatas"][0]

    # Remove exact duplicates
    unique = {}
    for d, m in zip(docs, metas):
        if d not in unique:
            unique[d] = m

    docs = list(unique.keys())
    metas = list(unique.values())

    # STEP 3 — Build reranker input
    pairs = [[query, d] for d in docs]
    scores = reranker.predict(pairs)

    # STEP 4 — Combine & sort
    ranked = sorted(zip(scores, docs, metas), key=lambda x: x[0], reverse=True)

    # STEP 5 — Filter out junk low-score items
    filtered = [(s, d, m) for (s, d, m) in ranked if s >= min_score]

    # If everything is filtered out → still return top-1
    if not filtered:
        filtered = [ranked[0]]

    # FINAL — return best top_k
    return filtered[:top_k]

# ============================================================
#  Initialize models
# ============================================================

if __name__ == "__main__":
  model = load_embedding_model()
  reranker = load_reranker_model()
  collection = init_chroma_collection()
  
  # ============================================================
#  Load URLs
# ============================================================
url_chunks = crawl_all_awareness()
store_chunks_in_chroma(url_chunks, collection, model, batch_size=128)
save_jsonl(url_chunks, r"C:\Users\haila\OneDrive\Desktop\Project\HealthCompanionBot\data\awareness_output.jsonl")

# ============================================================
#  Load PDF
# ============================================================
output_jsonl__pdf_path = (r"C:\Users\haila\OneDrive\Desktop\Project\HealthCompanionBot\data\pdf_output.jsonl")
pdf_chunks = chunk_pdf_pages(r"C:\Users\haila\OneDrive\Desktop\Project\HealthCompanionBot\Dose-fo-Awareness-2.pdf", output_jsonl__pdf_path,normalize=True)
store_chunks_in_chroma(pdf_chunks, collection, model, batch_size=64)
save_jsonl(pdf_chunks, "pdf_output.jsonl")

# ============================================================
#  Query Test
# ============================================================
#Hailah
#query = "ما أعراض إرتفاع ضغط الدم؟"
#results = rag_query(query, collection, model, reranker, top_k=10)

# ============================================================
#  Preview JSONL files
# ============================================================
def preview_jsonl(path, n=5, fields=None):
    """
    Preview the first N items in a JSONL file.

    Args:
        path (str): Path to .jsonl file
        n (int): Number of rows to preview
        fields (list or None): If provided, only show these fields
    """
    print(f"\n📄 Previewing first {n} entries from: {path}\n")

    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if count >= n:
                break

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print("❌ Skipped malformed JSON line.")
                continue

            if fields:
                filtered = {k: item.get(k) for k in fields}
                print(json.dumps(filtered, ensure_ascii=False, indent=2))
            else:
                print(json.dumps(item, ensure_ascii=False, indent=2))

            print("-" * 60)
            count += 1

    if count == 0:
        print("⚠️ No readable entries found.")
def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # Remove leading/trailing whitespace
            if line:  # Ensure the line is not empty
                try:
                    json_object = json.loads(line)
                    data.append(json_object)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line}. Error: {e}")
    return data

# Test Preview JSONL
preview_jsonl(r"C:\Users\haila\OneDrive\Desktop\Project\HealthCompanionBot\data\awareness_output.jsonl", n=20)

# Test Read JSONL
data = read_jsonl_file(r"C:\Users\haila\OneDrive\Desktop\Project\HealthCompanionBot\data\awareness_output.jsonl")
for i , data in enumerate(data):
  if 'url' in data and data['url'] == "https://www.moh.gov.sa/awarenessplateform/ChronicDisease/Pages/Hypertension.aspx":
    print(data)


#Hailah
# Loop through the results
# for i, (score, doc, meta) in enumerate(results, start=1):
#   print(i, score, meta.get("page_url"), meta.get("source"), doc[:1000])

#   # Top matching results and scores
# for score, doc, meta in results[:5]:
#     print("\nSCORE:", float(score))
#     print("SOURCE:", meta.get("source"))
#     print("TEXT:", doc[:400])
#     print("-" * 60)

    # ============================================================
#  LLM Setup
# ============================================================

import os
# Remove any cached keys
if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]

os.environ["OPENAI_API_KEY"] = "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"

print("API key updated.")
from openai import OpenAI
client = OpenAI()

# ============================================================
#  Generate response from LLM
# ============================================================

def generate_rag_answer_with_citations(query, results):
    """
    Hallucination-resistant answer generator using only retrieved chunks.
    """

    # ============================
    # LIMIT CHUNKS TO TOP-1 or TOP-2
    # ============================
    max_chunks = 2
    results = results[:max_chunks]

    # ============================
    # FORMAT CONTEXT + CITE SOURCES
    # ============================
    docs_context = []
    source_map = {}
    next_id = 1

    for i, (score, doc, meta) in enumerate(results, start=1):

        # —— Determine citation label
        if meta.get("source") == "MAQA":
            label = "MAQA"

        elif meta.get("type") == "PDF_page":
            name = meta.get("source", "PDF")
            label = f"PDF:{name}"

        elif meta.get("type") == "web_page":
            url = meta.get("page_url") or "UNKNOWN_URL"
            label = f"URL:{url}"

        else:
            label = meta.get("source", "UNKNOWN")

        # —— Assign citation ID if new
        if label not in source_map:
            source_map[label] = next_id
            next_id += 1

        cid = source_map[label]

        # —— Build context fed to LLM
        docs_context.append(f"[{cid}] {doc}")

    context_text = "\n\n".join(docs_context)

    # ============================
    # ANTI-HALLUCINATION PROMPT
    # ============================
    prompt = f"""
أنت مساعد طبي محترف، ودورك هو الإجابة فقط وفق النصوص المعطاة أدناه.

❗ **قواعد صارمة يجب اتباعها:**
1. لا تذكر أي معلومة غير موجودة داخل النصوص.
2. إذا لم تُجب النصوص عن السؤال، قل: "المعلومات المتوفرة لا تجيب عن السؤال مباشرة."
3. استخدم فقط الاستشهادات الخاصة بالنصوص مثل [1].
4. امزج النصوص المتاحة دون إضافة معرفة خارجية.
5. لا تضف أي مصادر إضافية غير التي تظهر في المقاطع
6. أجب بأنه لا يتوفر لديك معلومات إذا لم تجد إجابة للسؤال.


السؤال:
{query}

النصوص المتاحة:
{context_text}

الجواب (مع الاستشهادات):
"""

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    answer = completion.choices[0].message.content.strip()

    # ============================
    # BUILD CLEAN SOURCE LIST
    # ============================
    source_list = "### 📚 المصادر المستخدمة:\n"
    for label, cid in source_map.items():

        if label.startswith("URL:"):
            url = label.replace("URL:", "")
            source_list += f"- [{cid}] رابط: {url}\n"

        elif label.startswith("PDF:"):
            name = label.replace("PDF:", "")
            source_list += f"- ملف PDF: {name} [{cid}]\n"

        elif label == "MAQA":
            source_list += f"- MAQA [{cid}]\n"

        else:
            source_list += f"- {label} [{cid}]\n"

    return answer + "\n\n" + source_list

# ============================================================
#  Test RAG
# ============================================================
#Hailah 
#final_answer = generate_rag_answer_with_citations(query, results)
#print(final_answer)
main()
