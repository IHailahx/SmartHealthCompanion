# ============================================================
#  Installations
# ============================================================

# ============================================================
#  Package Import
# ============================================================
import easyocr
from PIL import Image
import numpy as np
import torch
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import chromadb
import re
import unicodedata
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
from pathlib import Path
import fitz
import time
import hashlib
import os
import nest_asyncio
import requests
nest_asyncio.apply()
from openai import OpenAI
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Evaluation packages
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional


# ============================================================
#  Helper Functions
# ============================================================

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n✅ Saved {len(data)} chunks → {path}")

def preview_jsonl(path, n=5, fields=None):
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
    import json # Local import to prevent name shadowing
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    json_object = json.loads(line)
                    data.append(json_object)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line}. Error: {e}")
    return data

# --------------------------------------------------------
# OCR, Embedding and Reranker Models and Vector DB setup
# --------------------------------------------------------

# --------------------------
# OCR
# --------------------------

print("🔍 Initializing EasyOCR (ar + en)...")
easyocr_reader = easyocr.Reader(['ar', 'en'], gpu=True)

# --------------------------
# Load Embedding Model (1024-dim)
# --------------------------
def load_embedding_model():
    model = SentenceTransformer(
        "jinaai/jina-embeddings-v3",
        trust_remote_code=True
    )
    return model

# --------------------------
# Load Reranker Model
# --------------------------
def load_reranker_model(device="cuda"):
    model_name = "Alibaba-NLP/gte-multilingual-reranker-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(device)

    return tokenizer, model

# --------------------------
# Initialize Chroma
# --------------------------

# ---------------------------------
# chromadb: a vector database with the core components id, content (i.e. documents), and metadata.
# ID (ids): A [unique] string identifier for each data record within a collection.
# Content (documents): The raw data (usually text) that you want to store and embed. Documents are converted into numerical vector embeddings using an embedding function.
# Metadata (metadatas): A dictionary of additional information associated with each document, used for filtering, categorizing, and providing context.
# ---------------------------------
def init_chroma_collection(path=r"C:\Users\haila\OneDrive\Desktop\Project\AI_Tools_Final\Data\chroma_store", name="healthcare_rag_v2"):

    client = chromadb.PersistentClient(path=path)

    # NEW collection name to avoid old schema!
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None
    )

    print(f"📦 Fresh ChromaDB collection ready → {name}")
    return collection

# --------------------------
# Store Chunks in Chroma
# --------------------------
def store_chunks_in_chroma(chunks, collection, embed_model, batch_size=32):

    print(f"\n🚀 Storing {len(chunks)} chunks with batch size = {batch_size}")

    texts = [c["content"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metas = [c["metadata"] for c in chunks]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids   = ids[i:i + batch_size]
        batch_metas = metas[i:i + batch_size]

        # Skip empty or None content
        valid_batch = [
            (t, idx, meta)
            for t, idx, meta in zip(batch_texts, batch_ids, batch_metas)
            if t and t.strip()
        ]

        if not valid_batch:
            continue

        batch_texts, batch_ids, batch_metas = zip(*valid_batch)

        # ==== Jina v3 embedding (correct call) ====
        embeddings = embed_model.encode(
            list(batch_texts),
            task="retrieval.passage",
            batch_size=1,            # best for Jina v3 on Colab
            convert_to_numpy=True
        )

        # ==== Store in Chroma ====
        collection.add(
            embeddings=embeddings.tolist(),
            documents=list(batch_texts),
            metadatas=list(batch_metas),
            ids=list(batch_ids)
        )

        print(f"   → Stored {len(batch_ids)} chunks")

# --------------------------------------------------------
# Preprocessing Phase
# --------------------------------------------------------

# --------------------------------------------------------
# Arabic Normalization
# --------------------------------------------------------

# Arabic Base
ARABIC_BASE = (
    r"\u0600-\u06FF"   # Arabic core (letters, digits, punctuation)
 )


# harakat
RE_DIACRITICS = re.compile(
    r"[\u064B-\u065F]"
)

# Tatweel
RE_TATWEEL = re.compile(r"\u0640")

# All character not Arabic or basic punctuation or whitespace
RE_NON_ARABIC = re.compile(
    fr"[^{ARABIC_BASE}\u0660-\u0669\u0030-\u0039\s\.,؛،؟?!\-\/]"
)

# Collapse multiple spaces
RE_MULTI_SPACE = re.compile(r"[ ]{2,}")

# Collapse multiple lines
RE_MULTI_NEWLINE = re.compile(r"\n+")


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

    # 6) Normalize whitespace
    text = RE_MULTI_SPACE.sub(" ", text).strip()

    # 6) Normalize newlines
    text = RE_MULTI_NEWLINE.sub("\n", text)

    return text


# --------------------------------------------------------
# Check string lines that are likely to be headers.
# first, remove duplicated spaces, and find string with multiple words and a total number of characters between 6 and 80
# if found, look for expected heading characters (words or digits)
# if not found, look for string with no ending punctation, less than 60 character length, and few inner punctations
# if not found, then the line is most likely not a header
# --------------------------------------------------------

def is_probable_arabic_heading(line: str) -> bool:

    s = re.sub(r"\s+", " ", line).strip() # remove the multiple spaces and replace them with one space. If single word, no space will be present (strip)
    if not s: # if string is empty, return false
        return False

    if 6 <= len(s) <= 80 and s.count(" ") <= 10: # test if the line string is short and has between 1 and 10 spaces
        # Common Arabic heading keywords
        if re.search(r"\b(الفصل|الباب|المبحث|المطلب|المسألة|تمهيد|مقدمة|خاتمة|الفرع|الجزء)\b", s):
            return True
        if re.match(r"^\s*(أولاً|ثانياً|ثالثاً|رابعاً|خامساً|سادساً|سابعاً|ثامناً|تاسعاً|عاشراً)\b", s):
            return True

        # All-caps doesn't apply in Arabic; but "title-like" lines sometimes end without punctuation
        if not re.search(r"[\.؟!]\s*$", s) and len(s) <= 60: # look for line strings that don't end with punctuation and contains less that 60 characters
            # If it has few punctuation marks and looks like a title line
            punct = re.findall(r"[:]", s) # extract punctations to further test the text validity as header, if less than 3 punctuations, then it is most likely a header
            if len(punct) == 1:
                return True
    return False

# --------------------------------------------------------
# Remove spaces present before or after newlines
# split text by punctation [. ؟ ! ؛]
# create a list of the splitted paragraph
# --------------------------------------------------------
def split_arabic_sentences(text: str):

    # Normalize whitespace before splitting
    text = re.sub(r"\s+\n", "\n", text) # remove any space before newlines
    text = re.sub(r"\n\s+", "\n", text) # remove any space after newlines
    text = re.sub(r"[ \t]+", " ", text) # remove tabs and replace them with space

    parts = re.split(r"[\.؟!\n]+", text) # split sentence using boundaries: . ؟ ! and Arabic semicolon "؛"
    return [p.strip() for p in parts if p and p.strip()] # remove additional spaces on any side of the splitted text

# --------------------------------------------------------
# First, check if (text) contain string, if yes, split the (text) by line and append it to the list (lines).
# Next, create a set (heading_lines) for headings and add the elements of (lines) that are most likely heading lines

# second, create an empty list (sentences) that contains the elements from (text) splitted by punctation
# Next, create an empty list (normalized_sents) to should contain splitted strings with less than or equal 350 characters

## Important output : normalized_sents & heading_lines ##

# third, create an empty list (chunks) for the final text chuck to be stored in chromadb
# Also, create a placeholder (current_sent) to incrementally add text from (normalized_sents) until we reach threshold 600 characters
# Also, create a placeholder (current_len) to track the length of text in the placeholder (current_sent)

# fourth, loop through each (normalized_sents) element (sent), if the element (sent) is not empty, check if it identified as heading line
# or present in the previously defined list (heading_lines), if present, append (current_sent) if exist to (chunks) and start a new chunk
# by resetting the (current_sent) and (current_len) variables.
# Next, start the (current_sent) with the heading line and assign the (current_len) the length of the (current_sent)
## ![Even if the current_sent didn't reach 600 yet, once we hit heading line, we will start new chunk] ##

# fifth, if the (sent) is not a heading line, it must be appended to the existing (current_len). Calculate the value len(sent) + len(current_len)
# if the result is less than 600, append (sent) to (current_sent) and update the length, else, flush the chunk and start a new chunk with (sent)
# Next, apply the overlap function to keep the context of text, however, if applying overlap will make the sentence exceed 600, then remove the overlap
# and just append the (sent) to empty (current_sent) and reset the (current_len)
## ![overlap is only applied when a chunk overflows]

# Finally, flush the (chunks) to reset variables for the next run
# --------------------------------------------------------

def semantic_chunk_text_v2(
    text: str,
    chunk_size: int = 300, # maximum full chunk size
    overlap_sentences: int = 2, # number of sentences to overlap
    max_sentence_len: int = 150, # maximum sentence length
):

    if not text or not text.strip(): # check if the text sent for chunking is empty, null, or consists of only whitespace
        return []

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]# split the text by line and remove proceeding and preceeding spaces

    heading_lines = set() # create set for headings

    for ln in lines: # for each line, test if it was a heading line
        if is_probable_arabic_heading(ln):
            heading_lines.add(ln)

    sentences = split_arabic_sentences(text) # create splitted list using punctations [this includes heading, we will split heading later using the heading_lines set]

    normalized_sents = [] # create a list to contain text splits with size less than or equal to the maximum sentence length
    for s in sentences: # for each item in the list of text splitted by punctations
        s = s.strip()
        if len(s) <= max_sentence_len: # check the length, if it is less than the maximum sentence size, append it to the list
            normalized_sents.append(s)
        else:
            subparts = re.split(r"[،,]+", s) # if the length is more than 350 characters, look for commas, and split by comma
            subparts = [sp.strip() for sp in subparts if sp.strip()] # create a list of the new splitted long text
            if len(subparts) <= 1: # if the number of items in the list is zero or 1 [zero is produced if the substring contains empty since we use strip()]
                for k in range(0, len(s), max_sentence_len):  # recursively split and append 350 characters until we cover the whole string
                    normalized_sents.append(s[k:k+max_sentence_len].strip())
            else:
                normalized_sents.extend(subparts)  # append the large text chunks to the final list

    chunks = [] # list to contain the list of PDF text chunks
    current_sents = [] # list to contain text to be stored in a single chunk
    current_len = 0 # the length of each current_sents to be stored in a chunk

    # --------------------------------------------------------
    # reset the current sentence and the current chunk length
    # but first, append the current sentence to the final chunk list and append white space before the sentence
    # --------------------------------------------------------
    def flush_chunk():
        nonlocal current_sents, current_len # define global variables inside function
        if current_sents:
            chunks.append(" ".join(current_sents).strip())
        current_sents = []
        current_len = 0

    # --------------------------------------------------------
    # append the current sentence to the final chunk list and append white space before the sentence
    # then reset the current_sents and current_len
    # --------------------------------------------------------
    def apply_overlap():
        nonlocal current_sents, current_len # define global variables inside function

        if overlap_sentences <= 0 or not chunks: # if no overlap is specified in function call or we don't have values in chunk list yet, skip
            return
        # if we have chunk and overlap values are set
        prev = chunks[-1] # get the last chunk stored in chunk list
        prev_sents = split_arabic_sentences(prev) # use split_arabic_sentences to get a list of sentences in last chunk
        tail = prev_sents[-overlap_sentences:] if prev_sents else []  # get only the last two sentences present in the last chunk
        current_sents = tail[:]  # start the current sentence with the last two sentences of the last chunk
        current_len = sum(len(x) + 1 for x in current_sents) # calculate the number of characters in the current sentence as initial chunk size

    # --------------------------------------------------------

    for sent in normalized_sents: # loop through the list of text divided to string values less than or equal to 350 characters
        if not sent: # if there is no string, move to the next element of the list
            continue

        if sent in heading_lines or is_probable_arabic_heading(sent): # test if the sentence is part of the heading list or could be a heading after splitted by punctation.
            flush_chunk() # reset the chunking trackers (length and current sentence)
            # Start new chunk with the heading line (no overlap)
            current_sents = [sent] # starting a new chunk with new line
            current_len = len(sent) + 1 # initialize the current chunk size
            continue

        add_len = len(sent) + (1 if current_sents else 0) # if the splitted sentence wasn't a heading and current sentence tracker wasn't empty, calculate the expected new current sentence
        if current_len + add_len <= chunk_size: # if merging the current sentence with the splitted sentence is less than the maximum chunk size, append the splitted sentence to current sentence and increase the length
            current_sents.append(sent)
            current_len += add_len
        else: # else reset the current_sentence and start with the sentence as a new chunk, apply the overlap
            flush_chunk()
            apply_overlap()
            # If overlap itself is already too big, drop overlap (rare but possible)
            if current_len + len(sent) + 1 > chunk_size and current_sents:
                current_sents = [] # remove overlap from current_sents
                current_len = 0 # reset the current_sents character counter
            add_len = len(sent) + (1 if current_sents else 0)
            current_sents.append(sent) # append the sentence to the current_sents
            current_len += add_len # update the character length in current_sents

    flush_chunk() # once we are done with all the normalized text, flush the chunk and reset all trackers
    return [c for c in chunks if c.strip()]


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

# --------------------------------------------------------
# PDF Loader
# --------------------------------------------------------

def chunk_pdf_pages(
    input_pdf_path,
    output_jsonl_path=None,
    normalize=True,
    min_chars=15,
    chunk_size=300
):
    pdf = fitz.open(input_pdf_path)
    chunks = []
    doc_name = Path(input_pdf_path).stem # get the file name

    print(f"\n📘 Processing PDF: {doc_name}")
    print(f"Total pages: {len(pdf)}")

    for i in range(len(pdf)):
        page = pdf[i]

        # Extract via hybrid method
        raw_text = extract_pdf_page_text(page)

        if not raw_text or len(raw_text.strip()) < min_chars:
            print(f"⚠️ Page {i+1}: No extractable text → Skipped")
            continue

        if normalize:
            cleaned = normalize_arabic_text(raw_text)

        # check if the cleaned text contain the least number of arabic characters
        arabic_chars = re.findall(r"[ء-ي]", cleaned)
        if len(arabic_chars) < 10:
            print(f"⚠️ Page {i+1}: Insufficient Arabic content → Skipped")
            continue

        # send the text to chunking
        page_chunks = semantic_chunk_text_v2(
            cleaned,
            chunk_size=chunk_size,
            overlap_sentences=2
        )

        # store the final chunk into the chromadb
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
#  URL Loader
# ============================================================

#--------------------------------------------------
# ChromaDB accepts only unique IDs, but when chunking URL-PDF files page by page, we still access the same URL,
# hence, we create a new ID value by concatenating URL with additional text ->
# {_pdf_{page_number+1}_chunk_{chunk_number}; page_number starts with 0 in the enumerator, so we add 1 to reflect actual page_number}
# the new string ID will be converted into a hashed format, so we first convert it into bytes.
# Later, we convert the hash result into hexadecimal string (to make it easily readable by human) and pick only the first 12 charachters
# (sufficient to get unique string, the ) Example below,
# 'https://example.com/page?id=42' -> b'\x8f\xb2\x8c\x8a\x0e\x17\x8a\xac\x9d\xad\x84\.. -> 8fb28c8a0e178aac..
#--------------------------------------------------
def make_uid(url: str, extra: str = ""):
    base = url + extra
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    return f"{h}"

#--------------------------------------------------
# Define a string to check that every link visited is within the
# domain of Ministry of Health website. We don't want to go to other domains like: google.com
#--------------------------------------------------
BASE_DOMAIN = "moh.gov.sa"

#--------------------------------------------------
# Define a list of string to exclude links that won't be considered during the crawling process
# awarenessplateform/EducationalSeries and awarenessplateform/PublishingImages: Contain mostly videos and images
# _layouts: Settings and admin tasks, not meant for user content
# Style%20Library: CSS, JS, fonts, and design assets
# SiteAssets: Replaces much of Style%20Library in modern sites. It has easier permission management.
# Images: Image storage folder
#--------------------------------------------------
EXCLUDE_PREFIXES = [ "/awarenessplateform/EducationalSeries", "/awarenessplateform/PublishingImages", "/_layouts/", "/Style%20Library/", "/SiteAssets/", "/Images/", ]

#--------------------------------------------------
# Define a list of string to the URLs with content type in text/html and application/xhtml+xml (the types for most popular well-structured human readable urls).
# text/html: page_type = text, subtype = html. i.e. this response (url) is an HTML document
# application/xhtml+xml: page_type = application, subtype = xhtml+xml. i.e. this response (url) is an XHTML, which is HTML written as strict XML. It was an attempt to control HTML documents structure.
#--------------------------------------------------
VALID_CONTENT_TYPES = ["text/html", "application/xhtml+xml"]

#--------------------------------------------------
# To make the HTTP request look like it’s coming from a real web browser, not from a script or bot.
# We add this header because when we use requests in python, many websites block the request to protect the content.
# Websites associate the python requests with bots, scrapers, and abuse
# User-Agent: the type of client we pretend to be.
#--------------------------------------------------

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Linux; X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/120.0"
}

#--------------------------------------------------
# Canonicalize links to remove the parts useless in the content url source.
# urlparset: takes a URL string and returns a structured object. ex.
# "https://example.com:8080/path/page?x=1#section"
# ParseResult(scheme='https',netloc='example.com:8080',path='/path/page',params='',query='x=1',fragment='section')
# To identfy source links, we keep [scheme + netloc + url + path + query] and remove fragmentation.
# Fragmentation appears in many cases, like when the link tag has an id value (eg. <a id = "section"> Click here </a>)
#--------------------------------------------------
def canonicalize_url(url):
    parsed = urlparse(url)
    query = f"?{parsed.query}" if parsed.query else "" # extract the query part if exists
    clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}{query}"
    clean = clean.split("#")[0] # additional checkup to make sure that we exclude everything after the first #. In rare cases, the parser may include the first # in query or path if there are multiple # values in the link (https://example.com/page#section#extra).
    return clean

#--------------------------------------------------
# check the validity of the url
# check if the link is within the valid domain (Ministry of Health), if yes, return false.
# check if the link is for an english content url, if yes, return false.
# check if the path is matching the path we are crawling, if no, return false.
# check if the path starts with any of the strings for link types we are excluding, if yes, return false.
#--------------------------------------------------

def is_valid_internal(url: str) -> bool:
    parsed = urlparse(url)

    if parsed.netloc.lower() not in [BASE_DOMAIN, f"www.{BASE_DOMAIN}"]:
        return False

    path = parsed.path.lower()

    # ✅ Reject English pages
    if path.startswith("/en/"):
        return False

    # Must be inside awareness platform
    if "/awarenessplateform/childshealth/" not in path:# update to selected targeted path

        return False

    for ex in EXCLUDE_PREFIXES:
        if path.startswith(ex.lower()):
            return False

    return True
#--------------------------------------------------
# check if the url points to a pdf file
#--------------------------------------------------
def is_pdf(url):
    return url.lower().endswith(".pdf")

def extract_sharepoint_main_text_from_soup(soup: BeautifulSoup, min_len: int = 200) -> str:
    selectors = [
        "#ctl00_PlaceHolderMain_ctl02__ControlWrapper_RichHtmlField",  # MOH common
        "#PageContent",
        ".ms-rtestate-field",  # classic SharePoint rich text
        "article",
        "main",
    ]

    for sel in selectors:
        node = soup.select_one(sel)
        if not node:
            continue

        text = node.get_text(" ", strip=True)
        if text and len(text) >= min_len:
            return text

    return ""

#--------------------------------------------------
# BeautifulSoup: a python library used for web scraping, creating a parse tree that allows for easy navigation,
# searching, and modification of data from web pages
# remove from the parsed tree all the tags that are not needed to be scraped.
# first check if there are any SharePoint related containers, extract their text, and return the cleaned version as the function output.
# if there is nothing related to SharePoint, check main tag for content, and return the cleaned version as the function output.
# if there is nothing related to SharePoint and nothing in main tag, return  the cleaned version of whatever is present in the HTML as the function output.
#--------------------------------------------------
def extract_main_content(html: str, min_len: int = 200) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # 1) Remove obvious junk / chrome
    for bad in ["script", "style", "noscript", "svg", "iframe", "header", "footer", "nav"]:
        for t in soup.find_all(bad):
            t.decompose()

    # 2) Try SharePoint / main containers
    text = extract_sharepoint_main_text_from_soup(soup, min_len=min_len)
    if text:
        return normalize_arabic_text(text)

    # 3) Fallback: body text
    if soup.body:
        return normalize_arabic_text(soup.body.get_text(" ", strip=True))

    # 4) Last resort
    return normalize_arabic_text(soup.get_text(" ", strip=True))

#--------------------------------------------------
# Request PDF from the HTTP server, if request is not failed, read the PDF file
# For each PDF page, extract the text, check the character threshold,
# check the Arabic characters threshold, clean the text, chunk it, and store it in DB
#--------------------------------------------------
def extract_pdf(
    url,
    normalize=True,
    min_chars=20,
    min_arabic_chars=10,
    chunk_size=300,
    overlap_sentences=2,
):
    print(f"📄 PDF → {url}")

    # GET request to the URL
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status() # throw exception if status was error code
    except Exception as e:
        print(f"❌ PDF download failed: {e}")
        return []

    # If no exception was thrown,
    try:
        pdf = fitz.open(stream=resp.content, filetype="pdf") # open PDF using PyMuPDF library
    except Exception as e:
        print(f"❌ PDF parse failed: {e}")
        return []

    out = []
    name = Path(urlparse(url).path).stem # extract the name of the PDF file, "https://example.com/files/report.pdf" -> report.pdf

    # enumerate through the PDF pages
    for i, page in enumerate(pdf):

        # extract text from the PDF page
        raw_text = extract_pdf_page_text(page)

        # if extracted text is less than 20 characters, ignore
        if not raw_text or len(raw_text.strip()) < min_chars:
            continue

        # clean the extracted text
        if normalize:
          cleaned = normalize_arabic_text(raw_text)

        # ensure that the text contains at least 20 Arabic character, else ignore
        arabic_chars = re.findall(r"[ء-ي]", cleaned)
        if len(arabic_chars) < min_arabic_chars:
            continue

        # apply semantic chunking to the extracted text
        page_chunks = semantic_chunk_text_v2(
            cleaned,
            chunk_size=chunk_size,
            overlap_sentences=overlap_sentences,
        )

        # enumerate through the PDF chunks and insert each chunk into the DB
        for j, ch_text in enumerate(page_chunks, start=1):
            out.append({
                "id": make_uid(url, f"_pdf_{i+1}_chunk_{j}"),
                "content": ch_text.strip(),
                "metadata": {
                    "type": "PDF_page",
                    "source": name,
                    "page": i + 1,
                    "chunk": j,
                    "page_url": url
                }
            })

    return out

#--------------------------------------------------
# Create a set of the links that are extracted from the URL
# look for traditional embedded links, validate them and add them to the set.
# create an string object to apply string search functions and find urls that are not contained in usual tags (a, iframe)
# first, look for complete URLs that are encapsulated in quotes (single/double)
# second, look for incomplete URLs that contain the targeted path and are encapsulated in quotes (single/double)
# third, look for links present in tags with attributes = data-url and data-href, these are no encapsulated in quotes
# finally, look for links in JS navigation
#--------------------------------------------------

def extract_links(soup, base):
    found = set()

    # look for normal and usual tags holding links within HTML structure
    for tag, attr in (("a", "href"), ("iframe", "src")):
      for el in soup.find_all(tag):
        link = el.get(attr)
        if not link:
            continue
        absolute = canonicalize_url(urljoin(base, link))
        if is_valid_internal(absolute):
            found.add(absolute)


    html = str(soup) # create text object to use findall

    # 1) Absolute URLs in quotes
    for link in re.findall(r'["\'](https?://[^"\']+)["\']', html):
        absolute = canonicalize_url(link)
        if is_valid_internal(absolute):
            found.add(absolute)

    # 2) Relative internal URLs in quotes. e.g., onclick="go('/awarenessplateform/home.aspx')"
    patterns = [r'["\'](/awarenessplateform/[^"\']*\.aspx[^"\']*)["\']',
    r'["\'](/healthawareness/[^"\']*\.aspx[^"\']*)["\']']

    for pat in patterns:
      for link in re.findall(pat, html, flags=re.I):
        link = link.lower()  # normalize the path
        absolute = canonicalize_url(urljoin(base, link))
        if is_valid_internal(absolute):
          found.add(absolute)

    # 3) Retrieve the data-href and data-url values. e.g., <div data-href="/awarenessplateform/home.aspx"> .. </div>, <span data-url="https://example.com/page"></span>
    # splitted from the above loop solution because it usually doesn't contain quotes
    for link in re.findall(r'(?:data-href|data-url)\s*=\s*["\']([^"\']+)["\']', html, flags=re.I):
        absolute = canonicalize_url(urljoin(base, link))
        if is_valid_internal(absolute):
            found.add(absolute)

    # 4) JS navigation assignments. e.g., window.location = "/awarenessplateform/home.aspx"; location.href = "https://example.com/login";
    for link in re.findall(r'(?:window\.location|location\.href)\s*=\s*["\']([^"\']+)["\']', html, flags=re.I):
        absolute = canonicalize_url(urljoin(base, link))
        if is_valid_internal(absolute):
            found.add(absolute)


    return found

#--------------------------------------------------
# Define a set for the visited links to avoid accessing the same link more than once.
# Define a deque object: a double-ended queue data structure, which allows for efficient addition and removal of elements from both the front and the rear.
# Define the list of URLs that we will access to manage the run time easily.
# Define a final list to contain all the web/web-PDF chunks
# Start with the first appended link, canonicalize it, append it to visited, check its type (PDF, URL, none)
# If it was for PDF, send it to PDF_extract, if it was for URL, send it to extract_main_content, if it wa not valid, ignore it.
# Append the chunks to the final chunk list
# Explore embedded links and add them to the links queue
#--------------------------------------------------
def crawl_all_awareness(max_pages=1500, delay=0.15):
    visited = set()
    queue = deque()

    SEED_URLS = [
        #"https://www.moh.gov.sa/AwarenessPlateform/ChronicDisease/Pages/default.aspx?PageIndex=1",
        # "https://www.moh.gov.sa/awarenessplateform/Patientsrights/Pages/default.aspx"
        # "https://www.moh.gov.sa/awarenessplateform/OralHealth/Pages/default.aspx"
        # "https://www.moh.gov.sa/awarenessplateform/HealthyLifestyle/Pages/default.aspx",
        # "https://www.moh.gov.sa/awarenessplateform/Firstaid/Pages/default.aspx",
        # "https://www.moh.gov.sa/awarenessplateform/SeasonalAndFestivalHealth/Pages/default.aspx",
        # "https://www.moh.gov.sa/awarenessplateform/WomensHealth/Pages/default.aspx",
        # "https://www.moh.gov.sa/awarenessplateform/ElderlysHealth/Pages/default.aspx",
        # "https://www.moh.gov.sa/awarenessplateform/ChildsHealth/Pages/default.aspx",
        # "https://www.moh.gov.sa/awarenessplateform/VariousTopics/Pages/default.aspx"
        # "https://www.moh.gov.sa/HealthAwareness/Beforemarriage/Pages/default.aspx",
        # "https://www.moh.gov.sa/HealthAwareness/Pilgrims-Health/Pages/default.aspx",
        # "https://www.moh.gov.sa/HealthAwareness/EducationalContent/Pages/default.aspx"

        ]

    for u in SEED_URLS: # for each link
        queue.append(canonicalize_url(u)) # append to the queue defined to contain the URLs

    pages = [] # create a list that will contain all the chunks for the web/web-PDF pages

    print("\n🌍 Starting platform crawl...\n")

    # start scrolling through the links queue
    while queue and len(visited) < max_pages:

        url = canonicalize_url(queue.popleft()) # start with the first inserted link, remove fragmentation

        if url in visited: # test if previously visited, else append it to the visisted list
            continue
        visited.add(url)

        print(f"\n🔎 Visiting: {url}")


        if is_pdf(url): # test if it is a url for PDF, if yes send it to extract_pdf, then append the result to the pages list, next move to the following link in the queue
            pdf_chunks = extract_pdf(url)
            pages.extend(pdf_chunks)
            continue


        try: # if it wasn't PDF, send HTTP request to get the web page and through "Request failed" if response time exceeded 12 seconds
            resp = requests.get(url, headers=HEADERS, timeout=12)
        except:
            print("⚠ Request failed")
            continue

        ctype = resp.headers.get("Content-Type", "").lower() # retrieve the content type of the recieved web page

        if not any(t in ctype for t in VALID_CONTENT_TYPES): # test if the content type is of a valid type (HTML, or strict HTML), else skip it
            print("⚠ Not HTML, skipped")
            continue

        html = resp.text # extract the text of the web-page
        soup = BeautifulSoup(html, "html.parser") # parse the text using BeautifulSoup to build the parsing tree


        text = extract_main_content(html) # send the parsed content to the extract_main_content
        if text: # if text is not empty, add the chunk to the pages list
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


        new_links = extract_links(soup, url) # send the link to extract_links to find embedded links

        print(f"🔗 Found {len(new_links)} links")

        for link in new_links: # if found and are not present in visited list, append them to the queue to be able to visit them
            if link not in visited:
                queue.append(link)

        time.sleep(delay) # time pauses between each HTTP request to avoid overwhelming the server

    print(f"\n🎉 Crawl completed → {len(pages)} items extracted.\n")
    return pages



# ============================================================
#  Initialize models & vector DB
# ============================================================
if __name__ == "__main__":
    # Load embedding model
    embed_model = load_embedding_model()

    # Load reranker (tokenizer + model)
    reranker_tokenizer, reranker_model = load_reranker_model()

    # Initialize Chroma collection
    collection = init_chroma_collection()

    print("\n✅ All models and vector database initialized successfully!")

# ============================================================
#  Load URLs
# ============================================================
url_chunks = crawl_all_awareness()
store_chunks_in_chroma(url_chunks, collection, embed_model, batch_size=128)
save_jsonl(url_chunks,r"C:\Users\haila\OneDrive\Desktop\Project\AI_Tools_Final\Data\ChildHealth.jsonl")

# ============================================================
#  Load PDF
# ============================================================
output_jsonl__pdf_path = r"C:\Users\haila\OneDrive\Desktop\Project\AI_Tools_Final\Data\pdf_output.jsonl"
pdf_chunks = chunk_pdf_pages(r"C:\Users\haila\OneDrive\Desktop\Project\AI_Tools_Final\Data\Dose-fo-Awareness-2.pdf", output_jsonl__pdf_path,normalize=True)
store_chunks_in_chroma(pdf_chunks, collection, embed_model, batch_size=128)

# ChildsHealth = read_jsonl_file(r"C:\Users\haila\OneDrive\Desktop\Project\AI_Tools_Final\Data\ChildHealth.jsonl")
# ElderlysHealth = read_jsonl_file(r"C:\Users\haila\OneDrive\Desktop\Project\AI_Tools_Final\Data\ElderlyHealth.jsonl")
# Firstaid = read_jsonl_file(r"C:\Users\haila\OneDrive\Desktop\Project\AI_Tools_Final\Data\FirstAid.jsonl")
# HealthyLifeStyle = read_jsonl_file(r"C:\Users\haila\OneDrive\Desktop\Project\AI_Tools_Final\Data\HealthyLifestyle.jsonl")
# OralHealth = read_jsonl_file(r"C:\Users\haila\OneDrive\Desktop\Project\AI_Tools_Final\Data\OralHealth.jsonl")
# PatientsRight = read_jsonl_file(r"C:\Users\haila\OneDrive\Desktop\Project\AI_Tools_Final\Data\PatientRight.jsonl")
# Pilgrims_Health = read_jsonl_file(r"C:\Users\haila\OneDrive\Desktop\Project\AI_Tools_Final\Data\Pilgrims_Health.jsonl")
# SeasonalAndFestivalHealth = read_jsonl_file(r"C:\Users\haila\OneDrive\Desktop\Project\AI_Tools_Final\Data\SeasonalFestivalHealth.jsonl")
# VariousTopics = read_jsonl_file(r"C:\Users\haila\OneDrive\Desktop\Project\AI_Tools_Final\Data\VariousTopic.jsonl")
# WomensHealth = read_jsonl_file(r"C:\Users\haila\OneDrive\Desktop\Project\AI_Tools_Final\Data\WomenHealth.jsonl")
# BeforeMarriage = read_jsonl_file(r"C:\Users\haila\OneDrive\Desktop\Project\AI_Tools_Final\Data\BeforeMarriage.jsonl")
# ChronicDisease = read_jsonl_file(r"C:\Users\haila\OneDrive\Desktop\Project\AI_Tools_Final\Data\ChronicDisease.jsonl")
# EducationalContent = read_jsonl_file(r"C:\Users\haila\OneDrive\Desktop\Project\AI_Tools_Final\Data\EducationalContent.jsonl")


# store_chunks_in_chroma(ChildsHealth, collection, embed_model, batch_size=128)
# store_chunks_in_chroma(ElderlysHealth, collection, embed_model, batch_size=128)
# store_chunks_in_chroma(Firstaid, collection, embed_model, batch_size=128)
# store_chunks_in_chroma(HealthyLifeStyle, collection, embed_model, batch_size=128)
# store_chunks_in_chroma(OralHealth, collection, embed_model, batch_size=128)
# store_chunks_in_chroma(PatientsRight, collection, embed_model, batch_size=128)
# store_chunks_in_chroma(Pilgrims_Health, collection, embed_model, batch_size=128)
# store_chunks_in_chroma(SeasonalAndFestivalHealth, collection, embed_model, batch_size=128)
# store_chunks_in_chroma(VariousTopics, collection, embed_model, batch_size=128)
# store_chunks_in_chroma(WomensHealth, collection, embed_model, batch_size=128)
# store_chunks_in_chroma(BeforeMarriage, collection, embed_model, batch_size=128)
# store_chunks_in_chroma(ChronicDisease, collection, embed_model, batch_size=128)
# store_chunks_in_chroma(EducationalContent, collection, embed_model, batch_size=128)

# ----------------------------
# Helpers: Arabic check (optional gate for contexts)
# ----------------------------

ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
LETTER_OR_DIGIT_RE = re.compile(r"[A-Za-z0-9\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")

def is_arabic_chunk(text: str, min_arabic_chars: int = 30, min_arabic_ratio: float = 0.20) -> bool:
    if not text or not text.strip():
        return False
    arabic_chars = ARABIC_CHAR_RE.findall(text)
    if len(arabic_chars) < min_arabic_chars:
        return False
    letters_digits = LETTER_OR_DIGIT_RE.findall(text)
    if not letters_digits:
        return False
    return (len(arabic_chars) / len(letters_digits)) >= min_arabic_ratio

def rag_query(
    query, collection, embed_model, reranker_tokenizer, reranker_model,
    top_k=3,
    overfetch=3,
    min_rerank_score=None,          # set e.g. 0.0 or 1.0 if you want a gate
    arabic_only=True,
    min_arabic_chars=30,
    min_arabic_ratio=0.20,
):
    # 1) query embedding
    query_emb = embed_model.encode(
        [query],
        task="retrieval.query",
        convert_to_numpy=True
    )[0].tolist()

    # 2) dense retrieve
    search = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k * overfetch,
        include=["documents", "metadatas", "distances"]
    )

    docs  = (search.get("documents") or [[]])[0]
    metas = (search.get("metadatas") or [[]])[0]

    # 2.5) remove empties + optional Arabic filter
    filtered = []
    for d, m in zip(docs, metas):
        if not d or not d.strip():
            continue
        filtered.append((d, m or {}))

    if not filtered:
        return []

    docs = [d for d, _ in filtered]
    metas = [m for _, m in filtered]

    # 2.6) dedupe by doc text (keeps first)
    unique = {}
    for d, m in zip(docs, metas):
        if d not in unique:
            unique[d] = m
    docs = list(unique.keys())
    metas = list(unique.values())

    # 3) rerank
    pairs = [[query, d] for d in docs]
    tokens = reranker_tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=1024
    ).to(reranker_model.device)

    with torch.no_grad():
        scores = reranker_model(**tokens).logits.squeeze(-1).float().cpu().tolist()

    ranked = sorted(zip(scores, docs, metas), key=lambda x: x[0], reverse=True)

    # 4) optional confidence gate (still returns consistent type)
    if min_rerank_score is not None and ranked and ranked[0][0] < min_rerank_score:
        return []

    return ranked[:top_k]


# ============================================================
#  Query Test
# ============================================================
query = "ما هو علاج التبول اللاإرادي عند الأطفال؟"
results = rag_query(query, collection, embed_model, reranker_tokenizer,reranker_model, top_k=3)

# Top matching results and scores
for score, doc, meta in results[:5]:
    print("\nSCORE:", float(score))
    print("SOURCE:", meta.get("source"))
    print("TEXT:", doc[:400])
    print("-" * 60)

# ============================================================
#  LLM Setup
# ============================================================

if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]

os.environ["OPENAI_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxx"

print("API key updated.")
client = OpenAI()

# ============================================================
#  Generate response from LLM
# ============================================================

def extract_citation_ids(text: str) -> set[int]:
    # matches [1], [12], etc. (and avoids [[]] edge cases)
    ids = set()
    for m in re.finditer(r"\[(\d{1,3})\]", text):
        ids.add(int(m.group(1)))
    return ids

def generate_rag_answer_with_citations(query, results):
    """
    Hallucination-resistant answer generator using only retrieved chunks.
    """
    if not results:
        return "المعلومات المتوفرة لا تجيب عن السؤال مباشرة."
    # ============================
    # LIMIT CHUNKS TO TOP-3
    # ============================
    max_chunks = 3
    results = results[:max_chunks]

    # ============================
    # FORMAT CONTEXT + CITE SOURCES
    # ============================
    docs_context = []
    source_map = {}
    next_id = 1

    for i, (score, doc, meta) in enumerate(results, start=1):

        # —— Determine citation label
        if meta.get("type") == "PDF_page":
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
5. لا تضف أي مصادر إضافية غير التي تظهر في المقاطع.
6. أجب بأنه لا يتوفر لديك معلومات إذا لم تجد إجابة للسؤال.
7. لا تقدم أي معلومات إضافية خارج السياق.
8.لا تذكر أي مرجع لم تستخدمه في كتابة الرد.
9. أجب فقط من مصادر تحتوي على نص عربي
السؤال:
{query}

النصوص المتاحة:
{context_text}

الجواب (مع الاستشهادات):
"""


    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    answer = completion.choices[0].message.content.strip()
    used_cids = extract_citation_ids(answer)

# If the model returned no citations at all, you can choose:
# - either show none, or
# - show all, or
# - enforce at least one citation if it answered.
# I'd recommend showing none to match your rule #8.
    source_list = "### 📚 المصادر المستخدمة:\n"

# Build an inverse map cid -> label (because your current map is label -> cid)
    cid_to_label = {cid: label for label, cid in source_map.items()}

    for cid in sorted(used_cids):
      label = cid_to_label.get(cid)
      if not label:
        continue

      if label.startswith("URL:"):
        url = label.replace("URL:", "")
        source_list += f"- [{cid}] رابط: {url}\n"
      elif label.startswith("PDF:"):
        name = label.replace("PDF:", "")
        source_list += f"- ملف PDF: {name} [{cid}]\n"
      else:
        source_list += f"- {label} [{cid}]\n"

# If nothing was used, don't show an empty sources section
    if used_cids:
      return answer + "\n\n" + source_list
    else:
      return answer


# ============================================================
#  Test RAG
# ============================================================
#final_answer = generate_rag_answer_with_citations(query, results)
#print(final_answer)

from urllib.parse import urlparse

def canonicalize_url_simple(url: str) -> str:
    p = urlparse(url.strip())
    query = f"?{p.query}" if p.query else ""
    return f"{p.scheme}://{p.netloc}{p.path}{query}".split("#")[0]

def meta_to_reference(meta: dict) -> str:
    meta = meta or {}
    t = meta.get("type")

    if t == "web_page":
        u = meta.get("page_url") or meta.get("url") or ""
        return "URL:" + canonicalize_url_simple(u) if u else "URL:UNKNOWN"

    if t == "PDF_page":
        name = meta.get("source", "PDF")
        page = meta.get("page", None)
        return f"PDF:{name}:p{int(page)}" if page is not None else f"PDF:{name}"

    return f"SRC:{meta.get('source','UNKNOWN')}"


def export_top_sources_for_annotation(
    questions_excel: str,
    out_excel: str,
    pool_chunks: int = 30,     # pull more chunks to ensure you get 10 unique sources
    target_unique_sources: int = 10
):
    df = pd.read_excel(questions_excel)
    rows = []

    for _, r in df.iterrows():
        qid = r["qid"]
        q = str(r["question"])

        # pull a wider chunk pool then dedupe by source
        results = rag_query(
            q, collection, embed_model, reranker_tokenizer, reranker_model,
            top_k=pool_chunks,
            overfetch=5,
            arabic_only=True
        )

        seen = set()
        unique_count = 0

        for rank, (score, chunk, meta) in enumerate(results, start=1):
            ref = meta_to_reference(meta)
            if ref in seen:
                continue
            seen.add(ref)
            unique_count += 1

            rows.append({
                "qid": qid,
                "question": q,
                "pool_rank": rank,         # rank among chunks
                "reference": ref,          # <-- gold unit you mark
                "rerank_score": float(score),
                "page_url": (meta or {}).get("page_url"),
                "source": (meta or {}).get("source"),
                "page": (meta or {}).get("page"),
                "example_chunk": chunk[:500],
                "is_gold": "",             # you fill 1/0
                "notes": ""
            })

            if unique_count >= target_unique_sources:
                break

    pd.DataFrame(rows).to_excel(out_excel, index=False)
    print("Saved:", out_excel)

export_top_sources_for_annotation(
    questions_excel=r"C:\Users\haila\OneDrive\Desktop\Project\AI_Tools_Final\Data\questions.xlsx",
    out_excel=r"C:\Users\haila\OneDrive\Desktop\Project\AI_Tools_Final\Data\top10_sources_for_gold.xlsx",
    pool_chunks=30,
    target_unique_sources=10
)


def load_gold_sources(marked_excel: str) -> dict:
    df = pd.read_excel(marked_excel)
    df = df[df["is_gold"] == 1]
    gold = {}
    for qid, g in df.groupby("qid"):
        gold[qid] = set(g["reference"].astype(str).tolist())
    return gold

def evaluate_hit_mrr_at_3(
    questions_excel: str,
    marked_gold_excel: str,
    out_excel: str
):
    df = pd.read_excel(questions_excel)
    gold = load_gold_sources(marked_gold_excel)

    rows = []
    for _, r in df.iterrows():
        qid = r["qid"]
        q = str(r["question"])
        gold_refs = gold.get(qid, set())

        res = rag_query(
            q, collection, embed_model, reranker_tokenizer, reranker_model,
            top_k=3,
            overfetch=5,
            arabic_only=True
        )

        retrieved_refs = []
        seen = set()
        for _, _, meta in res:
            ref = meta_to_reference(meta)
            if ref not in seen:
                retrieved_refs.append(ref)
                seen.add(ref)

        hit3 = 0
        mrr3 = 0.0
        for i, ref in enumerate(retrieved_refs[:3], start=1):
            if ref in gold_refs:
                hit3 = 1
                mrr3 = 1.0 / i
                break

        rows.append({
            "qid": qid,
            "num_gold_refs": len(gold_refs),
            "hit@3": hit3,
            "mrr@3": mrr3,
            "retrieved_top3": ";".join(retrieved_refs)
        })

    out_df = pd.DataFrame(rows)
    summary = {
        "Hit@3": out_df["hit@3"].mean(),
        "MRR@3": out_df["mrr@3"].mean(),
        "Avg # gold refs": out_df["num_gold_refs"].mean()
    }

    with pd.ExcelWriter(out_excel, engine="openpyxl") as w:
        out_df.to_excel(w, index=False, sheet_name="per_question")
        pd.DataFrame([summary]).to_excel(w, index=False, sheet_name="summary")

    print("Saved:", out_excel)


evaluate_hit_mrr_at_3(
    questions_excel=r"C:\Users\haila\OneDrive\Desktop\Project\AI_Tools_Final\Data\questions.xlsx",
    marked_gold_excel=r"C:\Users\haila\OneDrive\Desktop\Project\AI_Tools_Final\Data\top10_sources_for_gold.xlsx",
    out_excel=r"C:\Users\haila\OneDrive\Desktop\Project\AI_Tools_Final\Data\eval_hit3.xlsx"
)


def citation_validity(answer: str, n_ctx: int) -> dict:
    used = extract_citation_ids(answer)
    if not used:
        return {"valid": False, "used": [], "invalid": [], "notes": "No citations used."}
    invalid = sorted([c for c in used if c < 1 or c > n_ctx])
    return {
        "valid": len(invalid) == 0,
        "used": sorted(list(used)),
        "invalid": invalid,
        "notes": "" if len(invalid) == 0 else f"Out of range citations: {invalid}"
    }

def extract_citation_ids(text: str) -> set[int]:
  # matches [1], [12], etc. (and avoids [[]] edge cases)
  ids = set()
  for m in re.finditer(r"\[(\d{1,3})\]", text):
    ids.add(int(m.group(1)))
  return ids


import pandas as pd
import json
import time
from typing import List, Dict, Any, Tuple

# ----------------------------
# Config
# ----------------------------
EVAL_XLSX_PATH = r"C:\Users\haila\OneDrive\Desktop\Project\AI_Tools_Final\Data\EvaluationForGeneration.xlsx"
OUT_CSV_PATH = r"C:\Users\haila\OneDrive\Desktop\Project\AI_Tools_Final\Data\rag_generation_judge_report.csv"

JUDGE_MODEL = "gpt-4o-mini"
MAX_RETRIES = 3
SLEEP_BETWEEN = 1.5

TOP_K_RETRIEVAL = 3          # retrieval window
GEN_MAX_CHUNKS = 3           # contexts fed to judge (must match generator)

ABSTAIN_PHRASES = [
    "المعلومات المتوفرة لا تجيب عن السؤال مباشرة",
    "لا تتوفر معلومات كافية للإجابة",
]

# ----------------------------
# Helpers: parse gold sources from your sheet
# column "reference" is a JSON list string
# ----------------------------
def parse_reference_cell(x) -> List[str]:
    if x is None:
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(i).strip() for i in v if str(i).strip()]
    except Exception:
        pass
    # fallback: allow ; separated
    return [p.strip() for p in s.split(";") if p.strip()]

# ----------------------------
# Strip appended sources list from your generator output
# ----------------------------
def split_answer_and_sources(full_text: str) -> Tuple[str, str]:
    marker = "### 📚 المصادر المستخدمة:"
    if marker in (full_text or ""):
        ans, src = full_text.split(marker, 1)
        return ans.strip(), (marker + src).strip()
    return (full_text or "").strip(), ""

def did_abstain(answer: str) -> bool:
    a = (answer or "").strip()
    return any(p in a for p in ABSTAIN_PHRASES)

# ----------------------------
# Deterministic citation validity (consistent with top-3 contexts)
# Requires your extract_citation_ids(answer) function
# ----------------------------
def citation_validity(answer: str, n_ctx: int) -> Dict[str, Any]:
    used = extract_citation_ids(answer or "")
    if not used:
        return {"valid": False, "used": [], "invalid": [], "notes": "No citations found."}
    invalid = sorted([c for c in used if c < 1 or c > n_ctx])
    return {
        "valid": len(invalid) == 0,
        "used": sorted(list(used)),
        "invalid": invalid,
        "notes": "" if not invalid else f"Out-of-range citations: {invalid}"
    }

# ----------------------------
# Judge (LLM-as-judge)
# IMPORTANT: contexts are labeled [1],[2],[3] to match your citation style
# ----------------------------
def judge_rag_answer(question: str, answer: str, contexts: List[str], model: str = JUDGE_MODEL) -> Dict[str, Any]:
    trimmed = [c[:2500] for c in contexts]
    context_text = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(trimmed)])

    prompt = f"""
أنت مقيّم جودة لإجابات نظام RAG طبي عربي.
يجب أن تقيم الإجابة اعتماداً على النصوص المعطاة فقط، ولا تستخدم أي معرفة خارجية.

قيّم الإجابة عبر 4 محاور:

1) Faithfulness/Groundedness:
- هل كل الادعاءات في الإجابة مدعومة صراحة من النصوص [1..]؟
- إذا وُجدت أي معلومة غير مدعومة، اذكرها في unsupported_claims.
- passed=true فقط إذا كانت جميع الادعاءات مدعومة من النصوص.

2) Answer Relevance:
- هل الإجابة تجيب عن السؤال مباشرة؟

3) Contextual Correctness:
- هل الإجابة صحيحة "بالنسبة للنصوص"؟ (لا تفترض معلومات خارج النصوص)
- critical_error=true إذا تضمنت الإجابة نصيحة قد تكون خطرة أو مضللة مقارنة بما ورد في النصوص.

4) Fluency/Readability:
- سلامة اللغة العربية ووضوحها.

مقياس الدرجات لكل محور: 1 (سيئ جداً) إلى 5 (ممتاز).

أعد JSON فقط بهذا الشكل (بدون أي نص إضافي):

{{
  "faithfulness": {{"score": 1, "passed": false, "unsupported_claims": [], "notes": ""}},
  "relevance": {{"score": 1, "notes": ""}},
  "correctness": {{"score": 1, "critical_error": false, "notes": ""}},
  "fluency": {{"score": 1, "notes": ""}}
}}

السؤال:
{question}

الإجابة:
{answer}

النصوص:
{context_text}
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "أنت مقيّم صارم ومحايد. أعد JSON فقط بدون أي شرح خارج JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0
    )

    raw = resp.choices[0].message.content.strip().strip("`").strip()
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()

    try:
        j = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            j = json.loads(raw[start:end+1])
        else:
            raise

    for k in ["faithfulness", "relevance", "correctness", "fluency"]:
        if k not in j:
            raise ValueError(f"Judge JSON missing key: {k}. Raw head: {raw[:200]}")
    return j

# ----------------------------
# Run your RAG for one question (top-3)
# Also compute retrieved source references for gold matching
# ----------------------------
def run_rag_fn(question: str) -> Dict[str, Any]:
    results = rag_query(
        query=question,
        collection=collection,
        embed_model=embed_model,
        reranker_tokenizer=reranker_tokenizer,
        reranker_model=reranker_model,
        top_k=TOP_K_RETRIEVAL,
        overfetch=5,
        arabic_only=True
    )

    if not results:
        return {"answer": "لا تتوفر معلومات كافية للإجابة.", "contexts": [], "sources": [], "raw_results": []}

    # contexts (exactly what judge sees)
    top_for_generation = results[:GEN_MAX_CHUNKS]
    contexts = [doc for (_, doc, _) in top_for_generation]

    # compute source refs from metas
    sources = []
    seen = set()
    for (_, _, meta) in top_for_generation:
        ref = meta_to_reference(meta)
        if ref not in seen:
            sources.append(ref)
            seen.add(ref)

    full_answer = generate_rag_answer_with_citations(question, results)
    answer_text, sources_text = split_answer_and_sources(full_answer)

    return {
        "answer": answer_text,
        "sources_text": sources_text,
        "contexts": contexts,
        "sources": sources,
        "raw_results": top_for_generation
    }

# ----------------------------
# Main evaluation loop (Excel)
# ----------------------------
def evaluate_generation_from_excel() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df_eval = pd.read_excel(EVAL_XLSX_PATH)

    required = {"qid", "question", "answerable", "reference"}
    missing = required - set(df_eval.columns)
    if missing:
        raise ValueError(f"Missing columns in evaluation sheet: {missing}")

    records = []
    for idx, row in df_eval.iterrows():
        qid = row["qid"]
        q = str(row["question"]).strip()
        answerable = int(row["answerable"])
        gold_sources = set(parse_reference_cell(row["reference"]))

        rag_out = run_rag_fn(q)
        answer = rag_out["answer"]
        contexts = rag_out["contexts"]
        retrieved_sources = set(rag_out["sources"])

        abstained = did_abstain(answer)
        abstention_correct = (abstained and answerable == 0) or ((not abstained) and answerable == 1)

        used_gold_source = None
        if gold_sources:
            used_gold_source = len(gold_sources.intersection(retrieved_sources)) > 0

        cite_check = citation_validity(answer, n_ctx=len(contexts))

        # Judge with retries
        judgement, last_err = None, ""
        for attempt in range(MAX_RETRIES):
            try:
                judgement = judge_rag_answer(q, answer, contexts, model=JUDGE_MODEL)
                break
            except Exception as e:
                last_err = str(e)
                time.sleep(SLEEP_BETWEEN * (attempt + 1))

        if judgement is None:
            records.append({
                "qid": qid,
                "question": q,
                "answerable": answerable,
                "abstained": abstained,
                "abstention_correct": abstention_correct,
                "used_gold_source": used_gold_source,
                "retrieved_sources": ";".join(sorted(retrieved_sources)),
                "gold_sources": ";".join(sorted(gold_sources)),
                "answer": answer,
                "citation_valid": cite_check["valid"],
                "citation_used": ",".join(map(str, cite_check["used"])),
                "citation_invalid": ",".join(map(str, cite_check["invalid"])),
                "faithfulness_score": None,
                "faithfulness_passed": None,
                "unsupported_claims": None,
                "relevance_score": None,
                "correctness_score": None,
                "critical_error": None,
                "fluency_score": None,
                "judge_notes": None,
                "judge_error": last_err
            })
            continue

        records.append({
            "qid": qid,
            "question": q,
            "answerable": answerable,
            "abstained": abstained,
            "abstention_correct": abstention_correct,
            "used_gold_source": used_gold_source,
            "retrieved_sources": ";".join(sorted(retrieved_sources)),
            "gold_sources": ";".join(sorted(gold_sources)),
            "answer": answer,
            "citation_valid": cite_check["valid"],
            "citation_used": ",".join(map(str, cite_check["used"])),
            "citation_invalid": ",".join(map(str, cite_check["invalid"])),
            "faithfulness_score": judgement["faithfulness"]["score"],
            "faithfulness_passed": judgement["faithfulness"]["passed"],
            "unsupported_claims": " | ".join(judgement["faithfulness"]["unsupported_claims"]),
            "relevance_score": judgement["relevance"]["score"],
            "correctness_score": judgement["correctness"]["score"],
            "critical_error": judgement["correctness"]["critical_error"],
            "fluency_score": judgement["fluency"]["score"],
            "judge_notes": (
                f"F:{judgement['faithfulness']['notes']} | "
                f"R:{judgement['relevance']['notes']} | "
                f"C:{judgement['correctness']['notes']} | "
                f"L:{judgement['fluency']['notes']}"
            ),
            "judge_error": ""
        })

        if (idx + 1) % 10 == 0:
            print(f"✅ Evaluated {idx+1}/{len(df_eval)}")

    df_out = pd.DataFrame(records)

    ok = df_out[df_out["faithfulness_score"].notna()]
    summary = {
        "n_total": int(len(df_out)),
        "n_scored": int(len(ok)),
        "citation_valid_rate": float(df_out["citation_valid"].mean()),
        "abstention_accuracy": float(df_out["abstention_correct"].mean()),
        "gold_source_covered_rate": float(ok["used_gold_source"].mean()) if "used_gold_source" in ok.columns else None,
        "faithfulness_pass_rate": float(ok["faithfulness_passed"].mean()) if len(ok) else 0.0,
        "avg_faithfulness": float(ok["faithfulness_score"].mean()) if len(ok) else 0.0,
        "avg_relevance": float(ok["relevance_score"].mean()) if len(ok) else 0.0,
        "avg_correctness": float(ok["correctness_score"].mean()) if len(ok) else 0.0,
        "critical_error_rate": float(ok["critical_error"].mean()) if len(ok) else 0.0,
        "avg_fluency": float(ok["fluency_score"].mean()) if len(ok) else 0.0,
    }

    df_out.to_csv(OUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print("✅ Saved report to:", OUT_CSV_PATH)
    print("===== SUMMARY =====")
    print(summary)

    return df_out, summary

# Run
df_gen, summary_gen = evaluate_generation_from_excel()

TELEGRAM_TOKEN = "xxxxxxxxxxxxxxxxxxx"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "👋 أهلاً بك في مساعد الصحة الذكي.\n\n"
        "اكتب سؤالاً صحياً مثل:\n"
        "• ما أعراض ارتفاع ضغط الدم؟\n"
        "• ما هي مضاعفات السكري؟\n"
        "وسأبحث في قاعدة المعرفة ثم أجيبك مع ذكر المصادر."
    )
    await update.message.reply_text(msg)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    query = update.message.text.strip()

    await update.message.reply_text("جاري معالجة السؤال ")

    try:
        # Use RAG query with Alibaba reranker
        current_results = rag_query(
            query=query,
            collection=collection,
            embed_model=embed_model,
            reranker_tokenizer=reranker_tokenizer,
            reranker_model=reranker_model,
            top_k=10
        )

        if not current_results:
          await update.message.reply_text("المعلومات المتوفرة لا تجيب عن السؤال مباشرة.")
          return

        response = generate_rag_answer_with_citations(query, current_results)

    except Exception as e:
        print(" Error:", e)
        await update.message.reply_text("حدث خطأ أثناء الإجابة.")
        return

    await update.message.reply_text(response)
def start_bot():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 RAG Telegram bot is running")
    app.run_polling(drop_pending_updates=True)




start_bot()


