"""Corpus loading and article index construction."""

import json
import re
from collections import defaultdict

from .config import CORPUS_PATH


def load_corpus(path=CORPUS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        snapshot = json.load(f)
    chunks = snapshot["chunks"]
    by_id = {item["source_id"]: item for item in chunks}
    return chunks, by_id


def docs_from_source_ids(source_ids, corpus_by_id):
    docs = []
    for sid in source_ids:
        item = corpus_by_id.get(sid)
        if item:
            docs.append({
                "source_id": sid,
                "doc_id": item["doc_id"],
                "chunk_id": item["chunk_id"],
                "text": item["text"],
            })
    return docs


def format_doc_evidence(docs):
    return "\n\n".join(
        f"[D{i}] {d['source_id']}\n{d['text']}"
        for i, d in enumerate(docs, 1)
    )


# --- Article structure index ---

ARTICLE_HEADING_RE = re.compile(
    r"\bArticle\s+(\d+[a-zA-Z]?)\s+"
    r"([A-Z][A-Za-z0-9\u00C0-\u00FF\u2019''(),/\-\s]{2,180}?)"
    r"(?:\s+\([^)]{1,220}\))?"
    r"\s+1\.",
    re.IGNORECASE,
)

SOURCE_FILES = {
    "mifid2": "Directive 2014/65/EU (MiFID II)",
    "mifir": "Regulation (EU) No 600/2014 (MiFIR)",
    "delegated565": "Commission Delegated Regulation (EU) 2017/565",
}


def _clean_legal_text(text):
    text = re.sub(r"[\u25bc\u25ba\u25c4]\s*[A-Z]?\s*\d*", " ", text)
    text = re.sub(
        r"020\d{2}[A-Z]\d+\s+—\s+EN\s+—\s+\d{2}\.\d{2}\.\d{4}\s+—\s+\S+\s+—\s+\d+",
        " ", text,
    )
    return re.sub(r"\s+", " ", text).strip()


def build_article_index(corpus):
    article_index = defaultdict(list)
    article_metadata = {}

    docs_by_docid = defaultdict(list)
    for item in corpus:
        docs_by_docid[item["doc_id"]].append(item)

    for doc_id in SOURCE_FILES:
        doc_chunks = sorted(
            docs_by_docid.get(doc_id, []),
            key=lambda x: x["chunk_id"],
        )

        for i, item in enumerate(doc_chunks):
            current_text = _clean_legal_text(item["text"])

            for match in ARTICLE_HEADING_RE.finditer(current_text):
                article_number = match.group(1).lower()
                article_title = match.group(2).strip()
                key = (doc_id, article_number)
                sid = item["source_id"]
                if sid not in article_index[key]:
                    article_index[key].append(sid)
                article_metadata[sid] = {
                    "doc_id": doc_id,
                    "article": article_number,
                    "article_title": article_title,
                }

            if i == 0:
                continue
            prev_item = doc_chunks[i - 1]
            prev_text = _clean_legal_text(prev_item["text"])
            prev_tail = prev_text[-1200:]
            current_head = current_text[:1200]
            joined_text = prev_tail + " " + current_head
            boundary = len(prev_tail) + 1

            for match in ARTICLE_HEADING_RE.finditer(joined_text):
                if not (match.start() < boundary and match.end() > boundary):
                    continue
                article_number = match.group(1).lower()
                article_title = match.group(2).strip()
                key = (doc_id, article_number)
                sid = item["source_id"]
                if sid not in article_index[key]:
                    article_index[key].insert(0, sid)
                article_metadata[sid] = {
                    "doc_id": doc_id,
                    "article": article_number,
                    "article_title": article_title,
                }

    return dict(article_index), article_metadata
