"""Retrieve recently published peer-reviewed journal articles via Crossref.

Mirrors the Crossref-based chemRxiv retriever, but queries a configurable list of
journals by ISSN. Crossref often lacks abstracts for APS and Springer Nature
records, so abstracts are enriched with a chain of fallbacks:

1. Crossref ``abstract`` (JATS, cleaned)
2. OpenAlex ``abstract_inverted_index`` (batched by DOI)
3. nature.com landing page ``dc.description`` meta tag (10.1038 DOIs only)
4. arXiv title search (10.1103 / APS DOIs only, capped, fuzzy title match)
5. title only (the paper is kept so it can still be ranked/emailed)

Time window: ``lookback_days`` whole UTC days by Crossref ``created`` date, ending at
``floor_to_utc_day(now - cutoff_shift_hours)``. With the default shift of 12 h, any
run between D 12:00 UTC and D+1 11:59 UTC covers ``[D-1, D)``; e.g. the 22:00 UTC
cron on day D covers all of day D-1 even if GitHub delays the job past midnight.
Consecutive daily runs therefore tile the calendar without gaps or duplicates.
"""

import html
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from time import sleep
from typing import Any

import feedparser
import requests
from loguru import logger

from .base import BaseRetriever, register_retriever
from ..protocol import Paper

DEFAULT_JOURNALS = [
    {"name": "Physical Review Letters", "issn": "0031-9007"},
    {"name": "Physical Review B", "issn": "2469-9950"},
    {"name": "Physical Review X", "issn": "2160-3308"},
    {"name": "PRX Quantum", "issn": "2691-3399"},
    {"name": "Nature", "issn": "0028-0836"},
    {"name": "Nature Physics", "issn": "1745-2473"},
    {"name": "Nature Materials", "issn": "1476-1122"},
    {"name": "Nature Communications", "issn": "2041-1723"},
    {"name": "Science", "issn": "0036-8075"},
    {"name": "Science Advances", "issn": "2375-2548"},
    {"name": "Journal of Chemical Theory and Computation", "issn": "1549-9618"},
    {"name": "The Journal of Chemical Physics", "issn": "0021-9606"},
]

NON_RESEARCH_TITLE = re.compile(
    r"^\s*(?:"
    r"(?:publisher|author)(?:['’]s)?\s+correction|correction(?:\s+to)?\b|corrigendum|erratum|errata|addendum|"
    r"retraction|retracted|expression\s+of\s+concern|withdrawal|editorial\b|editor['’]s\s+note|publisher['’]s\s+note|"
    r"daily\s+briefing|news\s*(?:&|and)\s*views|research\s+highlights?|in\s+brief|in\s+this\s+issue|this\s+week|"
    r"editors['’]?\s+choice|books?\s+(?:review|in\s+brief)|book\s+review|obituary|reply\s+to\b|"
    r"announcement|front\s+matter|back\s+matter|cover\s+image|table\s+of\s+contents|masthead|issue\s+information"
    r")",
    re.IGNORECASE,
)
NON_RESEARCH_OPENALEX_TYPES = {"editorial", "erratum", "retraction", "paratext", "letter"}


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


@dataclass
class JournalItem:
    journal: str
    item: dict[str, Any]
    abstract: str = ""
    abstract_source: str = ""


@register_retriever("journal")
class JournalRetriever(BaseRetriever):
    crossref_url = "https://api.crossref.org/journals/{issn}/works"
    openalex_url = "https://api.openalex.org/works"
    arxiv_url = "https://export.arxiv.org/api/query"
    request_headers = {
        "User-Agent": "zotero-arxiv-daily (https://github.com/TideDra/zotero-arxiv-daily)",
        "Accept": "application/json",
    }
    _tag = re.compile(r"<[^>]+>")
    _jats_abstract_heading = re.compile(r"^\s*abstract\s*[:.]?\s*", re.IGNORECASE)

    def __init__(self, config):
        super().__init__(config)
        rc = self.retriever_config
        get = (lambda k, d: rc.get(k, d)) if rc is not None else (lambda k, d: d)
        journals = get("journals", None) or DEFAULT_JOURNALS
        self.journals = [{"name": str(j["name"]), "issn": str(j["issn"])} for j in journals]
        self.lookback_days = int(get("lookback_days", 1) or 1)
        self.cutoff_shift_hours = float(get("cutoff_shift_hours", 12))
        self.max_per_journal = int(get("max_per_journal", 80) or 80)
        self.max_total = int(get("max_total", 400) or 400)
        self.arxiv_fallback_max = int(get("arxiv_fallback_max", 60) or 0)
        self.page_fallback_max = int(get("page_fallback_max", 60) or 0)
        mailto = get("mailto", None)
        if mailto:
            self.request_headers = dict(self.request_headers)
            self.request_headers["User-Agent"] += f" mailto:{mailto}"
        self.mailto = mailto
        self.fetch_counts: dict[str, int] = {}

    # ------------------------------------------------------------------ http
    def _get(self, url: str, params: dict[str, Any] | None = None, retry_num: int = 5, delay: int = 10, **kw):
        for i in range(retry_num):
            try:
                resp = requests.get(url, params=params, headers=kw.pop("headers", self.request_headers), timeout=60, **kw)
                resp.raise_for_status()
                return resp
            except Exception as exc:
                if i == retry_num - 1:
                    raise
                logger.warning(f"Request to {url} failed: {exc}. Retry in {delay} seconds.")
                sleep(delay)

    # ---------------------------------------------------------------- window
    def _window(self) -> tuple[datetime, datetime]:
        anchor = datetime.now(timezone.utc) - timedelta(hours=self.cutoff_shift_hours)
        end = anchor.replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=self.lookback_days)
        return start, end

    @staticmethod
    def _parse_date(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    # ---------------------------------------------------------------- text
    @classmethod
    def _clean_text(cls, text: str | None) -> str:
        if not text:
            return ""
        text = cls._tag.sub(" ", text)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return cls._jats_abstract_heading.sub("", text).strip()

    @staticmethod
    def _invert_abstract(inv: dict[str, list[int]] | None) -> str:
        if not inv:
            return ""
        pos = {}
        for word, idxs in inv.items():
            for i in idxs:
                pos[i] = word
        return " ".join(pos[i] for i in sorted(pos))

    # ------------------------------------------------------------- filtering
    def _is_research(self, journal: str, item: dict[str, Any]) -> bool:
        if item.get("type") != "journal-article":
            return False
        if item.get("update-to"):  # correction / retraction notices
            return False
        titles = item.get("title") or []
        title = self._clean_text(titles[0] if titles else "")
        if not title or NON_RESEARCH_TITLE.search(title):
            return False
        doi = (item.get("DOI") or "").lower()
        if doi.startswith("10.1038/d4"):  # Nature news, comment, careers, briefings
            return False
        if journal == "Science":
            # Science deposits news, letters, policy forum, perspectives etc. as journal-article;
            # research articles/reports are online-first (no pages yet) or >= 4 printed pages.
            page = item.get("page") or ""
            m = re.match(r"^\s*[A-Za-z]?(\d+)\s*-\s*[A-Za-z]?(\d+)\s*$", page)
            if page and (not m or int(m.group(2)) - int(m.group(1)) + 1 < 4):
                return False
        return True

    # ------------------------------------------------------------- crossref
    def _fetch_journal(self, journal: dict[str, str], start: datetime, end: datetime) -> list[JournalItem]:
        last_day = (end - timedelta(days=1)).date().isoformat()
        rows = 300
        resp = self._get(
            self.crossref_url.format(issn=journal["issn"]),
            {
                "filter": f"from-created-date:{start.date().isoformat()},until-created-date:{last_day},type:journal-article",
                "sort": "created",
                "order": "desc",
                "rows": rows,
            },
        )
        items = resp.json().get("message", {}).get("items", [])
        out = []
        for item in items:
            created = self._parse_date((item.get("created") or {}).get("date-time"))
            if created is None or not (start <= created < end):
                continue
            if not self._is_research(journal["name"], item):
                continue
            out.append(JournalItem(journal=journal["name"], item=item))
            if len(out) >= self.max_per_journal:
                break
        return out

    # ----------------------------------------------------------- enrichment
    def _enrich_openalex(self, items: list[JournalItem]) -> list[JournalItem]:
        missing = [it for it in items if not it.abstract]
        by_doi = {it.item["DOI"].lower(): it for it in missing}
        dois = list(by_doi)
        drop = set()
        for i in range(0, len(dois), 50):
            chunk = dois[i : i + 50]
            params = {"filter": "doi:" + "|".join(chunk), "per_page": 50, "select": "doi,abstract_inverted_index,type"}
            if self.mailto:
                params["mailto"] = self.mailto
            try:
                results = self._get(self.openalex_url, params, retry_num=3, delay=5).json().get("results", [])
            except Exception as exc:
                logger.warning(f"OpenAlex lookup failed: {exc}")
                continue
            for r in results:
                doi = (r.get("doi") or "").lower().removeprefix("https://doi.org/")
                it = by_doi.get(doi)
                if it is None:
                    continue
                abstract = self._invert_abstract(r.get("abstract_inverted_index"))
                if abstract:
                    it.abstract, it.abstract_source = abstract, "openalex"
                elif r.get("type") in NON_RESEARCH_OPENALEX_TYPES:
                    drop.add(doi)
            sleep(0.2)
        return [it for it in items if it.item["DOI"].lower() not in drop]

    def _enrich_nature_pages(self, items: list[JournalItem]) -> None:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; zotero-arxiv-daily)"})
        n = 0
        for it in items:
            if it.abstract or not it.item["DOI"].lower().startswith("10.1038/"):
                continue
            if n >= self.page_fallback_max:
                break
            n += 1
            try:
                resp = session.get(f"https://www.nature.com/articles/{it.item['DOI'].split('/', 1)[1]}", timeout=30)
                m = re.search(r'<meta name="dc.description" content="([^"]+)"', resp.text)
                if m:
                    it.abstract, it.abstract_source = self._clean_text(m.group(1)), "nature.com"
            except Exception as exc:
                logger.debug(f"nature.com lookup failed for {it.item['DOI']}: {exc}")
            sleep(0.5)

    def _enrich_arxiv(self, items: list[JournalItem]) -> None:
        n = 0
        for it in items:
            if it.abstract or not it.item["DOI"].startswith("10.1103/"):
                continue
            if n >= self.arxiv_fallback_max:
                break
            n += 1
            title = self._clean_text((it.item.get("title") or [""])[0])
            words = re.findall(r"[A-Za-z0-9]+", title)[:8]
            if len(words) < 3:
                continue
            query = 'ti:"' + " ".join(words) + '"'  # phrase search on the leading words
            try:
                resp = self._get(self.arxiv_url, {"search_query": query, "max_results": 5}, retry_num=2, delay=5)
                feed = feedparser.parse(resp.content)
                target = normalize_title(title)
                for e in feed.entries:
                    if SequenceMatcher(None, normalize_title(e.title), target).ratio() >= 0.9:
                        it.abstract, it.abstract_source = self._clean_text(e.summary), "arxiv"
                        break
            except Exception as exc:
                logger.debug(f"arXiv lookup failed for {it.item['DOI']}: {exc}")
            sleep(3)  # arXiv API etiquette

    # ------------------------------------------------------------- main API
    def _retrieve_raw_papers(self) -> list[JournalItem]:
        start, end = self._window()
        logger.info(f"Journal window (UTC, by Crossref created date): {start.date()} to {(end - timedelta(days=1)).date()}")
        collection: list[JournalItem] = []
        for journal in self.journals:
            try:
                got = self._fetch_journal(journal, start, end)
            except Exception as exc:
                logger.warning(f"Failed to fetch {journal['name']} ({journal['issn']}): {exc}")
                got = []
            if self.config.executor.debug:
                got = got[:10]
            self.fetch_counts[journal["name"]] = len(got)
            collection.extend(got)
            sleep(0.5)
        collection = collection[: self.max_total]

        for it in collection:
            it.abstract = self._clean_text(it.item.get("abstract"))
            if it.abstract:
                it.abstract_source = "crossref"
        collection = self._enrich_openalex(collection)
        self._enrich_nature_pages(collection)
        self._enrich_arxiv(collection)

        sources: dict[str, int] = {}
        for it in collection:
            sources[it.abstract_source or "title-only"] = sources.get(it.abstract_source or "title-only", 0) + 1
        counts = ", ".join(f"{k}: {v}" for k, v in self.fetch_counts.items())
        logger.info(f"Journal candidates per journal: {counts}")
        logger.info(f"Journal abstract sources: {sources}")
        if not collection:
            logger.warning("No journal paper found in the window.")
        return collection

    @staticmethod
    def _author_name(author: dict[str, Any]) -> str:
        if author.get("name"):
            return author["name"].strip()
        return f"{author.get('given', '')} {author.get('family', '')}".strip()

    def convert_to_paper(self, raw: JournalItem) -> Paper | None:
        item = raw.item
        doi = item["DOI"]
        title = self._clean_text((item.get("title") or [""])[0])
        if not title:
            return None
        authors = [a for a in (self._author_name(x) for x in item.get("author", [])) if a]
        affiliations = []
        for a in item.get("author", []):
            for aff in a.get("affiliation", []) or []:
                name = (aff.get("name") or "").strip()
                if name and name not in affiliations:
                    affiliations.append(name)
        url = f"https://doi.org/{doi}"
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=raw.abstract or title,  # reranker needs text; fall back to title
            url=url,
            pdf_url=url,  # usually paywalled: link to the DOI landing page instead of a PDF
            full_text=None,  # no scraping of paywalled full text; TL;DR uses the abstract
            affiliations=affiliations or None,
            venue=raw.journal,
            doi=doi,
        )

    def retrieve_papers(self) -> list[Paper]:
        # Same as BaseRetriever.retrieve_papers but without the per-paper sleep:
        # conversion here is purely local.
        papers = []
        for raw in self._retrieve_raw_papers():
            try:
                p = self.convert_to_paper(raw)
            except Exception as exc:
                logger.warning(f"Skipping journal item {raw.item.get('DOI')}: {exc}")
                continue
            if p is not None:
                papers.append(p)
        return papers
