"""Sanity tests for JournalRetriever (Crossref + OpenAlex fallback) and cross-source dedup."""

from datetime import datetime, timezone
from types import SimpleNamespace

import requests
from omegaconf import open_dict

from zotero_arxiv_daily.executor import deduplicate_papers
from zotero_arxiv_daily.protocol import Paper
from zotero_arxiv_daily.retriever import get_retriever_cls
from zotero_arxiv_daily.retriever.journal_retriever import JournalRetriever, DEFAULT_JOURNALS
from zotero_arxiv_daily.construct_email import render_email

FIXED_NOW = datetime(2026, 10, 19, 22, 0, 0, tzinfo=timezone.utc)  # 22:00 UTC cron on 10-19 -> covers 10-18


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return FIXED_NOW if tz is None else FIXED_NOW.astimezone(tz)


def _item(doi, title, created, abstract=None, **extra):
    d = {
        "DOI": doi,
        "type": "journal-article",
        "title": [title],
        "created": {"date-time": created},
        "author": [{"given": "A", "family": "Bee", "affiliation": [{"name": "Some Univ"}]}],
    }
    if abstract:
        d["abstract"] = abstract
    d.update(extra)
    return d


CROSSREF = [
    _item("10.1103/abc1", "Memory kernels in time-dependent density functional theory", "2026-10-18T10:00:00Z"),
    _item("10.1103/abc2", "Erratum: Something wrong", "2026-10-18T11:00:00Z"),
    _item("10.1103/abc3", "Too old paper", "2026-10-17T23:59:00Z", abstract="<jats:p>old</jats:p>"),
    _item("10.1103/abc4", "Today paper not yet in window", "2026-10-19T00:30:00Z", abstract="x"),
    _item("10.1103/abc7", "Same day as run", "2026-10-19T21:00:00Z", abstract="x"),
    _item("10.1103/abc5", "Correction notice", "2026-10-18T12:00:00Z", **{"update-to": [{"DOI": "10.1103/x"}]}),
    _item("10.1103/abc6", "Berry connection in nonadiabatic dynamics", "2026-10-18T13:00:00Z",
          abstract="<jats:title>Abstract</jats:title><jats:p>We study <i>Berry</i> phases.</jats:p>"),
    _item("10.1038/d41586-026-1", "Daily briefing: news", "2026-10-18T13:00:00Z"),
]


def _patch(monkeypatch, calls):
    def _get(url, params=None, **kwargs):
        calls.append((url, params))
        if "api.crossref.org" in url:
            payload = {"message": {"items": CROSSREF if "0031-9007" in url else []}}
        elif "api.openalex.org" in url:
            payload = {"results": [{"doi": "https://doi.org/10.1103/abc1", "type": "article",
                                    "abstract_inverted_index": {"We": [0], "derive": [1], "kernels": [2]}}]}
        else:
            raise AssertionError(url)
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: payload, content=b"")
    monkeypatch.setattr(requests, "get", _get)


def _retriever(config):
    with open_dict(config.source):
        config.source.journal = {"journals": [{"name": "Physical Review Letters", "issn": "0031-9007"},
                                              {"name": "Nature", "issn": "0028-0836"}],
                                 "lookback_days": 1, "arxiv_fallback_max": 0, "page_fallback_max": 0}
    return JournalRetriever(config)


def test_registered_and_defaults():
    assert get_retriever_cls("journal") is JournalRetriever
    assert len(DEFAULT_JOURNALS) == 12


def test_journal_window_filter_and_abstracts(config, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.journal_retriever.datetime", _FixedDatetime)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.journal_retriever.sleep", lambda _: None)
    calls = []
    _patch(monkeypatch, calls)
    r = _retriever(config)
    papers = r.retrieve_papers()
    assert [p.title for p in papers] == [
        "Memory kernels in time-dependent density functional theory",
        "Berry connection in nonadiabatic dynamics",
    ]
    assert papers[0].abstract == "We derive kernels"  # OpenAlex fallback
    assert papers[1].abstract == "We study Berry phases."  # Crossref JATS cleaned
    assert all(p.venue == "Physical Review Letters" for p in papers)
    assert papers[0].pdf_url == "https://doi.org/10.1103/abc1"
    assert papers[0].affiliations == ["Some Univ"]
    cr = [p for u, p in calls if "crossref" in u][0]
    assert cr["filter"].startswith("from-created-date:2026-10-18,until-created-date:2026-10-18")
    assert r.fetch_counts == {"Physical Review Letters": 2, "Nature": 0}


def test_dedup_prefers_journal_and_keeps_fulltext():
    pre = Paper(source="arxiv", title="Berry Connection in Nonadiabatic Dynamics", authors=[], abstract="pre abs",
                url="u", pdf_url="arxiv.pdf", full_text="FULL")
    jr = Paper(source="journal", title="Berry connection in nonadiabatic dynamics", authors=[],
               abstract="Berry connection in nonadiabatic dynamics", url="https://doi.org/x", pdf_url="https://doi.org/x",
               venue="PRL", doi="10.1/x")
    out = deduplicate_papers([pre, jr])
    assert len(out) == 1 and out[0].venue == "PRL"
    assert out[0].full_text == "FULL" and out[0].abstract == "pre abs"


def test_email_shows_journal():
    p = Paper(source="journal", title="T", authors=["A"], abstract="a", url="https://doi.org/x",
              pdf_url="https://doi.org/x", tldr="tl", score=7.0, venue="Physical Review B")
    html = render_email([p])
    assert "Physical Review B" in html and ">DOI</a>" in html


def test_window_robust_to_cron_delay(config, monkeypatch):
    r = _retriever(config)
    for now in [datetime(2026, 10, 19, 22, 0, tzinfo=timezone.utc),   # on time
                datetime(2026, 10, 20, 3, 30, tzinfo=timezone.utc),   # delayed past midnight
                datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)]:  # early
        _D = type("_D", (datetime,), {"now": classmethod(lambda cls, tz=None, _n=now: _n)})
        monkeypatch.setattr("zotero_arxiv_daily.retriever.journal_retriever.datetime", _D)
        start, end = r._window()
        assert (start.date().isoformat(), end.date().isoformat()) == ("2026-10-18", "2026-10-19")
