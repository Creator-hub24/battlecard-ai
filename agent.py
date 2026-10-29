#!/usr/bin/env python3
"""AI GTM Competitive-Intelligence Agent.

Monitor -> Analyze -> Synthesize:

1. Monitor:    load competitor pricing pages and changelogs from local files or live URLs
               (requests + BeautifulSoup, or Firecrawl when FIRECRAWL_API_KEY is set), normalise them
               to markdown-like text and hash them so we know what changed since the last run.
2. Analyze:    with GEMINI_API_KEY set, Gemini returns a structured JSON analysis (pydantic schema)
               that is fact-checked against the source text. Without a key (or with --offline) a
               deterministic heuristic analyzer extracts plans, prices, feature signals and weak points.
3. Synthesize: render a markdown Sales Battlecard: TL;DR, what changed, feature matrix, pricing,
               landmine questions, objection handling, win themes and sources.

Run `python agent.py --help` for options.
"""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import hashlib
import json
import logging
import os
import re
import sys
import time
import urllib.robotparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import requests
import yaml
from bs4 import BeautifulSoup, Comment, NavigableString
from pydantic import BaseModel, Field, ValidationError

__version__ = "1.0.0"
ROOT = Path(__file__).resolve().parent
log = logging.getLogger("gtm_agent")

USER_AGENT = (f"ai-gtm-competitive-agent/{__version__} "
              "(+https://github.com/Creator-hub24/ai-gtm-competitive-agent)")
FIRECRAWL_URL = "https://api.firecrawl.dev/v2/scrape"
HTTP_TIMEOUT = 20

Status = Literal["yes", "partial", "no", "not_found"]
STATUS_ICON = {"yes": "✅ Yes", "partial": "🟡 Partial", "no": "❌ No", "not_found": "➖ Not mentioned"}


# ════════════════════════════════════════════════════════════════════════════
# Data models (shared by the offline and Gemini analyzers)
# ════════════════════════════════════════════════════════════════════════════
class Plan(BaseModel):
    name: str
    price: Optional[float] = None
    period: Optional[str] = None
    is_addon: bool = False
    highlights: list[str] = Field(default_factory=list)

    @property
    def monthly_equiv(self) -> Optional[float]:
        return monthly_equivalent(self.price, self.period)


class Change(BaseModel):
    date: Optional[str] = Field(None, description="ISO date YYYY-MM-DD if known")
    title: str
    category: Literal["pricing", "feature_added", "feature_removed", "improvement",
                      "reliability", "other"] = "other"
    detail: str = ""
    evidence: str = Field("", description="Verbatim sentence from the source")
    source: str = ""


class PricingShift(BaseModel):
    plan: str
    old_price: Optional[float] = None
    new_price: Optional[float] = None
    period: Optional[str] = None
    evidence: str = ""

    @property
    def direction(self) -> str:
        if self.old_price is None or self.new_price is None:
            return "new" if self.old_price is None else "removed"
        if self.new_price > self.old_price:
            return "up"
        return "down" if self.new_price < self.old_price else "same"

    @property
    def pct(self) -> Optional[float]:
        if self.old_price and self.new_price is not None:
            return (self.new_price - self.old_price) / self.old_price * 100
        return None


class FeatureShift(BaseModel):
    feature: str
    change: Literal["added", "removed", "improved", "limited"]
    evidence: str = ""


class FeatureStatus(BaseModel):
    feature_key: str
    status: Status
    evidence: str = ""


class WeakPoint(BaseModel):
    point: str
    evidence: str = ""
    source: str = ""


class Objection(BaseModel):
    objection: str
    response: str


class CompetitorAnalysis(BaseModel):
    competitor_id: str
    name: str
    mode: Literal["gemini", "offline"]
    summary: str = ""
    positioning: str = ""
    what_changed: list[Change] = Field(default_factory=list)
    pricing_shifts: list[PricingShift] = Field(default_factory=list)
    feature_shifts: list[FeatureShift] = Field(default_factory=list)
    feature_status: dict[str, FeatureStatus] = Field(default_factory=dict)
    weak_points: list[WeakPoint] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    plans: list[Plan] = Field(default_factory=list)
    landmine_questions: list[str] = Field(default_factory=list)
    objections: list[Objection] = Field(default_factory=list)
    threat: Literal["High", "Medium", "Low"] = "Medium"
    dropped_claims: int = 0  # Gemini claims removed by the fact-check guardrail


# Schema the LLM must return (no dicts/properties, which keeps it Gemini-schema friendly).
class LLMAnalysis(BaseModel):
    summary: str = Field(description="2-3 sentence summary of the competitor's current position")
    positioning: str = Field(description="One line: who they sell to and how")
    what_changed: list[Change] = Field(description="Dated changes inside the lookback window")
    pricing_shifts: list[PricingShift] = Field(description="Price changes with old and new price")
    feature_shifts: list[FeatureShift]
    feature_status: list[FeatureStatus] = Field(description="One entry per taxonomy feature key")
    weak_points: list[WeakPoint] = Field(description="Exploitable gaps, each with verbatim evidence")
    strengths: list[str]
    landmine_questions: list[str] = Field(description="Questions a rep asks the buyer to expose gaps")
    objections: list[Objection] = Field(description="Must include 'Competitor is cheaper' handling")


# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Source:
    kind: str
    url: Optional[str] = None
    local: Optional[str] = None
    baseline: Optional[str] = None


@dataclass
class Competitor:
    id: str
    name: str
    sources: list[Source]
    mock: bool = False


@dataclass
class Config:
    us: dict
    competitors: list[Competitor]
    features: dict[str, dict]
    settings: dict = field(default_factory=dict)
    base_dir: Path = ROOT

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.base_dir / p


def _norm_status(value: Any) -> str:
    """YAML 1.1 turns bare yes/no into booleans; accept both."""
    if value is True:
        return "yes"
    if value is False or value is None:
        return "no"
    v = str(value).strip().lower()
    return v if v in ("yes", "partial", "no") else "no"


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    base = Path(path).resolve().parent
    comps = []
    for c in raw.get("competitors", []):
        sources = [Source(**s) for s in c.get("sources", [])]
        if not sources:
            raise ValueError(f"competitor {c.get('id')} has no sources")
        comps.append(Competitor(id=c["id"], name=c["name"], sources=sources, mock=bool(c.get("mock"))))
    us_cfg = raw["us"]
    profile_path = Path(us_cfg["profile"])
    profile = yaml.safe_load((profile_path if profile_path.is_absolute() else base / profile_path)
                             .read_text(encoding="utf-8"))
    profile["features"] = {k: _norm_status(v) for k, v in (profile.get("features") or {}).items()}
    unknown = set(profile["features"]) - set(raw["features"])
    if unknown:
        raise ValueError(f"profile features not in taxonomy: {sorted(unknown)}")
    return Config(us=profile, competitors=comps, features=raw["features"],
                  settings=raw.get("settings", {}), base_dir=base)


# ════════════════════════════════════════════════════════════════════════════
# 1. MONITOR: fetch, normalise, hash, cache
# ════════════════════════════════════════════════════════════════════════════
def html_to_text(html: str) -> str:
    """Convert HTML to markdown-like text: '#' headings, '- ' bullets, table rows with '|'."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["head", "script", "style", "noscript", "svg", "nav", "footer", "iframe", "form"]):
        tag.decompose()
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    for tag in soup.find_all(re.compile(r"^h[1-6]$")):
        level = int(tag.name[1])
        tag.replace_with(NavigableString(f"\n{'#' * level} {tag.get_text(' ', strip=True)}\n"))
    for tag in soup.find_all("li"):
        tag.replace_with(NavigableString(f"\n- {tag.get_text(' ', strip=True)}\n"))
    for tr in soup.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        tr.replace_with(NavigableString("\n| " + " | ".join(cells) + " |\n"))
    for tag in soup.find_all(["p", "div", "section", "article", "br", "main"]):
        tag.insert_before(NavigableString("\n"))
        tag.insert_after(NavigableString("\n"))
    return normalise_text(soup.get_text())


def normalise_text(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    lines = [re.sub(r"[ \t\u00a0]+", " ", ln).strip() for ln in text.splitlines()]
    out, blank = [], False
    for ln in lines:
        if not ln:
            if not blank and out:
                out.append("")
            blank = True
            continue
        out.append(ln)
        blank = False
    return "\n".join(out).strip() + "\n"


def to_text(raw: str, hint: str = "") -> str:
    looks_html = hint.endswith((".html", ".htm")) or bool(re.search(r"<(html|body|div|p|h\d)\b", raw, re.I))
    return html_to_text(raw) if looks_html else normalise_text(raw)


def robots_allowed(url: str, session: requests.Session) -> bool:
    parts = urlparse(url)
    rp = urllib.robotparser.RobotFileParser()
    try:
        r = session.get(f"{parts.scheme}://{parts.netloc}/robots.txt", timeout=HTTP_TIMEOUT)
        if r.status_code >= 400:
            return True  # no robots.txt -> allowed
        rp.parse(r.text.splitlines())
        return rp.can_fetch(USER_AGENT, url)
    except requests.RequestException:
        return True


def fetch_url(url: str, session: Optional[requests.Session] = None, retries: int = 2) -> str:
    """Fetch a public page as text. Uses Firecrawl if FIRECRAWL_API_KEY is set, else requests+bs4."""
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", USER_AGENT)
    fc_key = os.getenv("FIRECRAWL_API_KEY")
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if fc_key:
                r = session.post(FIRECRAWL_URL, timeout=60,
                                 headers={"Authorization": f"Bearer {fc_key}"},
                                 json={"url": url, "formats": ["markdown"], "onlyMainContent": True})
                r.raise_for_status()
                data = r.json()
                if not data.get("success"):
                    raise RuntimeError(f"Firecrawl error: {data.get('error')}")
                return normalise_text(data["data"]["markdown"])
            if attempt == 0 and not robots_allowed(url, session):
                raise PermissionError(f"robots.txt disallows {url}")
            r = session.get(url, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            return to_text(r.text, url)
        except PermissionError:
            raise
        except Exception as e:  # network / HTTP / Firecrawl errors -> retry with backoff
            last = e
            log.warning("fetch %s failed (attempt %d/%d): %s", url, attempt + 1, retries + 1, e)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}: {last}")


@dataclass
class Snapshot:
    competitor_id: str
    kind: str
    origin: str               # URL or local path shown in the battlecard
    text: str
    retrieved_at: str
    sha256: str
    status: str = "new"       # new | changed | unchanged
    first_seen: str = ""
    baseline_text: Optional[str] = None
    previous_plans: Optional[list[dict]] = None


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Cache:
    """Tiny JSON state file: hash + extracted plans per source, for change detection."""

    def __init__(self, path: Path, enabled: bool = True):
        self.path, self.enabled = path, enabled
        self.data: dict[str, Any] = {}
        if enabled and path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("cache %s is corrupt, starting fresh", path)

    @staticmethod
    def key(cid: str, kind: str, origin: str) -> str:
        return f"{cid}|{kind}|{origin}"

    def get(self, key: str) -> dict:
        return self.data.get(key, {})

    def put(self, key: str, value: dict) -> None:
        self.data[key] = value

    def save(self) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")


def monitor(comp: Competitor, cfg: Config, cache: Cache, live: bool, now: str,
            session: Optional[requests.Session] = None) -> list[Snapshot]:
    snaps = []
    for src in comp.sources:
        if live and src.url:
            origin, text = src.url, fetch_url(src.url, session)
        elif src.local:
            p = cfg.path(src.local)
            origin, text = src.local, to_text(p.read_text(encoding="utf-8"), p.name)
        else:
            log.warning("%s: source %s has no local file and --live is off; skipped", comp.id, src.url)
            continue
        h = content_hash(text)
        key = Cache.key(comp.id, src.kind, origin)
        prev = cache.get(key)
        status = "new" if not prev else ("unchanged" if prev.get("sha256") == h else "changed")
        baseline = None
        if src.baseline:
            bp = cfg.path(src.baseline)
            baseline = to_text(bp.read_text(encoding="utf-8"), bp.name)
        snaps.append(Snapshot(comp.id, src.kind, origin, text, now, h, status,
                              prev.get("first_seen", now), baseline,
                              prev.get("plans") if status == "changed" else None))
        log.info("monitor %-10s %-9s %-9s %s", comp.id, src.kind, status, h[:12])
    return snaps


def update_cache(cache: Cache, snaps: list[Snapshot], analysis: CompetitorAnalysis) -> None:
    for s in snaps:
        key = Cache.key(s.competitor_id, s.kind, s.origin)
        prev = cache.get(key)
        cache.put(key, {
            "sha256": s.sha256, "first_seen": s.first_seen or s.retrieved_at, "last_seen": s.retrieved_at,
            "last_changed": s.retrieved_at if s.status != "unchanged" else prev.get("last_changed", s.retrieved_at),
            "plans": [p.model_dump() for p in analysis.plans] if s.kind == "pricing" else None,
        })


# ════════════════════════════════════════════════════════════════════════════
# 2a. ANALYZE (offline, deterministic heuristics)
# ════════════════════════════════════════════════════════════════════════════
PRICE_RE = re.compile(r"(?:₹|\bRs\.?\s?|\bINR\s?)\s?(\d[\d,]*(?:\.\d+)?)")
PERIOD_RE = re.compile(r"^\s*(?:/|per\s+|a\s+|every\s+)\s*(month|mo|year|yr|annum|quarter|week)\b", re.I)
PERIOD_MONTHS = {"month": 1, "mo": 1, "quarter": 3, "year": 12, "yr": 12, "annum": 12, "week": 12 / 52}
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
                                      "oct", "nov", "dec"], start=1)}
PARTIAL_MARKERS = re.compile(r"\b(beta|coming soon|coming in|on the roadmap|roadmap|add-on|only|limited|"
                             r"pilot|not yet|can (?:now )?add|(?:one|\d+)(?: [\w-]+){0,4} (?:per|every|a) (?:day|month))\b", re.I)
NEGATIVE_MARKERS = re.compile(r"\b(not included|not available|removed|deprecated|discontinued|no longer)\b",
                              re.I)

WEAK_RULES: list[tuple[str, re.Pattern]] = [
    ("Quality or reliability issue", re.compile(r"known issue|delayed|may be incorrect|outage|bug", re.I)),
    ("Feature removed or being retired", re.compile(r"\b(removed|deprecated|discontinued|no longer)\b", re.I)),
    ("Bigger batches, less attention per student",
     re.compile(r"batch size.*\b(up to|to|from)\b.*\d+\s+students|\bup to \d+ students", re.I)),
    ("Usage cap on the plan", re.compile(r"\b\d+ (?:[\w-]+ ){0,3}(?:per|a|every) (?:day|month)\b|reduced from",
                                         re.I)),
    ("Limited class coverage", re.compile(r"class(?:es)? \d+\s*(?:-|–|to)\s*\d+ only|only for class", re.I)),
    ("Promised but not shipped", re.compile(r"coming soon|coming in|on the roadmap|not available yet", re.I)),
    ("Beta or paid add-on, not core", re.compile(r"\b(beta|add-on)\b", re.I)),
    ("Introductory price that may not last", re.compile(r"festive offer|may be revised|limited.time", re.I)),
    ("Strict commercial terms", re.compile(r"no refunds?|auto-renew", re.I)),
]


def monthly_equivalent(price: Optional[float], period: Optional[str]) -> Optional[float]:
    if price is None:
        return None
    if price == 0:
        return 0.0
    months = PERIOD_MONTHS.get((period or "").lower())
    return round(price / months, 2) if months else None


def parse_price(s: str) -> float:
    return float(s.replace(",", ""))


def split_sections(text: str) -> list[tuple[int, str, list[str]]]:
    """[(heading_level, heading, body_lines)] for markdown-like text."""
    sections: list[tuple[int, str, list[str]]] = [(0, "", [])]
    for ln in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            sections.append((len(m.group(1)), m.group(2).strip(), []))
        elif ln.strip():
            sections[-1][2].append(ln.strip())
    return sections


def sentences(text: str, headings: bool = True) -> list[str]:
    out = []
    for ln in text.splitlines():
        if not headings and ln.lstrip().startswith("#"):
            continue
        ln = re.sub(r"^(#{1,6}|-|\|)\s*", "", ln.strip()).strip(" |")
        if not ln or "MOCK DATA" in ln:
            continue
        out.extend(s.strip() for s in re.split(r"(?<=[.!])\s+(?=[A-Z₹])", ln) if s.strip())
    return out


def extract_plans(text: str) -> list[Plan]:
    plans = []
    for _level, heading, body in split_sections(text):
        if not heading:
            continue
        blob = " ".join(body)
        m = PRICE_RE.search(blob)
        price, period = None, None
        if m:
            price = parse_price(m.group(1))
            pm = PERIOD_RE.match(blob[m.end():m.end() + 20])
            period = {"mo": "month", "yr": "year", "annum": "year"}.get(pm.group(1).lower(),
                                                                       pm.group(1).lower()) if pm else None
        elif re.search(r"\bfree\b", heading, re.I):
            price = 0.0
        if price is None:
            continue
        highlights = [b[2:] if b.startswith("- ") else b for b in body if b.startswith("- ")]
        plans.append(Plan(name=heading, price=price, period=period if price else "free",
                          is_addon=bool(re.search(r"add-?on", heading, re.I)), highlights=highlights))
    return plans


def parse_date(s: str) -> Optional[dt.date]:
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", s)
    if m:
        return dt.date(int(m[1]), int(m[2]), int(m[3]))
    m = re.search(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})\b", s)
    if m and m[2][:3].lower() in MONTHS:
        return dt.date(int(m[3]), MONTHS[m[2][:3].lower()], int(m[1]))
    m = re.search(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})\b", s)
    if m and m[1][:3].lower() in MONTHS:
        return dt.date(int(m[3]), MONTHS[m[1][:3].lower()], int(m[2]))
    return None


CHANGE_RULES = [  # checked against the title first, then the body
    ("reliability", re.compile(r"known issue|delayed|outage|incorrect|\bbug", re.I)),
    ("pricing", re.compile(r"₹|\bRs\.?\s?\d|\bpric(e|es|ing)\b", re.I)),
    ("feature_removed", re.compile(r"\b(removed|deprecated|discontinued|sunset|moves? to)\b", re.I)),
    ("feature_added", re.compile(r"\b(launch(ed)?|introducing|new|now available|generally available|"
                                 r"now live|are now|can now)\b", re.I)),
    ("improvement", re.compile(r"\b(improv\w*|faster|increased|bigger|upgrade\w*)\b", re.I)),
]


def classify_change(title: str, body: str) -> str:
    for text in (title, body):
        for cat, pat in CHANGE_RULES:
            if pat.search(text):
                return cat
    return "other"


def extract_changes(text: str, origin: str, since: dt.date, as_of: dt.date) -> list[Change]:
    changes = []
    for _level, heading, body in split_sections(text):
        d = parse_date(heading)
        if not d or not (since <= d <= as_of):
            continue
        title = re.sub(r"^\s*[^A-Za-z]*?(\d{4}-\d{2}-\d{2}|\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},\s+\d{4})"
                       r"\s*[:·—–-]*\s*", "", heading).strip() or heading
        detail = " ".join(body)
        first = sentences(detail)[:1]
        changes.append(Change(date=d.isoformat(), title=title, category=classify_change(title, detail),
                              detail=detail, evidence=first[0] if first else title, source=origin))
    return sorted(changes, key=lambda c: c.date or "", reverse=True)


def kw_pattern(keyword: str) -> re.Pattern:
    """Whole-word, case-insensitive, tolerant of plurals ('mock test' matches 'Mock tests')."""
    return re.compile(r"\b" + re.escape(keyword) + r"(?:s|es)?\b", re.I)


def mentions(text: str, keywords: list[str]) -> bool:
    return any(kw_pattern(k).search(text) for k in keywords)


def detect_features(text: str, taxonomy: dict[str, dict]) -> dict[str, FeatureStatus]:
    # Body sentences first so evidence quotes a claim, not a heading; FAQ questions are not claims.
    body = sentences(text, headings=False)
    sents = [s for s in body + [h for h in sentences(text) if h not in body] if not s.endswith("?")]
    out = {}
    for key, spec in taxonomy.items():
        found: dict[str, str] = {}
        pats = [kw_pattern(k) for k in spec.get("keywords", [])]
        ppats = [kw_pattern(k) for k in spec.get("partial_keywords", [])]
        for s in sents:
            if any(p.search(s) for p in ppats):
                found.setdefault("partial", s)
            elif any(p.search(s) for p in pats):
                if NEGATIVE_MARKERS.search(s):
                    found.setdefault("no", s)
                elif PARTIAL_MARKERS.search(s):
                    found.setdefault("partial", s)
                else:
                    found.setdefault("yes", s)
        for st in ("yes", "partial", "no"):
            if st in found:
                out[key] = FeatureStatus(feature_key=key, status=st, evidence=found[st])
                break
        else:
            out[key] = FeatureStatus(feature_key=key, status="not_found")
    return out


def diff_plans(old: list[Plan], new: list[Plan], evidence: str) -> list[PricingShift]:
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()  # noqa: E731
    old_by = {norm(p.name): p for p in old}
    shifts = []
    for p in new:
        o = old_by.get(norm(p.name))
        if o is None:
            if old:
                shifts.append(PricingShift(plan=p.name, new_price=p.price, period=p.period, evidence=evidence))
        elif o.price != p.price:
            shifts.append(PricingShift(plan=p.name, old_price=o.price, new_price=p.price,
                                       period=p.period, evidence=evidence))
    new_names = {norm(p.name) for p in new}
    shifts += [PricingShift(plan=o.name, old_price=o.price, period=o.period, evidence=evidence)
               for o in old if norm(o.name) not in new_names]
    return shifts


def find_weak_points(snaps: list[Snapshot]) -> list[WeakPoint]:
    out, seen = [], set()
    for s in snaps:
        for sent in sentences(s.text, headings=False):
            if sent.endswith("?") or sent in seen:
                continue
            bare = PRICE_RE.sub("PRICE", sent)  # "₹1,999 per month" is a price, not a usage cap
            for label, pat in WEAK_RULES:
                if pat.search(bare):
                    out.append(WeakPoint(point=label, evidence=sent, source=s.origin))
                    seen.add(sent)
                    break
    order = {label: i for i, (label, _) in enumerate(WEAK_RULES)}
    return sorted(out, key=lambda w: order.get(w.point, 99))


def fmt_inr(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    if x == int(x):
        n = int(x)
        s = str(n)
        if len(s) > 3:  # Indian grouping: 1,23,456
            head, tail = s[:-3], s[-3:]
            head = re.sub(r"(\d)(?=(\d{2})+$)", r"\1,", head)
            s = f"{head},{tail}"
        return f"₹{s}"
    return f"₹{x:,.0f}"


def lc_first(label: str) -> str:
    """'Live classes' -> 'live classes', but keep proper nouns like 'Hindi' or 'Class 6-10'."""
    if label.split(" ", 1)[0] in {"Hindi", "Class"}:
        return label
    return label[:1].lower() + label[1:]


def plan_floor(plans: list[Plan]) -> Optional[Plan]:
    """Lowest effective monthly price among paid, non add-on plans."""
    paid = [p for p in plans if not p.is_addon and p.price and p.monthly_equiv is not None]
    return min(paid, key=lambda p: p.monthly_equiv) if paid else None


def monthly_plan(plans: list[Plan]) -> Optional[Plan]:
    m = [p for p in plans if not p.is_addon and p.price and (p.period or "") == "month"]
    return min(m, key=lambda p: p.price) if m else None


def us_plans(cfg: Config) -> list[Plan]:
    return [Plan(name=p["name"], price=float(p["price"]), period=str(p.get("period", "month")))
            for p in cfg.us.get("plans", [])]


class Playbook:
    """Turns facts (plans, feature statuses, weak points) into sales plays. Used by both analyzers."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.us_name = cfg.us["name"]
        self.us_feat = cfg.us["features"]
        self.us_plans = us_plans(cfg)

    def gaps(self, a: CompetitorAnalysis) -> list[str]:
        """Taxonomy keys where we are 'yes' and they are not."""
        return [k for k, st in self.us_feat.items()
                if st == "yes" and a.feature_status.get(k) and a.feature_status[k].status != "yes"]

    def their_edges(self, a: CompetitorAnalysis) -> list[str]:
        return [k for k, fs in a.feature_status.items() if fs.status == "yes" and self.us_feat.get(k) != "yes"]

    def landmines(self, a: CompetitorAnalysis) -> list[str]:
        qs = [self.cfg.features[k]["landmine"] for k in self.gaps(a) if self.cfg.features[k].get("landmine")]
        if any(ps.direction == "up" for ps in a.pricing_shifts):
            qs.insert(0, "What will you pay at renewal? Has the price of the plan you are looking at "
                         "changed in the last few months?")
        if any(w.point.startswith("Introductory") for w in a.weak_points):
            qs.insert(0, "Is the price you were quoted locked in for renewal, or is it a festive or "
                         "introductory offer?")
        if any(w.point.startswith("Quality") for w in a.weak_points):
            qs.append("Do they publish known issues? Ask how long open issues (missing or wrong solutions) "
                      "have been unresolved.")
        return list(dict.fromkeys(qs))[:6]

    def objections(self, a: CompetitorAnalysis) -> list[Objection]:
        out = [self._price_objection(a)]
        counter = self.cfg.us.get("counter_positioning", {})
        for k in self.their_edges(a)[:2]:
            if k in counter:
                out.append(Objection(objection=f"\"{a.name} has {lc_first(self.cfg.features[k]['label'])}.\"",
                                     response=counter[k]))
        diffs = {d["key"]: d for d in self.cfg.us.get("differentiators", [])}
        for fs in a.feature_shifts:
            k = self._feature_key_for(fs.feature)
            if fs.change == "added" and k in diffs and self.us_feat.get(k) == "yes":
                land = self.cfg.features[k].get("landmine", "")
                out.append(Objection(
                    objection=f"\"{a.name} just launched {lc_first(self.cfg.features[k]['label'])} too.\"",
                    response=f"{diffs[k]['proof']} Ask them to show it live: {land}"))
                break
        return out[:4]

    def _feature_key_for(self, text: str) -> Optional[str]:
        for k, spec in self.cfg.features.items():
            if k == text or text == spec["label"] or mentions(text, spec.get("keywords", [])):
                return k
        return None

    def _price_objection(self, a: CompetitorAnalysis) -> Objection:
        q = f"\"{a.name} is cheaper.\""
        their_floor, our_floor = plan_floor(a.plans), plan_floor(self.us_plans)
        their_month, our_month = monthly_plan(a.plans), monthly_plan(self.us_plans)
        if not their_floor or not our_floor:
            return Objection(objection=q, response="Ask for their full price list in writing, including "
                             "add-ons and renewal price, then compare plan for plan.")
        gap_labels = [lc_first(self.cfg.features[k]["label"]) for k in self.gaps(a)][:3]
        gaps_txt = "; ".join(gap_labels) if gap_labels else "the same depth of board-exam practice"
        if their_floor.monthly_equiv < our_floor.monthly_equiv:
            addons = [p for p in a.plans if p.is_addon and p.price and p.monthly_equiv]
            parts = [f"On sticker price, yes: {a.name} starts at {fmt_inr(their_floor.monthly_equiv)}/month "
                     f"({their_floor.name}) vs our {fmt_inr(our_floor.monthly_equiv)}/month "
                     f"({our_floor.name})."]
            if their_month and addons:
                eff = their_month.price + sum(p.monthly_equiv for p in addons)
                parts.append(f"But compare like for like: {their_month.name} plus "
                             f"{', '.join(p.name for p in addons)} costs {fmt_inr(eff)}/month, against "
                             f"{fmt_inr(our_month.price if our_month else our_floor.monthly_equiv)}/month "
                             f"for our {our_month.name if our_month else our_floor.name} with everything included.")
            parts.append(f"What the lower price does not include: {gaps_txt}.")
            if any(w.point.startswith("Introductory") for w in a.weak_points):
                parts.append("Their current price is a promotional offer that may be revised.")
            parts.append("Reframe to cost per mark gained: one board-pattern mock with a weakness report "
                         "is worth more than a month of unlimited easy questions.")
            return Objection(objection=q, response=" ".join(parts))
        pct = (1 - our_floor.monthly_equiv / their_floor.monthly_equiv) * 100
        resp = (f"Check the numbers: we are actually {pct:.0f}% cheaper. {a.name} starts at "
                f"{fmt_inr(their_floor.monthly_equiv)}/month ({their_floor.name}); we start at "
                f"{fmt_inr(our_floor.monthly_equiv)}/month ({our_floor.name}).")
        if their_month and our_month:
            resp += (f" Month to month it is {fmt_inr(their_month.price)} vs {fmt_inr(our_month.price)}.")
        resp += f" And we include {gaps_txt}."
        return Objection(objection=q, response=resp)

    def win_themes(self, a: CompetitorAnalysis) -> list[dict]:
        diffs = self.cfg.us.get("differentiators", [])
        hit = [d for d in diffs if a.feature_status.get(d["key"]) and a.feature_status[d["key"]].status != "yes"]
        return (hit or diffs)[:3]

    def threat(self, a: CompetitorAnalysis) -> str:
        score = 0
        tf, of = plan_floor(a.plans), plan_floor(self.us_plans)
        if tf and of and tf.monthly_equiv < of.monthly_equiv:
            score += 2
        score += len(self.their_edges(a))
        overlap = sum(1 for k, fs in a.feature_status.items() if fs.status == "yes" and self.us_feat.get(k) == "yes")
        score += 1 if overlap >= 3 else 0
        score += sum(1 for fs in a.feature_shifts if fs.change == "added")
        return "High" if score >= 5 else "Medium" if score >= 3 else "Low"

    def positioning(self, a: CompetitorAnalysis) -> str:
        tf, of = plan_floor(a.plans), plan_floor(self.us_plans)
        tier = "Similar-priced"
        if tf and of:
            r = tf.monthly_equiv / of.monthly_equiv
            tier = "Budget" if r < 0.8 else "Premium" if r > 1.5 else "Similar-priced"
        yes = [k for k, fs in a.feature_status.items() if fs.status == "yes"]
        edges = [self.cfg.features[k]["label"] for k in sorted(yes, key=lambda k: self.us_feat.get(k) == "yes")]
        prices = sorted(p.monthly_equiv for p in a.plans if p.price and not p.is_addon and p.monthly_equiv)
        rng = (f" ({fmt_inr(prices[0])}-{fmt_inr(prices[-1])}/month)" if len(prices) > 1
               else f" ({fmt_inr(prices[0])}/month)" if prices else "")
        lead = ", ".join(lc_first(e) for e in edges[:3]) or "no clear feature focus"
        return f"{tier} player{rng}; leads with {lead}."


def analyze_offline(comp: Competitor, snaps: list[Snapshot], cfg: Config, as_of: dt.date,
                    lookback_days: int) -> CompetitorAnalysis:
    since = as_of - dt.timedelta(days=lookback_days)
    corpus = "\n".join(s.text for s in snaps)
    plans, changes, shifts = [], [], []
    for s in snaps:
        if s.kind == "pricing":
            cur = extract_plans(s.text)
            plans.extend(cur)
            old = (extract_plans(s.baseline_text) if s.baseline_text
                   else [Plan(**p) for p in s.previous_plans] if s.previous_plans else [])
            ev = f"Pricing page diff vs previous snapshot ({s.origin})"
            for ps in diff_plans(old, cur, ev):
                shifts.append(ps)
                per = f"/{ps.period}" if ps.period else ""
                title = {"new": f"New plan: {ps.plan} at {fmt_inr(ps.new_price)}{per}",
                         "removed": f"Plan removed: {ps.plan} (was {fmt_inr(ps.old_price)}{per})"}.get(
                    ps.direction, f"{ps.plan}: {fmt_inr(ps.old_price)} → {fmt_inr(ps.new_price)}{per}")
                changes.append(Change(date=None, title=title,
                                      category="pricing", detail="Detected by diffing the pricing page",
                                      evidence=ev, source=s.origin))
        else:
            changes.extend(extract_changes(s.text, s.origin, since, as_of))
    feature_status = detect_features(corpus, cfg.features)
    fshifts = []
    for c in changes:
        if c.category in ("feature_added", "feature_removed", "improvement"):
            keys = ([k for k, spec in cfg.features.items() if mentions(c.title, spec.get("keywords", []))]
                    or [k for k, spec in cfg.features.items() if mentions(c.detail, spec.get("keywords", []))])
            change = {"feature_added": "added", "feature_removed": "removed", "improvement": "improved"}[c.category]
            if c.category == "improvement" and re.search(r"batch size", c.detail, re.I):
                change = "limited"
            fshifts.append(FeatureShift(feature=cfg.features[keys[0]]["label"] if keys else c.title,
                                        change=change, evidence=c.evidence))
    a = CompetitorAnalysis(competitor_id=comp.id, name=comp.name, mode="offline", plans=plans,
                           what_changed=changes, pricing_shifts=shifts, feature_shifts=fshifts,
                           feature_status=feature_status, weak_points=find_weak_points(snaps))
    return finish_analysis(a, cfg)


def finish_analysis(a: CompetitorAnalysis, cfg: Config) -> CompetitorAnalysis:
    """Fill in sales plays and narrative that the analyzer did not provide."""
    pb = Playbook(cfg)
    gaps = pb.gaps(a)
    gap_word = {"not_found": "No public mention of", "partial": "Only partial", "no": "Does not offer"}
    a.weak_points = ([WeakPoint(point=f"{gap_word[a.feature_status[k].status]}: {cfg.features[k]['label']}",
                                evidence=f"Status: {STATUS_ICON[a.feature_status[k].status]}"
                                         + (f" ({a.feature_status[k].evidence})" if a.feature_status[k].evidence else ""),
                                source="feature matrix")
                      for k in gaps] + a.weak_points)
    if not a.strengths:
        a.strengths = [cfg.features[k]["label"] for k, fs in a.feature_status.items() if fs.status == "yes"]
    if not a.positioning:
        a.positioning = pb.positioning(a)
    if not a.landmine_questions:
        a.landmine_questions = pb.landmines(a)
    if not a.objections:
        a.objections = pb.objections(a)
    elif not any("cheap" in o.objection.lower() for o in a.objections):
        a.objections.insert(0, pb._price_objection(a))
    a.threat = pb.threat(a)
    if not a.summary:
        up = [s for s in a.pricing_shifts if s.direction == "up"]
        down = [s for s in a.pricing_shifts if s.direction == "down"]
        bits = [a.positioning]
        if down:
            bits.append(f"Cut prices on {len(down)} plan(s), e.g. {down[0].plan} "
                        f"{fmt_inr(down[0].old_price)} → {fmt_inr(down[0].new_price)}.")
        if up:
            bits.append(f"Raised prices on {len(up)} plan(s), e.g. {up[0].plan} "
                        f"{fmt_inr(up[0].old_price)} → {fmt_inr(up[0].new_price)}.")
        added = [f.feature for f in a.feature_shifts if f.change == "added"]
        if added:
            bits.append(f"Recently launched: {', '.join(dict.fromkeys(added))}.")
        a.summary = " ".join(bits)
    return a


# ════════════════════════════════════════════════════════════════════════════
# 2b. ANALYZE (Gemini, structured output + fact-check guardrail)
# ════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are a B2B product-marketing analyst building a sales battlecard.
You receive our product profile, a capability taxonomy and raw text from a competitor's public pages.
Rules:
- Use ONLY the competitor text as evidence. Every 'evidence' field must be a verbatim sentence copied from it.
- Prices must be numbers that appear in the text. Do not guess missing prices; use null.
- feature_status: one entry per taxonomy key. status is yes | partial (beta, add-on, limited classes,
  coming soon) | no (explicitly not offered) | not_found (not mentioned).
- what_changed: only dated entries between {since} and {as_of}; date as YYYY-MM-DD.
- Landmine questions are neutral questions a rep asks the BUYER that expose a competitor gap without naming it.
- objections must include how to respond to "{name} is cheaper", using real numbers from both price lists.
Return JSON matching the schema."""


def gemini_client():
    from google import genai
    from google.genai import types
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                        http_options=types.HttpOptions(timeout=90_000))


def _norm_for_match(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w₹.%-]+", " ", s.lower())).strip()


def evidence_ok(evidence: str, corpus: str) -> bool:
    """True if the evidence is (nearly) verbatim in the source text."""
    if not evidence.strip():
        return False
    ev, cp = _norm_for_match(evidence), _norm_for_match(corpus)
    if ev in cp:
        return True
    best = max((difflib.SequenceMatcher(None, ev, _norm_for_match(s)).ratio() for s in sentences(corpus)),
               default=0)
    return best >= 0.85


def price_in_text(x: Optional[float], text: str) -> bool:
    if x is None:
        return True
    nums = {parse_price(m) for m in PRICE_RE.findall(text)}
    return x in nums


def analyze_with_gemini(comp: Competitor, snaps: list[Snapshot], cfg: Config, as_of: dt.date,
                        lookback_days: int, client=None) -> CompetitorAnalysis:
    from google.genai import types

    client = client or gemini_client()
    model = os.getenv("GEMINI_MODEL") or cfg.settings.get("gemini_model", "gemini-flash-latest")
    since = as_of - dt.timedelta(days=lookback_days)
    max_chars = int(cfg.settings.get("max_chars_per_source", 12000))
    blocks = []
    for s in snaps:
        blocks.append(f"### SOURCE kind={s.kind} origin={s.origin}\n{s.text[:max_chars]}")
        if s.baseline_text:
            blocks.append(f"### PREVIOUS SNAPSHOT of {s.origin}\n{s.baseline_text[:max_chars]}")
    taxonomy = {k: v["label"] for k, v in cfg.features.items()}
    us = {k: cfg.us[k] for k in ("name", "segment", "plans", "features", "differentiators") if k in cfg.us}
    prompt = (f"OUR PRODUCT:\n{json.dumps(us, ensure_ascii=False)}\n\nTAXONOMY:\n{json.dumps(taxonomy)}\n\n"
              f"COMPETITOR: {comp.name}\n\n" + "\n\n".join(blocks))
    resp = client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT.format(since=since, as_of=as_of, name=comp.name),
            response_mime_type="application/json", response_schema=LLMAnalysis, temperature=0.2))
    llm = LLMAnalysis.model_validate_json(resp.text)

    # ── guardrail: keep only claims grounded in the source text ──
    corpus = "\n".join(s.text for s in snaps)
    full = corpus + "\n" + "\n".join(s.baseline_text or "" for s in snaps)
    dropped = 0

    def keep(items, ok):
        nonlocal dropped
        kept = [i for i in items if ok(i)]
        dropped += len(items) - len(kept)
        return kept

    changes = keep(llm.what_changed, lambda c: evidence_ok(c.evidence, corpus))
    shifts = keep(llm.pricing_shifts, lambda p: price_in_text(p.new_price, corpus) and price_in_text(p.old_price, full))
    fshifts = keep(llm.feature_shifts, lambda f: evidence_ok(f.evidence, corpus))
    weak = keep(llm.weak_points, lambda w: evidence_ok(w.evidence, corpus))
    offline_status = detect_features(corpus, cfg.features)
    status = dict(offline_status)
    for fs in llm.feature_status:
        if fs.feature_key in cfg.features:
            if fs.status in ("yes", "partial", "no") and not evidence_ok(fs.evidence, corpus):
                dropped += 1
                continue
            status[fs.feature_key] = fs
    plans = [p for s in snaps if s.kind == "pricing" for p in extract_plans(s.text)]  # prices stay deterministic
    if dropped:
        log.warning("%s: guardrail dropped %d ungrounded Gemini claim(s)", comp.id, dropped)
    a = CompetitorAnalysis(competitor_id=comp.id, name=comp.name, mode="gemini", summary=llm.summary,
                           positioning=llm.positioning, what_changed=changes, pricing_shifts=shifts,
                           feature_shifts=fshifts, feature_status=status, weak_points=weak,
                           strengths=llm.strengths, plans=plans, landmine_questions=llm.landmine_questions[:6],
                           objections=llm.objections[:4], dropped_claims=dropped)
    return finish_analysis(a, cfg)


# ════════════════════════════════════════════════════════════════════════════
# 3. SYNTHESIZE: markdown battlecard
# ════════════════════════════════════════════════════════════════════════════
CATEGORY_ICON = {"pricing": "💰 Pricing", "feature_added": "🚀 Launch", "feature_removed": "🗑️ Removed",
                 "improvement": "🔧 Change", "reliability": "⚠️ Issue", "other": "📝 Other"}
THREAT_ICON = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low"}


def _cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def _price_cell(p: Plan) -> str:
    if p.price == 0:
        return "Free"
    return f"{fmt_inr(p.price)}" + (f"/{p.period}" if p.period and p.period != "free" else "")


def biggest_move(a: CompetitorAnalysis) -> str:
    for cat in ("pricing", "feature_added", "feature_removed", "reliability", "improvement"):
        for c in a.what_changed:
            if c.category == cat:
                return c.title + (f" ({c.date})" if c.date else "")
    return "No dated changes in the window"


def render_battlecard(analyses: list[CompetitorAnalysis], snaps: dict[str, list[Snapshot]], cfg: Config,
                      generated_at: str, as_of: dt.date, lookback_days: int) -> str:
    pb = Playbook(cfg)
    us = cfg.us["name"]
    since = as_of - dt.timedelta(days=lookback_days)
    modes = sorted({a.mode for a in analyses})
    mode_txt = " + ".join("Gemini (structured JSON, fact-checked)" if m == "gemini"
                          else "offline heuristic analyzer (no LLM)" for m in modes)
    any_mock = any(c.mock for c in cfg.competitors if c.id in {a.competitor_id for a in analyses})
    L: list[str] = []
    L.append(f"# ⚔️ Sales Battlecard: {us} vs {', '.join(a.name for a in analyses)}")
    L.append("")
    if any_mock:
        L.append("> ⚠️ **MOCK DATA.** The competitors in this battlecard are **fictional companies** with invented "
                 f"prices and changelogs, created for a portfolio demo. The {us} profile is illustrative.")
        L.append("")
    L.append(f"**Generated:** {generated_at} · **Analysis mode:** {mode_txt} · "
             f"**Change window:** {since:%d %b %Y} to {as_of:%d %b %Y} · **Agent:** v{__version__}")
    L.append("")
    L.append("## 🧭 Executive TL;DR")
    L.append("")
    L.append("| Competitor | Threat | Positioning | Biggest recent move | How we win |")
    L.append("|---|---|---|---|---|")
    for a in analyses:
        wt = pb.win_themes(a)
        L.append(f"| **{a.name}** | {THREAT_ICON[a.threat]} | {_cell(a.positioning)} | {_cell(biggest_move(a))} | "
                 f"{_cell(wt[0]['theme']) if wt else '-'} |")
    L.append("")
    for i, a in enumerate(analyses, 1):
        L += render_competitor(i, a, pb, cfg)
    L.append("## 📚 Sources")
    L.append("")
    L.append("| Competitor | Source | Type | Retrieved | SHA-256 | Since last run |")
    L.append("|---|---|---|---|---|---|")
    for a in analyses:
        for s in snaps.get(a.competitor_id, []):
            label = {"new": "🆕 first run", "changed": "🔄 changed", "unchanged": "✔️ unchanged"}[s.status]
            L.append(f"| {a.name} | `{s.origin}` | {s.kind} | {s.retrieved_at} | `{s.sha256[:12]}` | {label} |")
    L.append("")
    L.append("---")
    L.append(f"*Generated by [ai-gtm-competitive-agent](https://github.com/Creator-hub24/ai-gtm-competitive-agent) "
             f"({', '.join(modes)} mode). Verify claims against the linked sources before "
             "sharing with customers.*")
    return "\n".join(L) + "\n"


def render_competitor(i: int, a: CompetitorAnalysis, pb: Playbook, cfg: Config) -> list[str]:
    us = cfg.us["name"]
    L = [f"## {i}. {a.name}", ""]
    L.append(f"**Threat:** {THREAT_ICON[a.threat]} · **Analyzed with:** "
             f"{'Gemini' if a.mode == 'gemini' else 'offline heuristics'}"
             + (f" · **Guardrail:** {a.dropped_claims} ungrounded claim(s) dropped" if a.dropped_claims else ""))
    L.append("")
    L.append(f"> {a.summary}")
    L.append("")
    # What changed
    L.append("### 🔔 What changed")
    L.append("")
    if a.what_changed:
        L.append("| Date | Type | Change | Evidence |")
        L.append("|---|---|---|---|")
        for c in a.what_changed:
            L.append(f"| {c.date or 'page diff'} | {CATEGORY_ICON.get(c.category, c.category)} | {_cell(c.title)} | "
                     f"{_cell(c.evidence)} |")
    else:
        L.append("_No dated changes found in the window._")
    L.append("")
    # Pricing
    L.append("### 💰 Pricing comparison")
    L.append("")
    L.append("| Plan | List price | Effective ₹/month | Change vs last snapshot |")
    L.append("|---|---|---|---|")
    shift_by = {re.sub(r"\W+", " ", s.plan.lower()).strip(): s for s in a.pricing_shifts}
    for p in a.plans:
        s = shift_by.get(re.sub(r"\W+", " ", p.name.lower()).strip())
        chg = "-"
        if s and s.direction in ("up", "down"):
            arrow = "🔺" if s.direction == "up" else "🔻"
            chg = f"{arrow} {fmt_inr(s.old_price)} → {fmt_inr(s.new_price)} ({s.pct:+.0f}%)"
        elif s and s.direction == "new":
            chg = "🆕 new plan"
        name = p.name + (" *(add-on)*" if p.is_addon and "add-on" not in p.name.lower() else "")
        L.append(f"| {a.name}: {_cell(name)} | {_price_cell(p)} | "
                 f"{fmt_inr(p.monthly_equiv) if p.monthly_equiv is not None else 'n/a'} | {chg} |")
    for p in pb.us_plans:
        L.append(f"| **{us}: {p.name}** | {_price_cell(p)} | "
                 f"{fmt_inr(p.monthly_equiv) if p.monthly_equiv is not None else 'n/a'} | |")
    tf, of = plan_floor(a.plans), plan_floor(pb.us_plans)
    if tf and of:
        ratio = tf.monthly_equiv / of.monthly_equiv
        verdict = (f"{(1 - ratio) * 100:.0f}% cheaper" if ratio < 1 else
                   f"{ratio:.1f}x our price" if ratio >= 2 else f"{(ratio - 1) * 100:.0f}% more expensive")
        L.append("")
        L.append(f"**Lowest paid entry point:** {a.name} {fmt_inr(tf.monthly_equiv)}/month vs {us} "
                 f"{fmt_inr(of.monthly_equiv)}/month, so {a.name} is **{verdict}**.")
    L.append("")
    # Feature matrix
    L.append(f"### 🧩 Feature comparison matrix ({us} vs {a.name})")
    L.append("")
    L.append(f"| Capability | {us} | {a.name} | Evidence ({a.name}) |")
    L.append("|---|---|---|---|")
    for k, spec in cfg.features.items():
        fs = a.feature_status.get(k) or FeatureStatus(feature_key=k, status="not_found")
        L.append(f"| {spec['label']} | {STATUS_ICON[pb.us_feat.get(k, 'no')]} | {STATUS_ICON[fs.status]} | "
                 f"{_cell(fs.evidence) or '-'} |")
    L.append("")
    if a.feature_shifts:
        L.append("**Feature shifts:** " + "; ".join(f"{f.change} *{f.feature}*" for f in a.feature_shifts) + ".")
        L.append("")
    # Weak points
    L.append("### 🎯 Weak points to exploit")
    L.append("")
    for w in a.weak_points[:7]:
        L.append(f"- **{w.point}.** {w.evidence}" + (f" *(source: `{w.source}`)*" if w.source and
                                                      w.source != 'feature matrix' else ""))
    L.append("")
    # Landmines
    L.append("### 💣 Landmine questions (ask the buyer)")
    L.append("")
    for n, q in enumerate(a.landmine_questions, 1):
        L.append(f"{n}. {q}")
    L.append("")
    # Objections
    L.append("### 🛡️ Objection handling")
    L.append("")
    for o in a.objections:
        L.append(f"**{o.objection}**")
        L.append(f"> {o.response}")
        L.append("")
    # Win themes
    L.append("### 🏆 Win themes")
    L.append("")
    for t in pb.win_themes(a):
        L.append(f"- **{t['theme']}.** {t['proof']}")
    L.append("")
    return L


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Monitor competitor pages, analyze changes (Gemini or offline) and write a Sales Battlecard.")
    p.add_argument("--config", default=str(ROOT / "competitors.yaml"), help="config YAML (default: %(default)s)")
    p.add_argument("--competitor", action="append", metavar="ID",
                   help="only analyze this competitor id (repeatable)")
    p.add_argument("--out", default=str(ROOT / "output" / "battlecard.md"), help="battlecard path")
    p.add_argument("--json", dest="json_out", metavar="PATH", help="also write the analyses as JSON")
    p.add_argument("--offline", action="store_true",
                   help="never call Gemini or the network: local files + heuristic analyzer")
    p.add_argument("--live", action="store_true", help="fetch competitor URLs instead of local files")
    p.add_argument("--as-of", help="reference date YYYY-MM-DD for the change window (default: config or today)")
    p.add_argument("--lookback-days", type=int, help="change window in days (default: config or 120)")
    p.add_argument("--no-cache", action="store_true", help="do not read or write the change-detection cache")
    p.add_argument("--changed-only", action="store_true",
                   help="exit without writing if no source changed since the last run")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def run(args: argparse.Namespace, client=None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    cfg = load_config(Path(args.config))
    st = cfg.settings
    tz = ZoneInfo(st.get("timezone", "Asia/Kolkata"))
    now = dt.datetime.now(tz)
    as_of = dt.date.fromisoformat(args.as_of or str(st.get("as_of") or now.date()))
    lookback = args.lookback_days or int(st.get("lookback_days", 120))
    if args.offline and args.live:
        log.error("--offline and --live are mutually exclusive")
        return 2
    comps = cfg.competitors
    if args.competitor:
        wanted = set(args.competitor)
        unknown = wanted - {c.id for c in comps}
        if unknown:
            log.error("unknown competitor id(s): %s (known: %s)", ", ".join(sorted(unknown)),
                      ", ".join(c.id for c in comps))
            return 2
        comps = [c for c in comps if c.id in wanted]
    use_gemini = not args.offline and (client is not None or bool(os.getenv("GEMINI_API_KEY")))
    log.info("mode: %s | sources: %s | as-of %s | window %d days",
             "gemini" if use_gemini else "offline", "live URLs" if args.live else "local files", as_of, lookback)

    cache = Cache(cfg.path(st.get("cache_file", ".cache/agent_state.json")), enabled=not args.no_cache)
    stamp = now.strftime("%Y-%m-%d %H:%M %Z")
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    analyses, all_snaps = [], {}
    for comp in comps:
        snaps = monitor(comp, cfg, cache, live=args.live, now=stamp, session=session)
        if not snaps:
            log.warning("%s: no sources loaded, skipped", comp.id)
            continue
        all_snaps[comp.id] = snaps
        a = None
        if use_gemini:
            try:
                a = analyze_with_gemini(comp, snaps, cfg, as_of, lookback, client=client)
            except (Exception, ValidationError) as e:  # never lose the battlecard because the LLM failed
                log.warning("%s: Gemini analysis failed (%s); falling back to offline analyzer", comp.id, e)
        a = a or analyze_offline(comp, snaps, cfg, as_of, lookback)
        analyses.append(a)
        log.info("analyze %-10s mode=%s threat=%s changes=%d weak=%d", comp.id, a.mode, a.threat,
                 len(a.what_changed), len(a.weak_points))
    if not analyses:
        log.error("nothing to analyze")
        return 1
    if args.changed_only and all(s.status == "unchanged" for ss in all_snaps.values() for s in ss):
        log.info("no source changed since the last run; battlecard not rewritten")
        return 0
    md = render_battlecard(analyses, all_snaps, cfg, stamp, as_of, lookback)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    log.info("battlecard written: %s (%d lines)", out, md.count("\n"))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps([a.model_dump() for a in analyses], indent=2,
                                                  ensure_ascii=False), encoding="utf-8")
        log.info("analysis JSON written: %s", args.json_out)
    for a in analyses:
        update_cache(cache, all_snaps[a.competitor_id], a)
    cache.save()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    try:
        return run(args)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        log.error("%s", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
