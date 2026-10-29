"""Tests for the competitive-intelligence agent. Offline only: no network, no API keys."""
from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

import pytest

import agent

ROOT = Path(agent.__file__).resolve().parent
SAMPLES = ROOT / "sample_competitors"
AS_OF = dt.date(2026, 10, 29)


@pytest.fixture(scope="module")
def cfg():
    return agent.load_config(ROOT / "competitors.yaml")


def text_of(rel: str) -> str:
    p = SAMPLES / rel
    return agent.to_text(p.read_text(encoding="utf-8"), p.name)


# ── Monitor ─────────────────────────────────────────────────────────────────
def test_html_to_text_keeps_structure_and_drops_noise():
    html = ("<html><head><title>T</title><script>x=1</script></head><body><nav>Home</nav>"
            "<h2>Plan A</h2><p>&#8377;99/month</p><ul><li>One</li><li>Two</li></ul><footer>f</footer></body></html>")
    t = agent.html_to_text(html)
    assert "## Plan A" in t and "₹99/month" in t and "- One" in t and "- Two" in t
    assert "x=1" not in t and "Home" not in t and "f\n" not in t


def test_content_hash_is_stable():
    assert agent.content_hash("abc") == agent.content_hash("abc") != agent.content_hash("abd")


def test_change_detection_new_then_unchanged_then_changed(tmp_path):
    work = tmp_path / "repo"
    shutil.copytree(SAMPLES, work / "sample_competitors")
    shutil.copy(ROOT / "competitors.yaml", work / "competitors.yaml")
    cfg = agent.load_config(work / "competitors.yaml")
    comp = next(c for c in cfg.competitors if c.id == "sumzilla")
    cache = agent.Cache(tmp_path / "state.json")

    def run_once():
        snaps = agent.monitor(comp, cfg, cache, live=False, now="t")
        a = agent.analyze_offline(comp, snaps, cfg, AS_OF, 120)
        agent.update_cache(cache, snaps, a)
        cache.save()
        return {s.kind: s.status for s in snaps}

    assert run_once() == {"pricing": "new", "changelog": "new"}
    assert run_once() == {"pricing": "unchanged", "changelog": "unchanged"}
    page = work / "sample_competitors/sumzilla/pricing.html"
    page.write_text(page.read_text(encoding="utf-8").replace("&#8377;249/month", "&#8377;229/month"),
                    encoding="utf-8")
    assert run_once() == {"pricing": "changed", "changelog": "unchanged"}
    saved = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert any(p["price"] == 229 for v in saved.values() if v["plans"] for p in v["plans"])


def test_fetch_url_uses_firecrawl_when_key_set(monkeypatch):
    calls = {}

    class Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"success": True, "data": {"markdown": "# Pricing\n\n## Pro\n₹199/month"}}

    class Session:
        headers: dict = {}

        def post(self, url, **kw):
            calls["url"], calls["json"] = url, kw["json"]
            return Resp()

    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
    text = agent.fetch_url("https://rival.example/pricing", session=Session())
    assert calls["url"] == agent.FIRECRAWL_URL and calls["json"]["formats"] == ["markdown"]
    assert agent.extract_plans(text)[0].price == 199


# ── Offline analysis ────────────────────────────────────────────────────────
def test_extract_plans_prices_periods_and_addons():
    plans = {p.name: p for p in agent.extract_plans(text_of("sumzilla/pricing.html"))}
    assert plans["Free"].price == 0
    assert (plans["Sumzilla Plus"].price, plans["Sumzilla Plus"].period) == (249, "month")
    assert plans["Sumzilla Plus Annual"].monthly_equiv == pytest.approx(166.58, abs=0.01)
    assert plans["AI Doubt Bot add-on (beta)"].is_addon
    assert "Frequently asked questions" not in plans


@pytest.mark.parametrize("price,period,expected", [
    (0, None, 0.0), (499, "month", 499), (1299, "quarter", 433), (3999, "year", 333.25), (100, None, None)])
def test_monthly_equivalent(price, period, expected):
    assert agent.monthly_equivalent(price, period) == expected


@pytest.mark.parametrize("s,expected", [
    ("2026-10-05 · Plus", dt.date(2026, 10, 5)), ("1 Oct 2026: New pricing", dt.date(2026, 10, 1)),
    ("October 20, 2026 — Adaptive", dt.date(2026, 10, 20)), ("no date here", None)])
def test_parse_date(s, expected):
    assert agent.parse_date(s) == expected


def test_changelog_window_and_classification():
    since = AS_OF - dt.timedelta(days=120)
    changes = agent.extract_changes(text_of("sumzilla/changelog.md"), "x", since, AS_OF)
    dates = [c.date for c in changes]
    assert "2026-05-02" not in dates and dates == sorted(dates, reverse=True)
    by_date = {c.date: c.category for c in changes}
    assert by_date["2026-10-05"] == "pricing"
    assert by_date["2026-09-14"] == "feature_added"
    assert by_date["2026-08-20"] == "feature_removed"
    assert by_date["2026-07-10"] == "reliability"


def test_feature_detection_statuses(cfg):
    nest = agent.detect_features(text_of("numeronest/pricing.html"), cfg.features)
    assert nest["adaptive_practice"].status == "yes"
    assert nest["board_pattern_mocks"].status == "no"          # "not available yet"
    assert nest["full_coverage_6_10"].status == "partial"      # "Class 6-8 only"
    assert nest["live_classes"].status == "not_found"
    zilla = agent.detect_features(text_of("sumzilla/pricing.html"), cfg.features)
    assert zilla["parent_reports"].status == "not_found"       # FAQ question is not a claim
    assert zilla["doubt_solving"].status == "partial"          # beta add-on
    assert zilla["gamification"].status == "yes"


def test_diff_plans_detects_up_down_new_removed():
    old = [agent.Plan(name="Pro", price=499, period="month"), agent.Plan(name="Gone", price=1, period="month")]
    new = [agent.Plan(name="Pro", price=599, period="month"), agent.Plan(name="Fresh", price=9, period="month")]
    shifts = {s.plan: s for s in agent.diff_plans(old, new, "ev")}
    assert shifts["Pro"].direction == "up" and shifts["Pro"].pct == pytest.approx(20.04, abs=0.01)
    assert shifts["Fresh"].direction == "new" and shifts["Gone"].direction == "removed"


def test_price_is_not_mistaken_for_usage_cap():
    snap = agent.Snapshot("c", "pricing", "o", "## Plan\n₹1,999 per month\n- 5 doubts per day\n", "t", "h")
    points = [w.evidence for w in agent.find_weak_points([snap]) if w.point.startswith("Usage cap")]
    assert points == ["5 doubts per day"]


def test_fmt_inr_uses_indian_grouping():
    assert agent.fmt_inr(123456) == "₹1,23,456"
    assert agent.fmt_inr(17999) == "₹17,999"
    assert agent.fmt_inr(None) == "n/a"


def test_yaml_boolean_statuses_are_normalised():
    assert agent._norm_status(True) == "yes" and agent._norm_status(False) == "no"
    assert agent._norm_status("Partial") == "partial"


def test_offline_analysis_sales_plays(cfg):
    comp = next(c for c in cfg.competitors if c.id == "sumzilla")
    snaps = agent.monitor(comp, cfg, agent.Cache(Path("/nonexistent"), enabled=False), live=False, now="t")
    a = agent.analyze_offline(comp, snaps, cfg, AS_OF, 120)
    assert a.mode == "offline"
    assert {(s.plan, s.old_price, s.new_price) for s in a.pricing_shifts} >= {("Sumzilla Plus", 349, 249)}
    cheaper = a.objections[0]
    assert "cheaper" in cheaper.objection and "₹348/month" in cheaper.response  # plan + add-on, like for like
    assert a.landmine_questions and a.threat in ("High", "Medium", "Low")


# ── Gemini mode (fake client, no network) ───────────────────────────────────
class FakeModels:
    def __init__(self, payload=None, error=None):
        self.payload, self.error, self.calls = payload, error, []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.error:
            raise self.error
        return type("R", (), {"text": json.dumps(self.payload)})()


class FakeClient:
    def __init__(self, **kw):
        self.models = FakeModels(**kw)


GEMINI_PAYLOAD = {
    "summary": "Budget gamified app that just cut prices.",
    "positioning": "Low-price gamified practice for Class 5-10.",
    "what_changed": [
        {"date": "2026-10-05", "title": "Price cut", "category": "pricing", "detail": "",
         "evidence": "We cut the price of Sumzilla Plus from ₹349 to ₹249 per month"},
        {"date": "2026-10-10", "title": "Invented launch", "category": "feature_added", "detail": "",
         "evidence": "Sumzilla now offers live classes with IIT faculty."},  # not in the source
    ],
    "pricing_shifts": [
        {"plan": "Sumzilla Plus", "old_price": 349, "new_price": 249, "period": "month", "evidence": "x"},
        {"plan": "Plus Annual", "old_price": 2499, "new_price": 1499, "period": "year", "evidence": "x"},  # wrong
    ],
    "feature_shifts": [],
    "feature_status": [
        {"feature_key": "gamification", "status": "yes", "evidence": "Earn XP, keep your streak, climb the leaderboard."},
        {"feature_key": "made_up_key", "status": "yes", "evidence": "whatever"},
    ],
    "weak_points": [{"point": "Promo price", "evidence": "Festive offer, prices may be revised after 31 December 2026."}],
    "strengths": ["Low price"],
    "landmine_questions": ["Is the quoted price locked in at renewal?"],
    "objections": [{"objection": "Sumzilla is cheaper", "response": "Compare like for like."}],
}


def test_gemini_analysis_is_fact_checked(cfg):
    comp = next(c for c in cfg.competitors if c.id == "sumzilla")
    snaps = agent.monitor(comp, cfg, agent.Cache(Path("/x"), enabled=False), live=False, now="t")
    client = FakeClient(payload=GEMINI_PAYLOAD)
    a = agent.analyze_with_gemini(comp, snaps, cfg, AS_OF, 120, client=client)
    assert a.mode == "gemini"
    assert [c.title for c in a.what_changed] == ["Price cut"]                 # invented launch dropped
    assert [s.new_price for s in a.pricing_shifts] == [249]                   # ₹1,499 is not in the text
    assert a.dropped_claims == 2
    assert "made_up_key" not in a.feature_status and len(a.feature_status) == len(cfg.features)
    assert a.plans and a.plans[1].price == 249                               # prices stay deterministic
    cfg_sent = client.models.calls[0]["config"]
    assert cfg_sent.response_mime_type == "application/json" and cfg_sent.response_schema is agent.LLMAnalysis


def test_run_falls_back_to_offline_when_gemini_fails(tmp_path):
    out = tmp_path / "bc.md"
    args = agent.build_parser().parse_args(["--competitor", "numeronest", "--out", str(out), "--no-cache"])
    assert agent.run(args, client=FakeClient(error=RuntimeError("503 UNAVAILABLE"))) == 0
    assert "offline heuristic analyzer" in out.read_text(encoding="utf-8")


def test_run_uses_gemini_when_client_available(tmp_path):
    out = tmp_path / "bc.md"
    args = agent.build_parser().parse_args(["--competitor", "sumzilla", "--out", str(out), "--no-cache"])
    assert agent.run(args, client=FakeClient(payload=GEMINI_PAYLOAD)) == 0
    md = out.read_text(encoding="utf-8")
    assert "Gemini (structured JSON, fact-checked)" in md and "2 ungrounded claim(s) dropped" in md


# ── CLI / end to end ────────────────────────────────────────────────────────
def test_end_to_end_offline_battlecard(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "should-be-ignored-with-offline")
    out, js = tmp_path / "battlecard.md", tmp_path / "a.json"
    assert agent.main(["--offline", "--no-cache", "--out", str(out), "--json", str(js)]) == 0
    md = out.read_text(encoding="utf-8")
    for heading in ["Executive TL;DR", "What changed", "Feature comparison matrix", "Pricing comparison",
                    "Landmine questions", "Objection handling", "Win themes", "Sources", "MOCK DATA"]:
        assert heading in md, heading
    for name in ["Sumzilla", "Ganitron Live", "NumeroNest AI"]:
        assert f'"{name} is cheaper."' in md
    assert "offline heuristic analyzer" in md and "Gemini" not in md.split("## 🧭")[0].replace("no LLM", "")
    data = json.loads(js.read_text(encoding="utf-8"))
    assert {d["competitor_id"] for d in data} == {"sumzilla", "ganitron", "numeronest"}


def test_cli_rejects_unknown_competitor(tmp_path):
    assert agent.main(["--offline", "--competitor", "nope", "--out", str(tmp_path / "x.md")]) == 2


def test_cli_rejects_offline_with_live(tmp_path):
    assert agent.main(["--offline", "--live", "--out", str(tmp_path / "x.md"), "--no-cache"]) == 2


def test_committed_sample_battlecard_matches_agent_output(tmp_path):
    """The committed sample must be real agent output (only the timestamps may differ)."""
    out = tmp_path / "bc.md"
    assert agent.main(["--offline", "--no-cache", "--out", str(out)]) == 0
    import re
    strip = lambda s: re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} \w+", "<ts>", s)  # noqa: E731
    assert strip(out.read_text(encoding="utf-8")) == strip(
        (ROOT / "output" / "sample_battlecard.md").read_text(encoding="utf-8"))
