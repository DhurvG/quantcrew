"""Data-cleaning logic, tested without network access."""
from quantcrew.tools.market_data import normalise_fundamentals, parse_google_news_rss


def test_debt_to_equity_percentage_is_converted_to_ratio():
    out = normalise_fundamentals({"debtToEquity": 9.541, "trailingPE": 13.8}, "USD", {})
    assert out["debtToEquity"] == 0.095


def test_statement_currency_and_units_note_are_attached():
    out = normalise_fundamentals({"trailingPE": 13.8}, "USD", {"Total Revenue": {}})
    assert out["statements_currency"] == "USD"
    assert "never divide" in out["units"]
    assert normalise_fundamentals({}, None, {})["statements_currency"] == "unknown"


def test_google_news_rss_parsing():
    xml = b"""<rss><channel>
      <item><title>TCS wins deal</title><source>Mint</source><pubDate>Mon, 05 Oct 2026</pubDate></item>
      <item><title>TCS Q2 preview</title><source>ET</source><pubDate>Sun, 04 Oct 2026</pubDate></item>
      <item><title>Third</title></item></channel></rss>"""
    out = parse_google_news_rss(xml, limit=2)
    assert [h["title"] for h in out] == ["TCS wins deal", "TCS Q2 preview"]
    assert out[0]["publisher"] == "Mint" and out[0]["feed"] == "google_news"
