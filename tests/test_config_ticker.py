from quantcrew.config import NIFTY50_SYMBOLS, normalise_ticker


def test_ticker_gets_nse_suffix_and_uppercase():
    assert normalise_ticker("infy") == "INFY.NS"
    assert normalise_ticker("  tcs ") == "TCS.NS"


def test_ticker_with_suffix_is_unchanged():
    assert normalise_ticker("INFY.NS") == "INFY.NS"


def test_nifty50_list_has_fifty_unique_symbols():
    assert len(NIFTY50_SYMBOLS) == 50
    assert len(set(NIFTY50_SYMBOLS)) == 50