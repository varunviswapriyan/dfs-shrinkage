"""Re-download the price and volume data used in the paper from Yahoo Finance.

The paper's results were computed from the cached files in data/ (downloaded
with yfinance, auto_adjust=True, 1 Dec 2004 - 31 Dec 2024). With auto_adjust=True
the "Close" column is Yahoo's adjusted close, which is adjusted for stock splits
AND dividend distributions, so returns computed from it include dividends
(total returns), not price-only returns. Yahoo revises adjusted prices
retroactively, so a fresh download can differ slightly from the cache; use the
cached files to reproduce the reported numbers exactly.

  python fetch_data.py --out ../data_redownload
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    import pandas as pd
    import yfinance as yf
    from _suite import CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data_redownload"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for tag, tick in [("stocks", CFG["stocks"]), ("etfs", CFG["etfs"])]:
        raw = yf.download(tick, start=CFG["start"], end=CFG["end"], auto_adjust=True,
                          progress=False, threads=False)
        px = raw["Close"].sort_index(axis=1)
        vol = raw["Volume"].sort_index(axis=1)
        px.to_csv(os.path.join(a.out, f"prices_{tag}.csv"))
        vol.to_csv(os.path.join(a.out, f"volume_{tag}.csv"))
        print(tag, px.shape)


if __name__ == "__main__":
    main()
