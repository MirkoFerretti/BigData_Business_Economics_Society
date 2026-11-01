# Exact source-data snapshot

Source: Yahoo Finance via yfinance. The requested sample is 2015-01-01 through
2025-12-31 (the download end parameter is 2026-01-01, exclusive). The first
available observation in every CSV is 2015-01-02.

These twelve ticker CSVs preserve the exact local data snapshot used for the
submitted analysis. BRK.B uses Yahoo ticker BRK-B. Source prices, adjusted close,
volume and corporate-action fields are preserved as supplied; historical vendor
fields are not independently reconstructed unadjusted tape data.

SHA256SUMS.txt records each CSV's SHA-256 checksum, with filenames relative to
this directory. Use this snapshot for exact-data reproduction and follow the
repository-root README pipeline. A fresh Yahoo download may differ because
providers can revise historical data. These files were not redownloaded during
release cleanup.

The .yfinance_cache directory contains local cache/cookie/certificate support
files and is intentionally excluded from publication and the checksum list.
