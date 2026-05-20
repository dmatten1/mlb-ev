"""Read odds snapshots from S3 (or local) and normalize to per-game lines.

Each snapshot JSON from The Odds API contains:
  data[*]: one game
    home_team, away_team, commence_time (UTC)
    bookmakers[*]: one book
      markets[*]: h2h, totals, etc.
        outcomes[*]: home/away with American price

This module flattens that into long-form DataFrames at two levels:

* :func:`load_snapshots_long` — one row per (snapshot_ts, game, book).
  Useful for diagnostics, line-movement analysis, etc.
* :func:`best_lines_per_game` — collapse to one row per game with the
  best-available home & away prices across all books in the latest
  snapshot before commence_time. This is what the backtest / inference
  layer consumes.

Team-name resolution: The Odds API uses full English names matching
exactly what MLB-StatsAPI emits ("New York Yankees", "Athletics", etc.),
so we use the per-season team-id lookup built from
``data/features/training_<year>.parquet`` rather than maintaining a
parallel mapping.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_S3_BUCKET = "mlb-ev-dcm92"


def resolve_odds_s3_bucket(explicit: str | None = None) -> str:
    """Bucket for odds snapshots (env ``ODDS_S3_BUCKET`` / ``MLB_EV_S3_BUCKET``)."""
    if explicit:
        return explicit
    return (
        os.getenv("ODDS_S3_BUCKET")
        or os.getenv("MLB_EV_S3_BUCKET")
        or DEFAULT_S3_BUCKET
    )
DEFAULT_S3_PREFIX = "raw/odds/baseball_mlb/h2h"
DEFAULT_LOCAL_ROOT = Path("data/raw/odds/baseball_mlb/h2h")

logger = logging.getLogger("inference.odds_loader")


# ---------------------------------------------------------------------------
# Snapshot enumeration / fetch
# ---------------------------------------------------------------------------

def list_snapshot_keys_s3(
    *,
    bucket: str = DEFAULT_S3_BUCKET,
    prefix: str = DEFAULT_S3_PREFIX,
    date_lo: str | None = None,
    date_hi: str | None = None,
) -> list[str]:
    """Return S3 object keys for snapshots in ``[date_lo, date_hi]`` inclusive.

    Dates are strings in ``YYYY-MM-DD`` form. Bound is the partition
    name (UTC capture date), not the game date — but capture is daily
    and games happen close-by, so this is close enough for filtering.
    """
    import boto3

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []) or []:
            k = obj["Key"]
            partition = k[len(prefix) + 1 :].split("/", 1)[0]  # YYYY-MM-DD
            if date_lo is not None and partition < date_lo:
                continue
            if date_hi is not None and partition > date_hi:
                continue
            keys.append(k)
    return sorted(keys)


def read_snapshot_s3(bucket: str, key: str) -> dict:
    """Download one snapshot JSON from S3."""
    import boto3

    obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())


def read_snapshot_local(path: Path) -> dict:
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# Long-form parser
# ---------------------------------------------------------------------------

def _parse_snapshot(record: dict, *, source_id: str) -> pd.DataFrame:
    """One snapshot dict -> long DataFrame.

    Columns:
        snapshot_ts (UTC), source (S3 key or local path), commence_time (UTC),
        odds_api_game_id, home_team, away_team,
        book_key, book_title, last_update,
        home_price_american, away_price_american

    Filters out any market that isn't h2h. Drops books missing either
    side's price.
    """
    fetched_at = record.get("fetched_at_utc")
    snapshot_ts = pd.Timestamp(fetched_at)
    rows: list[dict] = []
    for game in record.get("data", []) or []:
        commence = pd.Timestamp(game.get("commence_time"))
        home, away = game.get("home_team"), game.get("away_team")
        gid = game.get("id")
        for book in game.get("bookmakers", []) or []:
            h2h = next(
                (m for m in (book.get("markets") or []) if m.get("key") == "h2h"),
                None,
            )
            if h2h is None:
                continue
            outcomes = {o.get("name"): o.get("price") for o in (h2h.get("outcomes") or [])}
            home_price = outcomes.get(home)
            away_price = outcomes.get(away)
            if home_price is None or away_price is None:
                continue
            rows.append({
                "snapshot_ts": snapshot_ts,
                "source": source_id,
                "commence_time": commence,
                "odds_api_game_id": gid,
                "home_team": home,
                "away_team": away,
                "book_key": book.get("key"),
                "book_title": book.get("title"),
                "last_update": pd.Timestamp(h2h.get("last_update")),
                "home_price_american": float(home_price),
                "away_price_american": float(away_price),
            })
    return pd.DataFrame(rows)


def load_snapshots_long(
    *,
    date_lo: str | None = None,
    date_hi: str | None = None,
    bucket: str | None = None,
    prefix: str = DEFAULT_S3_PREFIX,
    local_root: Path | str | None = None,
) -> pd.DataFrame:
    """Read all snapshots in the date range and return one long DataFrame.

    If ``local_root`` is provided, reads from disk instead of S3 (useful
    if you've cached snapshots locally). Otherwise reads from S3 using
    ``bucket`` + ``prefix``.

    Returns columns documented in :func:`_parse_snapshot`, sorted by
    ``snapshot_ts``.
    """
    pieces: list[pd.DataFrame] = []
    if local_root is not None:
        local_root = Path(local_root)
        paths = sorted(local_root.rglob("*.json"))
        for p in paths:
            partition = p.parent.name
            if date_lo is not None and partition < date_lo:
                continue
            if date_hi is not None and partition > date_hi:
                continue
            try:
                rec = read_snapshot_local(p)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed snapshot: %s", p)
                continue
            pieces.append(_parse_snapshot(rec, source_id=str(p)))
    else:
        bucket = resolve_odds_s3_bucket(bucket)
        keys = list_snapshot_keys_s3(
            bucket=bucket, prefix=prefix,
            date_lo=date_lo, date_hi=date_hi,
        )
        logger.info("Reading %d snapshots from s3://%s/%s", len(keys), bucket, prefix)
        for k in keys:
            rec = read_snapshot_s3(bucket, k)
            pieces.append(_parse_snapshot(rec, source_id=k))
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True).sort_values("snapshot_ts").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-game collapse: "what was the closing line?"
# ---------------------------------------------------------------------------

# Default to the user's actual book set (DK / FanDuel / BetMGM).
# Pass ``allowed_books=None`` to use every book in the snapshot, or pass
# a different list to override.
DEFAULT_ALLOWED_BOOKS: tuple[str, ...] = ("draftkings", "fanduel", "betmgm")


def best_lines_per_game(
    long_df: pd.DataFrame,
    *,
    close_window_minutes: int = 30,
    price_strategy: str = "best",
    drop_extreme_books: bool = True,
    extreme_price_threshold: int = 500,
    allowed_books: tuple[str, ...] | list[str] | None = DEFAULT_ALLOWED_BOOKS,
) -> pd.DataFrame:
    """Collapse the long frame to one row per game with pre-game lines.

    For each (commence_time, home_team, away_team):
      1. Find the latest snapshot ``<= commence_time - close_window_minutes``
         (gives us a "pre-game-time" line, not the very last second).
         **Games with no pre-cutoff snapshot are dropped** — falling back
         to post-commence snapshots would inject in-game odds (which our
         pre-game model can't price honestly).
      2. Optionally drop books with extreme prices from that snapshot
         (defaults to dropping any book where ``|price| > 500``). These
         are typically stale prices set when a book first opened a market
         and never updated, or live in-game prices stamped with a
         pre-game commence_time. Removing them protects line-shopping
         from picking a price nobody could actually have hit.
      3. From the cleaned snapshot, aggregate prices across books.

    ``price_strategy``:
      * ``"best"`` (default) — max per side (i.e. highest American odds,
        most generous to the bettor). With 5-9 books in S3 every
        snapshot this is the realistic line-shopping price. Also
        records which book offered each side's best price (``home_book``
        / ``away_book`` columns).
      * ``"median"`` — median across books. More conservative; useful
        for ablating "is the model edgy independent of line shopping?"
        in backtests.
      * ``"mean"`` — mean across books.

    ``allowed_books`` filters the long frame to a specific set of book
    keys before any aggregation. Defaults to the user's real account
    set (``DEFAULT_ALLOWED_BOOKS``) so the line-shopping result is
    achievable. Pass ``None`` to consider every book in the snapshot.

    Returns columns:
        commence_time, home_team, away_team,
        snapshot_ts, n_books,
        home_price_american, away_price_american,
        home_book, away_book   (best-price book per side; only populated
                                for price_strategy="best")
    """
    if long_df.empty:
        return long_df
    df = long_df.copy()
    if allowed_books is not None:
        before = len(df)
        df = df[df["book_key"].isin(list(allowed_books))]
        if df.empty:
            logger.warning(
                "best_lines_per_game: allowed_books filter %s removed all "
                "%d rows (no matching books in snapshot)",
                list(allowed_books), before,
            )
            return pd.DataFrame()
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"])
    df["commence_time"] = pd.to_datetime(df["commence_time"])
    df["effective_cutoff"] = (
        df["commence_time"] - pd.Timedelta(minutes=close_window_minutes)
    )

    keys = ["commence_time", "home_team", "away_team"]
    out_rows: list[dict] = []
    dropped_no_pregame = 0
    for game_key, group in df.groupby(keys, sort=False):
        cutoff = group["effective_cutoff"].iloc[0]
        pre = group[group["snapshot_ts"] <= cutoff]
        if pre.empty:
            dropped_no_pregame += 1
            continue
        chosen_ts = pre["snapshot_ts"].max()
        snap = group[group["snapshot_ts"] == chosen_ts]
        if drop_extreme_books:
            ok = (
                snap["home_price_american"].abs() <= extreme_price_threshold
            ) & (
                snap["away_price_american"].abs() <= extreme_price_threshold
            )
            snap = snap[ok]
        if snap.empty:
            continue
        home_book: str | None = None
        away_book: str | None = None
        if price_strategy == "best":
            # American odds: higher value = better for the bettor on both
            # sides (since e.g. -110 > -130 numerically AND has lower
            # implied probability).
            ih = snap["home_price_american"].idxmax()
            ia = snap["away_price_american"].idxmax()
            home_price = snap.loc[ih, "home_price_american"]
            away_price = snap.loc[ia, "away_price_american"]
            home_book = str(snap.loc[ih, "book_title"])
            away_book = str(snap.loc[ia, "book_title"])
        elif price_strategy == "median":
            home_price = snap["home_price_american"].median()
            away_price = snap["away_price_american"].median()
        elif price_strategy == "mean":
            home_price = snap["home_price_american"].mean()
            away_price = snap["away_price_american"].mean()
        else:
            raise ValueError(f"Unknown price_strategy: {price_strategy!r}")
        out_rows.append({
            "commence_time": game_key[0],
            "home_team": game_key[1],
            "away_team": game_key[2],
            "snapshot_ts": chosen_ts,
            "n_books": int(snap["book_key"].nunique()),
            "home_price_american": float(home_price),
            "away_price_american": float(away_price),
            "home_book": home_book,
            "away_book": away_book,
        })
    if dropped_no_pregame:
        logger.info(
            "best_lines_per_game: dropped %d games with no pre-cutoff snapshot "
            "(only in-game odds available)",
            dropped_no_pregame,
        )
    return pd.DataFrame(out_rows).sort_values("commence_time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Team-name -> MLBAM id resolver
# ---------------------------------------------------------------------------

def build_team_name_to_id(training_parquet_path: Path | str) -> dict[str, int]:
    """Build ``{team_name: mlbam_id}`` from a training parquet.

    Reads either home or away rows — both side columns have the same set
    of (name, id) pairs.
    """
    df = pd.read_parquet(training_parquet_path)
    home = df[["home_name", "home_id"]].rename(columns={"home_name": "name", "home_id": "id"})
    away = df[["away_name", "away_id"]].rename(columns={"away_name": "name", "away_id": "id"})
    pairs = pd.concat([home, away], ignore_index=True).drop_duplicates()
    return dict(zip(pairs["name"].astype(str), pairs["id"].astype(int)))


def attach_team_ids(
    odds_df: pd.DataFrame,
    name_to_id: dict[str, int],
) -> pd.DataFrame:
    """Add ``home_id`` and ``away_id`` columns to an odds frame.

    Rows whose team names don't resolve are dropped with a warning so
    we don't silently misalign bets.
    """
    out = odds_df.copy()
    out["home_id"] = out["home_team"].map(name_to_id).astype("Int64")
    out["away_id"] = out["away_team"].map(name_to_id).astype("Int64")
    bad = out[out["home_id"].isna() | out["away_id"].isna()]
    if not bad.empty:
        unmatched = set(bad["home_team"]) | set(bad["away_team"])
        unmatched -= set(name_to_id.keys())
        if unmatched:
            logger.warning("Dropping %d odds rows with unmapped teams: %s",
                           len(bad), sorted(unmatched))
    return out.dropna(subset=["home_id", "away_id"]).reset_index(drop=True)
