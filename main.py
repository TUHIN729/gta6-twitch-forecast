"""Forecast GTA VI's launch-day Twitch viewers, hour by hour, from the first hour.

Method, in one paragraph: a big console launch's Twitch audience does not stay
at its midnight-UTC hour-1 print. The game has already been live for hours in
New Zealand, Australia and Asia by then, and Europe wakes up through the day,
so hour 1 is a local dip and the day's peak lands in the UTC evening, two to
nine times hour 1 on past launches. The level that predicts the day is not
hour 1 but the evening before it: the mean of the last three hourly counts
before midnight UTC, when the launch was already rolling in the east. This
program pools that shape from the case's own comparables: for every past big
launch it takes the log ratio of each release-day hour to that launch's
pre-midnight level, and forecasts each hour as the pooled median ratio times
this case's pre-midnight level, with the five quantiles taken from the pooled
distribution of the same ratios, so the spread is the honest cross-launch
spread. Two guardrails keep it from trusting a misleading pre-midnight level:
a comparable only counts when its hour 1 was substantial (at least 20,000
viewers, so tiny categories whose midnight print was near zero cannot blow
the ratios up) and its hour 1 sat within 0.2x to 3x of its own pre-midnight
level; and when this case's own hour 1 sits far from its pre-midnight level
(outside 0.25x to 2.5x), the forecast blends halfway toward the same curves
anchored on hour 1 instead, which is what saves a launch whose audience only
arrived at midnight. Everything is a deterministic function of the case file:
no network, no clock, no randomness, standard library only.

On the kit's four practice launches this scores a mean pinball loss of about
0.20 against the house baseline's 0.387, and on a leave-one-out run over the
26 pooled big launches about 0.28 against the baseline's 0.32.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves the entry's directory OFF sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# A comparable counts only when its release-day hour 1 was a real audience.
MIN_HOUR1 = 20000
# A comparable counts only when its hour 1 sat in this band around its own
# pre-midnight level: outside it the pre-midnight level did not describe
# that launch at all (its audience arrived only at midnight, or its
# pre-midnight hours were a one-off spike).
SANE_LO = 0.2
SANE_HI = 3.0
# How many hourly counts before midnight UTC make up the pre-midnight level.
PRE_HOURS = 3
PRE_WINDOW = 6  # the wider window the sanity band above is measured over
# This case's own hour 1 outside this band around its pre-midnight level
# means the pre-midnight level may mislead: blend halfway to hour 1.
FALLBACK_LO = 0.25
FALLBACK_HI = 2.5
# Too few pooled launches for an hour to trust: fall back to persistence.
MIN_POOL = 5
# The house baseline's spread, used only for such a fallback hour.
BASELINE_SIGMA = 0.15
BASELINE_Z = [-1.645, -0.674, 0, 0.674, 1.645]


def parse_time(text):
    """An ISO 8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def quantile(sorted_values, tau):
    """The tau-quantile of a sorted list, linearly interpolated."""
    n = len(sorted_values)
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def hour_offset(moment, day_start):
    """Whole hours from the release day's start to a timestamp (floor)."""
    return int((moment - day_start).total_seconds() // 3600)


def comparable_shapes(case):
    """Each comparable's release-day shape, as log ratios to two anchors.

    Returns a list of (hour1, pre_level, day) where day maps hour 1..24 to
    viewers and pre_level is the mean of the last PRE_HOURS hourly counts
    before the release day. A comparable is kept only when its hour 1 was
    substantial and sat in the sanity band around its own pre-midnight
    level; its day's hours are used as they come, missing hours skipped.
    """
    kept = []
    for comp in case.get("comparables") or []:
        day_start = parse_time(comp["day_start"])
        by_hour = {}
        for point in comp.get("hourly") or []:
            viewers = point.get("viewers")
            if viewers is None or viewers <= 0:
                continue
            offset = hour_offset(parse_time(point["from"]), day_start)
            if -24 <= offset < 24:
                by_hour[offset + 1 if offset >= 0 else offset] = viewers
        if 1 not in by_hour:
            continue
        hour1 = by_hour[1]
        if hour1 < MIN_HOUR1:
            continue
        pre_wide = [v for h, v in by_hour.items() if -PRE_WINDOW <= h <= -1]
        pre_recent = [v for h, v in by_hour.items() if -PRE_HOURS <= h <= -1]
        if not pre_wide or not pre_recent:
            continue
        pre_level = sum(pre_recent) / len(pre_recent)
        sanity = hour1 / max(pre_wide)
        if sanity < SANE_LO or sanity > SANE_HI:
            continue
        day = {h: v for h, v in by_hour.items() if 1 <= h <= 24}
        kept.append((hour1, pre_level, day))
    return kept


def case_pre_level(case):
    """This case's pre-midnight level: the mean of its last PRE_HOURS hourly
    counts before the release day, from history_hourly, or from history
    binned by hour when history_hourly is absent."""
    day_start = parse_time(case["day_start"])
    hourly = case.get("history_hourly")
    if hourly:
        values = []
        for row in hourly:
            viewers = row.get("viewers")
            if viewers is None or viewers <= 0:
                continue
            offset = hour_offset(parse_time(row["from"]), day_start)
            if -PRE_HOURS <= offset <= -1:
                values.append(viewers)
        if values:
            return sum(values) / len(values)
    by_hour = {}
    for point in case.get("history") or []:
        viewers = point.get("viewers")
        if viewers is None or viewers <= 0:
            continue
        offset = hour_offset(parse_time(point["at"]), day_start)
        if -PRE_HOURS <= offset <= -1:
            # Minute reads in the final case: average within each hour
            # first, so a short hour does not outweigh a full one.
            by_hour.setdefault(offset, []).append(viewers)
    if by_hour:
        values = [sum(v) / len(v) for v in (by_hour[h] for h in sorted(by_hour))]
        return sum(values) / len(values)
    return None


def pooled_ratios(shapes, anchor):
    """Per-hour sorted log ratios of release-day hours to the anchor level.

    anchor is "pre" (each launch's pre-midnight level) or "hour1".
    """
    per_hour = {h: [] for h in range(2, 25)}
    for hour1, pre_level, day in shapes:
        base = pre_level if anchor == "pre" else hour1
        for h in range(2, 25):
            if h in day:
                per_hour[h].append(math.log(day[h] / base))
    for h in per_hour:
        per_hour[h].sort()
    return per_hour


def baseline_quantiles(p1, h):
    """The house baseline's row for hour h (persistence, widening spread)."""
    return [round2(p1 * math.exp(z * BASELINE_SIGMA * math.sqrt(h - 1)))
            for z in BASELINE_Z]


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    pre = case_pre_level(case)
    shapes = comparable_shapes(case)

    rows = []
    if pre is None or len(shapes) < MIN_POOL:
        # No usable pre-midnight level, or too few pooled launches: the
        # persistence baseline is the honest fallback for every hour.
        for row in case["forecast"]:
            rows.append({"hour": row["hour"], "q": baseline_quantiles(p1, row["hour"])})
        return {"hours": rows}

    ratios_pre = pooled_ratios(shapes, "pre")
    ratios_p1 = pooled_ratios(shapes, "hour1")
    ratio = p1 / pre
    weight_pre = 1.0 if FALLBACK_LO <= ratio <= FALLBACK_HI else 0.5
    log_pre = math.log(pre)
    log_p1 = math.log(p1)

    for row in case["forecast"]:
        h = row["hour"]
        if len(ratios_pre[h]) < MIN_POOL:
            rows.append({"hour": h, "q": baseline_quantiles(p1, h)})
            continue
        med = (weight_pre * (log_pre + quantile(ratios_pre[h], 0.5))
               + (1.0 - weight_pre) * (log_p1 + quantile(ratios_p1[h], 0.5)))
        qs = []
        for tau in LEVELS:
            spread = (weight_pre * (quantile(ratios_pre[h], tau) - quantile(ratios_pre[h], 0.5))
                      + (1.0 - weight_pre) * (quantile(ratios_p1[h], tau) - quantile(ratios_p1[h], 0.5)))
            value = math.exp(med + spread)
            value = min(max(value, 1.0), 1_000_000_000.0)
            qs.append(round2(value))
        for i in range(1, 5):
            qs[i] = max(qs[i], qs[i - 1])
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
