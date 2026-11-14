"""Daily MA cross signal, executed on hourly bars with close-confirm / next-day reversal."""

from __future__ import annotations

from typing import Literal

import pandas as pd
import numpy as np
from pydantic import Field, model_validator

from powerbacktest.strategy.base import BarContext, PlotSpec, Strategy, StrategyParams
from powerbacktest.strategy.registry import register
from powerbacktest.strategy.tdx import CROSS, EMA, MA, REF


class MAHourCrossParams(StrategyParams):
    short_window: int = Field(20, ge=1, le=500)
    long_window: int = Field(50, ge=2, le=1000)
    ma_type: Literal["sma", "ema"] = "sma"
    mode: Literal["cross", "state"] = "cross"
    turn_bars: int = Field(1, ge=1, le=20)
    bars_per_day: int = Field(4, ge=1, le=24, description="hourly bars per trading day")
    close_time: str = Field(
        "16:00",
        description="end-label of the last intraday bar of a day",
    )

    @model_validator(mode="after")
    def _order(self) -> "MAHourCrossParams":
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be less than long_window")
        return self


@register
class MAHourCrossStrategy(Strategy):
    name = "ma_hour_cross"
    title = "Daily MA cross, hourly execution"
    description = (
        "Daily short/long MA cross decides direction; executes intraday on hourly bars, "
        "confirms at the daily close, and reverses at next day's open if the signal vanished."
    )
    Params = MAHourCrossParams
    plots = (
        PlotSpec("ma_short", "price", "line", "MA short (daily)"),
        PlotSpec("ma_long", "price", "line", "MA long (daily)"),
        PlotSpec("daily_sig_live", "lower", "line", "Daily signal (live)"),  #1108新增
    )
    params: MAHourCrossParams

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._day_signal: dict = {}  # date -> signal triggered today
        self._last_sig: dict[str, float] = {}
        #self._last_sig: float = float("nan")  # last committed daily signal

    def warmup_bars(self) -> int:
        p = self.params
        base = p.long_window * (3 if p.ma_type == "ema" else 1)
        daily_bars_needed = base + 3 * p.turn_bars
        return daily_bars_needed * p.bars_per_day

    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        fn = MA if p.ma_type == "sma" else EMA
        day = bars.index.normalize()

        # Completed daily closes; today's entry is only used via .shift(1)+ views.
        daily_close = bars.groupby(day)["close"].last()
        d_ma_s = fn(daily_close, p.short_window)
        d_ma_l = fn(daily_close, p.long_window)

        # Broadcast completed-day values back to bar level as plain arrays
        # (.reindex(day) keeps the day index -> must use .to_numpy()).
        c = bars["close"].to_numpy()
        prev_ma_s = d_ma_s.shift(1).reindex(day).to_numpy()
        prev_ma_l = d_ma_l.shift(1).reindex(day).to_numpy()
        ref_l = d_ma_l.shift(p.turn_bars).reindex(day).to_numpy()

        # Provisional daily MA: today's close = this bar's own close.
        if p.ma_type == "sma":
            prev_s = daily_close.rolling(p.short_window - 1).sum().shift(1).reindex(day).to_numpy()
            prev_l = daily_close.rolling(p.long_window - 1).sum().shift(1).reindex(day).to_numpy()
            ma_short = (prev_s + c) / p.short_window
            ma_long = (prev_l + c) / p.long_window
        else:
            a_s, a_l = 2 / (p.short_window + 1), 2 / (p.long_window + 1)
            ma_short = a_s * c + (1 - a_s) * prev_ma_s
            ma_long = a_l * c + (1 - a_l) * prev_ma_l

        up = ma_long > ref_l
        down = ma_long < ref_l

        if p.mode == "state":
            live_sig = pd.Series(
                np.where(np.isnan(ma_long), np.nan, (ma_short > ma_long).astype(float)),
                index=bars.index,
            )
        else:
            live_sig = pd.Series(float("nan"), index=bars.index)
            live_sig[(ma_short > ma_long) & (prev_ma_s <= prev_ma_l) & up] = 1.0
            live_sig[(ma_long > ma_short) & (prev_ma_l <= prev_ma_s) & down] = 0.0
            live_sig = live_sig.ffill()

        close_t = pd.Timestamp(p.close_time).time()
        is_last_bar_of_day = bars.index.time == close_t

        return pd.DataFrame(
            {
                "ma_short": ma_short,
                "ma_long": ma_long,
                "daily_sig_live": live_sig,
                "is_last_bar_of_day": is_last_bar_of_day,
            },
            index=bars.index,
        )

    def signals(self, bars: pd.DataFrame, ind: pd.DataFrame) -> pd.Series:
        # Decision fully delegated to on_bar.
        return pd.Series(float("nan"), index=bars.index)

    def on_bar(self, ctx: BarContext) -> float:
        sym = ctx.symbol
        d = ctx.time.normalize()
        live_sig = ctx.indicator("daily_sig_live")
        is_close = bool(ctx.indicator("is_last_bar_of_day"))

        if live_sig != live_sig:  # NaN, no opinion yet
            return ctx.signal

        triggered = self._day_signal.get((sym, d))
        last_sig = self._last_sig.get(sym, float("nan"))
        out = ctx.signal

        if triggered is None and live_sig != last_sig:
            self._day_signal[(sym, d)] = live_sig
            out = live_sig
        elif triggered is not None and is_close and live_sig != triggered:
            self._day_signal[(sym, d)] = live_sig
            out = live_sig

        if is_close:
            self._last_sig[sym] = live_sig
        return out