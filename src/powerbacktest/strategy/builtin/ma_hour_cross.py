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
    turn_bars: int = Field(3, ge=1, le=20)
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
    )
    params: MAHourCrossParams

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._day_signal: dict = {}  # date -> signal recorded at intraday trigger

    def warmup_bars(self) -> int:
        p = self.params
        base = p.long_window * (3 if p.ma_type == "ema" else 1)
        daily_bars_needed = base + 3 * p.turn_bars
        return daily_bars_needed * p.bars_per_day

    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        day = bars.index.normalize()
        close = bars["close"]

        # 每个交易日的最终收盘价 —— 只允许 shift 之后使用（即只用"昨天及以前"的值）
        prev_close = close.groupby(day).last()
        k = pd.Series(day).map(pd.Series(range(len(prev_close)), index=prev_close.index)).to_numpy()

        # "已完成日"均线值：对 prev_close 计算后 shift(1)，等价于只用昨天及以前的日收盘
        def completed_ma(w: int) -> np.ndarray:
            if p.ma_type == "ema":
                return prev_close.ewm(span=w, adjust=False).mean().shift(1).to_numpy()
            return prev_close.rolling(w).mean().shift(1).to_numpy()

        s_prev, l_prev = completed_ma(p.short_window), completed_ma(p.long_window)

        # 当前 bar 看到的均线 = 已完成日均线 与"当前价充当今日收盘"的一步递推
        if p.ma_type == "ema":
            a_s, a_l = 2 / (p.short_window + 1), 2 / (p.long_window + 1)
            ma_short = a_s * close + (1 - a_s) * s_prev[k]
            ma_long = a_l * close + (1 - a_l) * l_prev[k]
        else:
            sum_s = prev_close.shift(1).rolling(p.short_window - 1).sum().to_numpy()
            sum_l = prev_close.shift(1).rolling(p.long_window - 1).sum().to_numpy()
            ma_short = (sum_s[k] + close) / p.short_window
            ma_long = (sum_l[k] + close) / p.long_window

            # 均线转向：比较"昨日收盘口径"的 l_prev 与 n 天前的值（全部为已完成日，因果）
        n = p.turn_bars
        l_ref = pd.Series(l_prev).shift(n).to_numpy()
        up = pd.Series(l_prev[k] > l_ref[k], index=bars.index)
        down = pd.Series(l_prev[k] < l_ref[k], index=bars.index)

        ms, ml = pd.Series(ma_short, index=bars.index), pd.Series(ma_long, index=bars.index)
        if p.mode == "state":
            live_sig = (ms > ml).astype(float).where(ml.notna())
        else:
            cross_up = (ms > ml) & (ms.shift(1) <= ml.shift(1))
            cross_dn = (ml > ms) & (ml.shift(1) <= ms.shift(1))
            live_sig = pd.Series(float("nan"), index=bars.index)
            live_sig[cross_up & up.fillna(False)] = 1.0
            live_sig[cross_dn & down.fillna(False)] = 0.0
            live_sig = live_sig.ffill()

            # 因果的日末判定：按 bar 的结束时间标签 == 市场收盘时间（HK 1H 为 16:00）
        close_time = pd.Timestamp(p.close_time).time()  # 见下方新参数
        is_last_bar_of_day = bars.index.time == close_time

        return pd.DataFrame(
            {
                "ma_short": ma_short,
                "ma_long": ma_long,
                "daily_sig_live": live_sig.to_numpy(),
                "is_last_bar_of_day": is_last_bar_of_day,
            },
            index=bars.index,
        )

    def signals(self, bars: pd.DataFrame, ind: pd.DataFrame) -> pd.Series:
        # Decision fully delegated to on_bar.
        return pd.Series(float("nan"), index=bars.index)

    def on_bar(self, ctx: BarContext) -> float:
        d = ctx.time.normalize()
        live_sig = ctx.indicator("daily_sig_live")
        is_close = bool(ctx.indicator("is_last_bar_of_day"))

        if live_sig != live_sig:  # NaN check, no opinion yet
            return ctx.signal

        triggered = self._day_signal.get(d)

        if triggered is None:
            # First trigger today -> act immediately on the hourly bar.
            self._day_signal[d] = live_sig
            return live_sig

        if is_close:
            if live_sig != triggered:
                # Signal vanished by the daily close -> flip; next_open fill
                # defers execution to the next trading day's open.
                self._day_signal[d] = live_sig
                return live_sig
            return ctx.signal  # confirmed -> keep current position

        return ctx.signal  # mid-day, no new trigger -> hold