import type { RuntimeStatus } from "../types";

type Props = {
  status: RuntimeStatus | null;
  delayBusy: boolean;
  onDelayAdjust: (deltaSec: number) => Promise<void>;
  onDelayReset: () => Promise<void>;
  onJumpToLatest: () => Promise<void>;
  jumpBusy: boolean;
  onFlushPending: () => Promise<void>;
  flushBusy: boolean;
};

export function TopMetricsBar({
  status,
  delayBusy,
  onDelayAdjust,
  onDelayReset,
  onJumpToLatest,
  jumpBusy,
  onFlushPending,
  flushBusy,
}: Props) {
  const mode = (status?.mode ?? "-").toUpperCase();
  const alertLevel = status?.alert_level ?? "green";
  const queue = status?.queue;
  const stats = status?.window_stats;
  const adjust = status?.delay_adjust_sec ?? 0;
  const effective = status?.effective_delay_sec ?? 0;
  const latestReadySec =
    status?.latest_ready_remaining_ms == null
      ? null
      : status.latest_ready_remaining_ms / 1000.0;
  const latestReadyText =
    status?.latest_ready_anomaly
      ? "anomaly"
      : latestReadySec == null
        ? "-"
        : `${latestReadySec.toFixed(1)}s`;
  const fastRatio = status?.translate_fast_ratio_recent ?? 0;
  const fallbackRatio = status?.translate_fallback_ratio_recent ?? 0;
  const lineUtil = status?.line_utilization_ratio_recent ?? 0;
  const riskRatio = status?.risk_ratio;
  const deadlineRemainingMs = status?.deadline_remaining_ms;
  const termsTs = status?.terms_version?.names_mtime_ns ?? status?.terms_version?.glossary_mtime_ns ?? 0;
  const termsText = termsTs > 0 ? new Date(Math.floor(termsTs / 1_000_000)).toLocaleTimeString() : "-";

  return (
    <div className="top-grid">
      <Metric title="Mode" value={mode} emphasis={status?.mode ?? "green"} />
      <Metric title="Alert" value={alertLevel.toUpperCase()} emphasis={alertLevel} />
      <Metric title="Slack" value={status?.slack_ms == null ? "-" : `${status.slack_ms} ms`} />
      <Metric title="Backlog" value={String(queue?.unfinished_count ?? 0)} />
      <Metric
        title="Arrival / Service"
        value={`${(status?.arrival_rate_lps ?? 0).toFixed(2)} / ${(status?.service_rate_lps ?? 0).toFixed(2)} lps`}
      />
      <Metric title="Late Rate" value={`${((status?.late_rate_recent ?? 0) * 100).toFixed(1)}%`} />
      <Metric title="Latest Ready" value={latestReadyText} />
      <Metric title="Deadline Left" value={deadlineRemainingMs == null ? "-" : `${(deadlineRemainingMs / 1000).toFixed(1)}s`} />
      <Metric title="Risk Ratio" value={riskRatio == null ? "-" : `${riskRatio.toFixed(2)}x`} />
      <Metric title="Predict Px" value={status?.predict_percentile == null ? "-" : `p${status.predict_percentile}`} />
      <Metric title="Warmup/Fast Gate" value={`${status?.warmup_active ? "warmup" : "steady"} / ${status?.fast_blocked_by_latest_ready ? "blocked" : "open"}`} />
      <Metric title="Fast Model" value={`${(fastRatio * 100).toFixed(1)}%`} />
      <Metric title="Fallback" value={`${(fallbackRatio * 100).toFixed(1)}%`} />
      <Metric title="Join Ratio" value={`${((status?.s1_join_ratio_recent ?? 0) * 100).toFixed(1)}%`} />
      <Metric title="2nd Line Use" value={`${(lineUtil * 100).toFixed(1)}%`} />
      <Metric title="Terms Ver" value={termsText} />
      <Metric title="Window" value={`${stats?.displayed_count ?? 0}/${stats?.translated_count ?? 0} shown`} />

      <div className="metric-card delay-control-card">
        <div className="metric-title">Delay Align</div>
        <div className="delay-control-row">
          <button className="btn mini" onClick={() => void onDelayAdjust(-1)} disabled={delayBusy}>
            -1s
          </button>
          <div className="delay-values">
            <div>offset: {adjust.toFixed(0)}s</div>
            <div>effective: {effective.toFixed(1)}s</div>
          </div>
          <button className="btn mini" onClick={() => void onDelayAdjust(1)} disabled={delayBusy}>
            +1s
          </button>
          <button className="btn mini ghost" onClick={() => void onDelayReset()} disabled={delayBusy}>
            reset
          </button>
          <button className="btn mini warn" onClick={() => void onJumpToLatest()} disabled={jumpBusy}>
            跳转最新
          </button>
          <button className="btn mini warn" onClick={() => void onFlushPending()} disabled={flushBusy}>
            Flush未处理
          </button>
        </div>
        <div className="delay-hint">退出进程时会写回 config.toml</div>
      </div>
    </div>
  );
}

function Metric({ title, value, emphasis }: { title: string; value: string; emphasis?: string }) {
  return (
    <div className={`metric-card ${emphasis ? `mode-${emphasis}` : ""}`}>
      <div className="metric-title">{title}</div>
      <div className="metric-value">{value}</div>
    </div>
  );
}
