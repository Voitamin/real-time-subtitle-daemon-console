import type { CueItem } from "../types";

type Props = {
  cue: CueItem | null;
};

export function CueDetailDrawer({ cue }: Props) {
  if (!cue) {
    return (
      <div className="detail-drawer empty">
        <div className="detail-title">详情面板</div>
        <div className="detail-empty">点击列表中的字幕查看详细信息</div>
      </div>
    );
  }

  return (
    <div className="detail-drawer">
      <div className="detail-head">
        <div>
          <div className="detail-title">{cue.source_key}</div>
          <div className="detail-sub">revision: {cue.revision}</div>
        </div>
      </div>

      <div className="detail-grid">
        <DetailItem label="原文 JP" value={cue.jp_raw} />
        <DetailItem label="纠错 JP" value={cue.jp_corrected || "-"} />
        <DetailItem label="系统译文 ZH" value={cue.zh_text || "-"} />
        <DetailItem label="人工译文 ZH" value={cue.manual_zh_text || "-"} />
        <DetailItem
          label="阶段耗时"
          value={`S1 ${cue.stage1_latency_ms ?? "-"}ms | S2 ${cue.stage2_latency_ms ?? "-"}ms | P ${cue.pipeline_latency_ms ?? "-"}ms`}
        />
        <DetailItem label="Provider" value={`${cue.stage1_provider ?? "-"} -> ${cue.stage2_provider ?? "-"}`} />
        <DetailItem label="Fast Model" value={cue.used_fast_model ? "是" : "否"} />
        <DetailItem label="Fallback" value={cue.fallback_used ? "是" : "否"} />
        <DetailItem label="并入下一句" value={cue.display_suppressed ? `是 -> ${cue.join_target_source_key ?? "-"}` : "否"} />
        <DetailItem label="已展示" value={cue.displayed_at_mono_ms ? "是" : "否"} />
        <DetailItem label="错误" value={cue.last_error || "-"} />
      </div>
    </div>
  );
}

function DetailItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="detail-item">
      <div className="detail-label">{label}</div>
      <div className="detail-value">{value}</div>
    </div>
  );
}
