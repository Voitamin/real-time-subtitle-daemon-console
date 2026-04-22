import { useEffect, useState } from "react";
import type { CueItem } from "../types";
import { CountdownBar } from "./CountdownBar";
import { StatusPill } from "./StatusPill";

type Props = {
  cue: CueItem;
  selected: boolean;
  actionBusy: boolean;
  onSelect: (key: string) => void;
  onSaveText: (cue: CueItem, text: string) => Promise<void>;
  onDelete: (cue: CueItem) => Promise<void>;
  onRestore: (cue: CueItem) => Promise<void>;
};

export function CueRow({ cue, selected, actionBusy, onSelect, onSaveText, onDelete, onRestore }: Props) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(cue.display_text);
  const [saving, setSaving] = useState(false);

  const canEdit = cue.editable && cue.status !== "NEW" && !cue.deleted_soft && !cue.display_suppressed;
  const canDelete = cue.displayed_at_mono_ms == null && cue.countdown_sec > 0 && !cue.deleted_soft;
  const canRestore = cue.deleted_soft;

  useEffect(() => {
    if (!editing) {
      setDraft(cue.display_text);
    }
  }, [cue.display_text, editing]);

  async function handleSave() {
    if (!canEdit || saving) return;
    const next = draft.trim();
    if (!next) return;
    setSaving(true);
    try {
      await onSaveText(cue, next);
      setEditing(false);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div
      className={`cue-row ${selected ? "cue-row-selected" : ""} ${cue.deleted_soft ? "cue-row-deleted" : ""}`}
      onClick={() => onSelect(cue.source_key)}
    >
      <div className="cue-main">
        <div className="cue-meta-line">
          <span className="cue-key">{cue.source_key}</span>
          {cue.manual_locked ? <span className="cue-tag">人工锁定</span> : null}
          {cue.used_fast_model ? <span className="cue-tag fast">FAST</span> : null}
          {cue.fallback_used ? <span className="cue-tag warn">fallback</span> : null}
          {cue.display_suppressed ? <span className="cue-tag">并入下一句</span> : null}
        </div>
        {editing && canEdit ? (
          <div className="cue-edit-wrap" onClick={(e) => e.stopPropagation()}>
            <input
              className="cue-edit-input"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  void handleSave();
                }
                if (e.key === "Escape") {
                  setEditing(false);
                  setDraft(cue.display_text);
                }
              }}
            />
            <button className="btn mini" onClick={() => void handleSave()} disabled={saving || actionBusy}>
              保存
            </button>
            <button
              className="btn mini ghost"
              onClick={() => {
                setEditing(false);
                setDraft(cue.display_text);
              }}
              disabled={saving || actionBusy}
            >
              取消
            </button>
          </div>
        ) : (
          <button
            className={`cue-text-btn ${canEdit ? "editable" : "readonly"}`}
            onClick={(e) => {
              e.stopPropagation();
              if (!canEdit) return;
              setDraft(cue.display_text);
              setEditing(true);
            }}
            title={canEdit ? "点击编辑" : "当前不可编辑（未翻译/已展示/已过期/已删除）"}
          >
            {cue.display_text}
          </button>
        )}
      </div>

      <div className="cue-time">
        <CountdownBar countdownSec={cue.countdown_sec} progress={cue.progress} />
      </div>

      <div className="cue-status">
        <StatusPill
          displayed={cue.displayed_at_mono_ms !== null}
          deleted={cue.deleted_soft}
          late={cue.dropped_late}
          suppressed={cue.display_suppressed}
        />
        <div className="cue-actions" onClick={(e) => e.stopPropagation()}>
          {!cue.deleted_soft ? (
            <button className="btn mini warn" onClick={() => void onDelete(cue)} disabled={!canDelete || actionBusy}>
              删除
            </button>
          ) : (
            <button className="btn mini" onClick={() => void onRestore(cue)} disabled={!canRestore || actionBusy}>
              恢复
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
