import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { FixedSizeList as List, type ListOnScrollProps } from "react-window";
import type { CueItem } from "../types";
import { CueRow } from "./CueRow";

type Props = {
  cues: CueItem[];
  selectedKey: string | null;
  actionBusy: boolean;
  onSelect: (key: string) => void;
  onSaveText: (cue: CueItem, text: string) => Promise<void>;
  onDelete: (cue: CueItem) => Promise<void>;
  onRestore: (cue: CueItem) => Promise<void>;
};

type RowData = {
  cues: CueItem[];
  selectedKey: string | null;
  actionBusy: boolean;
  onSelect: (key: string) => void;
  onSaveText: (cue: CueItem, text: string) => Promise<void>;
  onDelete: (cue: CueItem) => Promise<void>;
  onRestore: (cue: CueItem) => Promise<void>;
};

type ManualAnchor = {
  sourceKey: string;
  intraOffset: number;
};

const ROW_HEIGHT = 92;
const LIST_HEIGHT = 640;

export function CueList({ cues, selectedKey, actionBusy, onSelect, onSaveText, onDelete, onRestore }: Props) {
  const listRef = useRef<List>(null);
  const [autoFollow, setAutoFollow] = useState(true);
  const lastCount = useRef(0);
  const scrollOffsetRef = useRef(0);
  const manualAnchorRef = useRef<ManualAnchor | null>(null);

  function captureAnchor(scrollOffset: number) {
    if (cues.length === 0) return;
    const startIndex = Math.max(0, Math.floor(scrollOffset / ROW_HEIGHT));
    const anchorCue = cues[startIndex];
    if (!anchorCue) return;
    manualAnchorRef.current = {
      sourceKey: anchorCue.source_key,
      intraOffset: scrollOffset - startIndex * ROW_HEIGHT,
    };
  }

  useEffect(() => {
    if (autoFollow && cues.length > 0 && cues.length !== lastCount.current) {
      listRef.current?.scrollToItem(0, "start");
    }

    if (!autoFollow && !manualAnchorRef.current) {
      captureAnchor(scrollOffsetRef.current);
    }

    if (!autoFollow && manualAnchorRef.current) {
      const anchor = manualAnchorRef.current;
      const idx = cues.findIndex((it) => it.source_key === anchor.sourceKey);
      if (idx >= 0) {
        const targetOffset = idx * ROW_HEIGHT + anchor.intraOffset;
        listRef.current?.scrollTo(targetOffset);
      }
    }

    lastCount.current = cues.length;
  }, [autoFollow, cues]);

  const itemData = useMemo<RowData>(
    () => ({
      cues,
      selectedKey,
      actionBusy,
      onSelect,
      onSaveText,
      onDelete,
      onRestore,
    }),
    [cues, selectedKey, actionBusy, onSelect, onSaveText, onDelete, onRestore],
  );

  function handleScroll(props: ListOnScrollProps) {
    scrollOffsetRef.current = props.scrollOffset;

    if (!props.scrollUpdateWasRequested && props.scrollOffset > 40 && autoFollow) {
      captureAnchor(props.scrollOffset);
      setAutoFollow(false);
      return;
    }

    if (!autoFollow) {
      captureAnchor(props.scrollOffset);
    }
  }

  return (
    <div className="panel cue-list-panel">
      <div className="panel-head">
        <h3>字幕调度列表</h3>
        <div className="panel-actions">
          <span className={`follow-state ${autoFollow ? "on" : "off"}`}>{autoFollow ? "自动跟随" : "手动查看"}</span>
          <button
            className="btn mini"
            onClick={() => {
              setAutoFollow(true);
              manualAnchorRef.current = null;
              listRef.current?.scrollToItem(0, "start");
            }}
          >
            回到最新
          </button>
        </div>
      </div>

      <div className="list-wrap">
        <List<RowData>
          ref={listRef}
          height={LIST_HEIGHT}
          width="100%"
          itemCount={cues.length}
          itemSize={ROW_HEIGHT}
          itemData={itemData}
          onScroll={handleScroll}
        >
          {RowRenderer}
        </List>
      </div>
    </div>
  );
}

function RowRenderer({ index, style, data }: { index: number; style: CSSProperties; data: RowData }) {
  const cue = data.cues[index];
  return (
    <div style={style} className="row-slot">
      <CueRow
        cue={cue}
        selected={cue.source_key === data.selectedKey}
        actionBusy={data.actionBusy}
        onSelect={data.onSelect}
        onSaveText={data.onSaveText}
        onDelete={data.onDelete}
        onRestore={data.onRestore}
      />
    </div>
  );
}
