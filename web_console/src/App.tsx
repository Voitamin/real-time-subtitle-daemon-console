import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  adjustDelay,
  deleteCue,
  deleteGlossary,
  deleteName,
  fetchCuesWindow,
  fetchGlossary,
  fetchNames,
  flushPending,
  jumpToLatest,
  patchCueText,
  resetDelayAdjust,
  restoreCue,
  upsertName,
  upsertGlossary,
} from "./api";
import { CueDetailDrawer } from "./components/CueDetailDrawer";
import { CueList } from "./components/CueList";
import { TermsEditorGlossary } from "./components/TermsEditorGlossary";
import { TermsEditorNames } from "./components/TermsEditorNames";
import { TopMetricsBar } from "./components/TopMetricsBar";
import type { CueItem, RuntimeStatus } from "./types";

type StreamState = "connecting" | "connected" | "reconnecting";

function isTranslated(cue: CueItem): boolean {
  return cue.status === "TRANSLATED" || cue.status === "FALLBACK_READY";
}

export default function App() {
  const qc = useQueryClient();
  const [status, setStatus] = useState<RuntimeStatus | null>(null);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string>("");
  const [termsMsg, setTermsMsg] = useState<string>("");
  const [streamState, setStreamState] = useState<StreamState>("connecting");

  useEffect(() => {
    let stopped = false;
    let es: EventSource | null = null;
    let retryTimer: number | null = null;

    function cleanupStream() {
      if (es) {
        es.close();
        es = null;
      }
      if (retryTimer != null) {
        window.clearTimeout(retryTimer);
        retryTimer = null;
      }
    }

    function connect() {
      if (stopped) return;
      setStreamState((prev) => (prev === "connected" ? "reconnecting" : "connecting"));
      es = new EventSource("/api/stream");

      es.onopen = () => {
        setStreamState("connected");
      };

      es.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as RuntimeStatus;
          setStatus(payload);
        } catch {
          // ignore malformed events
        }
      };

      es.onerror = () => {
        if (stopped) return;
        cleanupStream();
        setStreamState("reconnecting");
        retryTimer = window.setTimeout(connect, 1000);
      };
    }

    connect();
    return () => {
      stopped = true;
      cleanupStream();
    };
  }, []);

  const windowSec = status?.future_window_sec ?? status?.window_sec ?? 270;

  const cuesQ = useQuery({
    queryKey: ["cues-window", windowSec],
    queryFn: () => fetchCuesWindow(windowSec, 800),
    refetchInterval: 1000,
  });
  const glossaryQ = useQuery({
    queryKey: ["terms-glossary"],
    queryFn: fetchGlossary,
    refetchInterval: 3000,
  });
  const namesQ = useQuery({
    queryKey: ["terms-names"],
    queryFn: fetchNames,
    refetchInterval: 3000,
  });

  const cues = cuesQ.data?.items ?? [];
  const selectedCue = useMemo(() => cues.find((cue) => cue.source_key === selectedKey) ?? null, [cues, selectedKey]);

  useEffect(() => {
    if (cues.length === 0) {
      setSelectedKey(null);
      return;
    }

    if (selectedKey && cues.some((cue) => cue.source_key === selectedKey)) {
      return;
    }

    const preferred = status?.latest_editable_source_key;
    if (preferred && cues.some((cue) => cue.source_key === preferred)) {
      setSelectedKey(preferred);
      return;
    }

    const latestTranslated = cues.find((cue) => isTranslated(cue));
    setSelectedKey((latestTranslated ?? cues[0]).source_key);
  }, [cues, selectedKey, status?.latest_editable_source_key]);

  const editMutation = useMutation({
    mutationFn: async ({ cue, text }: { cue: CueItem; text: string }) => {
      await patchCueText(cue.source_key, text, cue.revision);
    },
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ["cues-window"] });
    },
    onError: (err) => setErrorMsg(String(err)),
  });

  const deleteMutation = useMutation({
    mutationFn: async (cue: CueItem) => {
      await deleteCue(cue.source_key, cue.revision);
    },
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ["cues-window"] });
    },
    onError: (err) => setErrorMsg(String(err)),
  });

  const restoreMutation = useMutation({
    mutationFn: async (cue: CueItem) => {
      await restoreCue(cue.source_key, cue.revision);
    },
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ["cues-window"] });
    },
    onError: (err) => setErrorMsg(String(err)),
  });

  const delayMutation = useMutation({
    mutationFn: async (deltaSec: number) => adjustDelay(deltaSec),
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ["cues-window"] });
    },
    onError: (err) => setErrorMsg(String(err)),
  });

  const delayResetMutation = useMutation({
    mutationFn: async () => resetDelayAdjust(),
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ["cues-window"] });
    },
    onError: (err) => setErrorMsg(String(err)),
  });

  const jumpMutation = useMutation({
    mutationFn: async () => jumpToLatest(),
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ["cues-window"] });
    },
    onError: (err) => setErrorMsg(String(err)),
  });

  const flushMutation = useMutation({
    mutationFn: async () => flushPending(),
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ["cues-window"] });
    },
    onError: (err) => setErrorMsg(String(err)),
  });

  const actionBusy =
    editMutation.isPending ||
    deleteMutation.isPending ||
    restoreMutation.isPending ||
    delayMutation.isPending ||
    delayResetMutation.isPending ||
    jumpMutation.isPending ||
    flushMutation.isPending;

  return (
    <div className="app-root">
      <header className="app-header">
        <div>
          <h1>Console V3</h1>
          <div className="subhead">字幕调度台 | 可编辑未展示字幕 | 术语热更新</div>
        </div>
        <div className={`stream-state ${streamState}`}>{streamState}</div>
      </header>

      <TopMetricsBar
        status={status}
        delayBusy={delayMutation.isPending || delayResetMutation.isPending}
        onDelayAdjust={async (deltaSec) => {
          await delayMutation.mutateAsync(deltaSec);
        }}
        onDelayReset={async () => {
          await delayResetMutation.mutateAsync();
        }}
        onJumpToLatest={async () => {
          await jumpMutation.mutateAsync();
        }}
        jumpBusy={jumpMutation.isPending}
        onFlushPending={async () => {
          await flushMutation.mutateAsync();
        }}
        flushBusy={flushMutation.isPending}
      />

      {errorMsg ? (
        <div className="toast error">
          <span>{errorMsg}</span>
          <button className="btn mini ghost" onClick={() => setErrorMsg("")}>关闭</button>
        </div>
      ) : null}

      <main className="main-layout">
        <section className="left-pane">
          <CueList
            cues={cues}
            selectedKey={selectedKey}
            actionBusy={actionBusy}
            onSelect={setSelectedKey}
            onSaveText={async (cue, text) => {
              await editMutation.mutateAsync({ cue, text });
            }}
            onDelete={async (cue) => {
              await deleteMutation.mutateAsync(cue);
            }}
            onRestore={async (cue) => {
              await restoreMutation.mutateAsync(cue);
            }}
          />
        </section>

        <aside className="right-pane">
          <TermsEditorGlossary
            items={glossaryQ.data?.items ?? []}
            onUpsert={async (ja, zh) => {
              try {
                await upsertGlossary(ja, zh);
                await qc.invalidateQueries({ queryKey: ["terms-glossary"] });
                setTermsMsg(`Glossary 已保存：${ja}`);
              } catch (err) {
                setErrorMsg(String(err));
              }
            }}
            onDelete={async (ja) => {
              try {
                await deleteGlossary(ja);
                await qc.invalidateQueries({ queryKey: ["terms-glossary"] });
                setTermsMsg(`Glossary 已删除：${ja}`);
              } catch (err) {
                setErrorMsg(String(err));
              }
            }}
          />

          <TermsEditorNames
            items={namesQ.data?.items ?? []}
            onUpsert={async (content, soundsLike, prevContent) => {
              try {
                await upsertName(content, soundsLike, prevContent);
                await qc.invalidateQueries({ queryKey: ["terms-names"] });
                setTermsMsg(`Names 已保存：${content}`);
              } catch (err) {
                setErrorMsg(String(err));
              }
            }}
            onDelete={async (content) => {
              try {
                await deleteName(content);
                await qc.invalidateQueries({ queryKey: ["terms-names"] });
                setTermsMsg(`Names 已删除：${content}`);
              } catch (err) {
                setErrorMsg(String(err));
              }
            }}
          />

          <div className="hint-box">
            <div>术语保存后预计 2 秒内生效（由后端热加载）。</div>
            {termsMsg ? <div className="terms-msg">{termsMsg}</div> : null}
          </div>
        </aside>
      </main>

      <CueDetailDrawer cue={selectedCue} />
    </div>
  );
}
