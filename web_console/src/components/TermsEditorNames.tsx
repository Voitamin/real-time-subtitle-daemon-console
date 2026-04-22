import { useMemo, useState } from "react";
import type { NameItem } from "../types";

type Props = {
  items: NameItem[];
  onUpsert: (content: string, soundsLike: string[], prevContent?: string) => Promise<void>;
  onDelete: (content: string) => Promise<void>;
};

function parseSoundsLike(text: string): string[] {
  const normalized = text.replace(/，/g, ",").replace(/、/g, ",");
  const out: string[] = [];
  const seen = new Set<string>();
  for (const token of normalized.split(",")) {
    const value = token.trim();
    if (!value || seen.has(value)) continue;
    seen.add(value);
    out.push(value);
  }
  return out;
}

export function TermsEditorNames({ items, onUpsert, onDelete }: Props) {
  const [content, setContent] = useState("");
  const [soundsLikeText, setSoundsLikeText] = useState("");
  const [drafts, setDrafts] = useState<Record<string, { content: string; soundsLikeText: string }>>({});

  const sorted = useMemo(
    () => [...items].sort((a, b) => a.content.localeCompare(b.content)),
    [items],
  );

  return (
    <div className="panel terms-panel">
      <div className="panel-head">
        <h3>Names Whitelist</h3>
      </div>

      <div className="terms-add-row terms-add-row-names">
        <input value={content} onChange={(e) => setContent(e.target.value)} placeholder="content（名称本体）" />
        <input
          value={soundsLikeText}
          onChange={(e) => setSoundsLikeText(e.target.value)}
          placeholder="sounds_like（かな1,かな2，可空）"
        />
        <button
          className="btn mini"
          onClick={() => {
            const nextContent = content.trim();
            if (!nextContent) return;
            const nextSoundsLike = parseSoundsLike(soundsLikeText);
            void onUpsert(nextContent, nextSoundsLike);
            setContent("");
            setSoundsLikeText("");
          }}
        >
          新增
        </button>
      </div>

      <div className="terms-table-wrap">
        <table className="terms-table">
          <thead>
            <tr>
              <th>Content</th>
              <th>Sounds Like</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {sorted.map((item) => {
              const draft = drafts[item.content];
              const contentValue = draft?.content ?? item.content;
              const soundsLikeValue = draft?.soundsLikeText ?? item.sounds_like.join(",");
              return (
                <tr key={item.content}>
                  <td>
                    <input
                      className="terms-inline-input"
                      value={contentValue}
                      onChange={(e) =>
                        setDrafts((old) => ({
                          ...old,
                          [item.content]: { content: e.target.value, soundsLikeText: soundsLikeValue },
                        }))
                      }
                    />
                  </td>
                  <td>
                    <input
                      className="terms-inline-input"
                      value={soundsLikeValue}
                      onChange={(e) =>
                        setDrafts((old) => ({
                          ...old,
                          [item.content]: { content: contentValue, soundsLikeText: e.target.value },
                        }))
                      }
                    />
                  </td>
                  <td className="terms-actions-cell">
                    <button
                      className="btn mini"
                      onClick={() => {
                        const nextContent = contentValue.trim();
                        if (!nextContent) return;
                        const nextSoundsLike = parseSoundsLike(soundsLikeValue);
                        void onUpsert(nextContent, nextSoundsLike, item.content);
                      }}
                    >
                      保存
                    </button>
                    <button className="btn mini ghost" onClick={() => void onDelete(item.content)}>
                      删除
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
