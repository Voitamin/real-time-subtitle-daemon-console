import { useMemo, useState } from "react";
import type { GlossaryItem } from "../types";

type Props = {
  items: GlossaryItem[];
  onUpsert: (ja: string, zh: string) => Promise<void>;
  onDelete: (ja: string) => Promise<void>;
};

export function TermsEditorGlossary({ items, onUpsert, onDelete }: Props) {
  const [ja, setJa] = useState("");
  const [zh, setZh] = useState("");
  const [drafts, setDrafts] = useState<Record<string, string>>({});

  const sortedItems = useMemo(() => [...items].sort((a, b) => a.ja.localeCompare(b.ja)), [items]);

  return (
    <div className="panel terms-panel">
      <div className="panel-head">
        <h3>Glossary</h3>
      </div>

      <div className="terms-add-row">
        <input value={ja} onChange={(e) => setJa(e.target.value)} placeholder="日文术语" />
        <input value={zh} onChange={(e) => setZh(e.target.value)} placeholder="中文翻译" />
        <button
          className="btn mini"
          onClick={() => {
            const a = ja.trim();
            const b = zh.trim();
            if (!a || !b) return;
            void onUpsert(a, b);
            setJa("");
            setZh("");
          }}
        >
          新增
        </button>
      </div>

      <div className="terms-table-wrap">
        <table className="terms-table">
          <thead>
            <tr>
              <th>JA</th>
              <th>ZH</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {sortedItems.map((item) => {
              const value = drafts[item.ja] ?? item.zh;
              return (
                <tr key={item.ja}>
                  <td>{item.ja}</td>
                  <td>
                    <input
                      className="terms-inline-input"
                      value={value}
                      onChange={(e) => setDrafts((old) => ({ ...old, [item.ja]: e.target.value }))}
                    />
                  </td>
                  <td className="terms-actions-cell">
                    <button
                      className="btn mini"
                      onClick={() => {
                        const next = (drafts[item.ja] ?? item.zh).trim();
                        if (!next) return;
                        void onUpsert(item.ja, next);
                      }}
                    >
                      保存
                    </button>
                    <button className="btn mini ghost" onClick={() => void onDelete(item.ja)}>
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
