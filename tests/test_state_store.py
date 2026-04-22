from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from live_sub_daemon.models import SourceCue
from live_sub_daemon.state_store import StateStore


class StateStoreTests(unittest.TestCase):
    def test_source_debug_fields_persist(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "state.sqlite3"
            state = StateStore(db)
            cue = SourceCue(
                source_key="speechmatics:1:1000",
                source_kind="speechmatics",
                srt_index=None,
                start_ms=None,
                end_ms=None,
                jp_raw="raw_text",
                jp_aggregated="agg_text",
                jp_canonicalized="canon_text",
                aggregator_reason="sentence_punct",
            )
            state.upsert_source_cues([cue], now_mono_ms=1000, delay_ms=1000)
            row = state.fetch_cue_by_key(cue.source_key)
            state.close()

            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row.jp_raw, "raw_text")
            self.assertEqual(row.jp_aggregated, "agg_text")
            self.assertEqual(row.jp_canonicalized, "canon_text")
            self.assertEqual(row.aggregator_reason, "sentence_punct")
            self.assertEqual(row.display_suppressed, False)
            self.assertIsNone(row.join_target_source_key)

    def test_late_translation_not_rendered_but_kept_for_srt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "state.sqlite3"
            state = StateStore(db)
            cue = SourceCue(
                source_key="srt:1:0:1000",
                source_kind="srt",
                srt_index=1,
                start_ms=0,
                end_ms=1000,
                jp_raw="原文",
            )
            state.upsert_source_cues([cue], now_mono_ms=1000, delay_ms=1000)
            batch = state.fetch_new_batch(5)
            state.save_pipeline_results(
                cues=batch,
                corrected_texts={cue.source_key: "纠错"},
                translated_texts={cue.source_key: "翻译"},
                translated_at_mono_ms=2500,
                fallback_mode="jp_raw",
                llm_latency_ms=500,
                error_message=None,
            )

            render_candidates = state.fetch_render_candidates(now_mono_ms=9999, limit=2)
            srt_cues = state.fetch_srt_ready()
            state.close()

            self.assertEqual(render_candidates, [])
            self.assertEqual(len(srt_cues), 1)
            self.assertEqual(srt_cues[0].zh_text, "翻译")

    def test_fetch_and_claim_batch_sets_inflight(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "state.sqlite3"
            state = StateStore(db)
            cues = [
                SourceCue(
                    source_key=f"srt:{i}:0:1000",
                    source_kind="srt",
                    srt_index=i,
                    start_ms=0,
                    end_ms=1000,
                    jp_raw=f"原文{i}",
                )
                for i in (1, 2, 3)
            ]
            state.upsert_source_cues(cues, now_mono_ms=1000, delay_ms=1000)
            batch = state.fetch_and_claim_batch(limit=2, owner="w1", now_mono_ms=1200)
            stats = state.fetch_new_queue_stats(now_mono_ms=1300)
            state.close()

            self.assertEqual(len(batch), 2)
            self.assertEqual(stats["new_count"], 1)
            self.assertEqual(stats["inflight_count"], 2)

    def test_min_due_unfinished_includes_inflight(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "state.sqlite3"
            state = StateStore(db)
            c1 = SourceCue("srt:1:0:1000", "srt", 1, 0, 1000, "a")
            c2 = SourceCue("srt:2:0:1000", "srt", 2, 0, 1000, "b")
            state.upsert_source_cues([c1], now_mono_ms=1000, delay_ms=1000)  # due=2000
            state.upsert_source_cues([c2], now_mono_ms=2000, delay_ms=1000)  # due=3000
            state.fetch_and_claim_batch(limit=1, owner="w1", now_mono_ms=2100)
            stats = state.fetch_new_queue_stats(now_mono_ms=2200)
            state.close()

            self.assertEqual(stats["inflight_count"], 1)
            self.assertEqual(stats["new_count"], 1)
            self.assertEqual(stats["min_due_unfinished_mono_ms"], 2000)

    def test_manual_edit_revision_and_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "state.sqlite3"
            state = StateStore(db)
            cue = SourceCue("srt:1:0:1000", "srt", 1, 0, 1000, "原文")
            state.upsert_source_cues([cue], now_mono_ms=1000, delay_ms=5000)
            batch = state.fetch_new_batch(5)
            state.save_pipeline_results(
                cues=batch,
                corrected_texts={cue.source_key: "纠错"},
                translated_texts={cue.source_key: "翻译"},
                translated_at_mono_ms=1500,
                fallback_mode="jp_raw",
                llm_latency_ms=200,
            )
            row = state.fetch_cue_by_key(cue.source_key)
            self.assertIsNotNone(row)
            assert row is not None
            rev = row.revision

            ok = state.upsert_manual_translation(
                source_key=cue.source_key,
                text="人工翻译",
                now_mono_ms=2000,
                expected_revision=rev,
            )
            self.assertEqual(ok.get("ok"), True)

            conflict = state.upsert_manual_translation(
                source_key=cue.source_key,
                text="人工翻译2",
                now_mono_ms=2100,
                expected_revision=rev,
            )
            row2 = state.fetch_cue_by_key(cue.source_key)
            state.close()

            self.assertEqual(conflict.get("ok"), False)
            self.assertEqual(conflict.get("code"), "revision_conflict")
            self.assertIsNotNone(row2)
            assert row2 is not None
            self.assertEqual(row2.manual_zh_text, "人工翻译")
            self.assertEqual(row2.manual_locked, True)

    def test_soft_delete_excludes_render_and_srt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "state.sqlite3"
            state = StateStore(db)
            cue = SourceCue("srt:1:0:1000", "srt", 1, 0, 1000, "原文")
            state.upsert_source_cues([cue], now_mono_ms=1000, delay_ms=5000)  # due=6000
            batch = state.fetch_new_batch(5)
            state.save_pipeline_results(
                cues=batch,
                corrected_texts={cue.source_key: "纠错"},
                translated_texts={cue.source_key: "翻译"},
                translated_at_mono_ms=1500,
                fallback_mode="jp_raw",
                llm_latency_ms=200,
            )

            row = state.fetch_cue_by_key(cue.source_key)
            assert row is not None
            deleted = state.soft_delete_cue(
                source_key=cue.source_key,
                now_mono_ms=2000,
                expected_revision=row.revision,
            )
            self.assertEqual(deleted.get("ok"), True)
            render_candidates = state.fetch_render_candidates(now_mono_ms=7000, limit=2)
            srt_cues = state.fetch_srt_ready()

            restored = state.restore_cue(source_key=cue.source_key, expected_revision=None)
            render_after_restore = state.fetch_render_candidates(now_mono_ms=7000, limit=2)
            state.close()

            self.assertEqual(restored.get("ok"), True)
            self.assertEqual(render_candidates, [])
            self.assertEqual(srt_cues, [])
            self.assertEqual(len(render_after_restore), 1)

    def test_mark_displayed_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "state.sqlite3"
            state = StateStore(db)
            cue = SourceCue("srt:1:0:1000", "srt", 1, 0, 1000, "原文")
            state.upsert_source_cues([cue], now_mono_ms=1000, delay_ms=1000)
            batch = state.fetch_new_batch(5)
            state.save_pipeline_results(
                cues=batch,
                corrected_texts={cue.source_key: "纠错"},
                translated_texts={cue.source_key: "翻译"},
                translated_at_mono_ms=1200,
                fallback_mode="jp_raw",
                llm_latency_ms=100,
            )
            c1 = state.mark_displayed([cue.source_key], displayed_at_mono_ms=5000)
            c2 = state.mark_displayed([cue.source_key], displayed_at_mono_ms=8000)
            row = state.fetch_cue_by_key(cue.source_key)
            state.close()

            self.assertEqual(c1, 1)
            self.assertEqual(c2, 0)
            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row.displayed_at_mono_ms, 5000)

    def test_save_pipeline_results_marks_s1_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "state.sqlite3"
            state = StateStore(db)
            cue = SourceCue("srt:1:0:1000", "srt", 1, 0, 1000, "原文")
            state.upsert_source_cues([cue], now_mono_ms=1000, delay_ms=1000)
            batch = state.fetch_new_batch(5)
            state.save_pipeline_results(
                cues=batch,
                corrected_texts={cue.source_key: "原文"},
                translated_texts={cue.source_key: "翻译"},
                translated_at_mono_ms=1200,
                fallback_mode="jp_raw",
                llm_latency_ms=100,
                s1_skipped=True,
            )
            row = state.fetch_cue_by_key(cue.source_key)
            state.close()

            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row.s1_skipped, True)

    def test_display_suppressed_excluded_from_render_and_srt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "state.sqlite3"
            state = StateStore(db)
            cues = [
                SourceCue("srt:1:0:1000", "srt", 1, 0, 1000, "前半"),
                SourceCue("srt:2:1000:2000", "srt", 2, 1000, 2000, "后半"),
            ]
            state.upsert_source_cues(cues, now_mono_ms=1000, delay_ms=1000)
            batch = state.fetch_new_batch(10)
            state.save_pipeline_results(
                cues=batch,
                corrected_texts={
                    cues[0].source_key: "前半",
                    cues[1].source_key: "前半后半",
                },
                translated_texts={
                    cues[1].source_key: "合并后的中文",
                },
                translated_at_mono_ms=1500,
                fallback_mode="jp_raw",
                llm_latency_ms=120,
                display_suppressed_map={cues[0].source_key: True},
                join_target_map={cues[0].source_key: cues[1].source_key},
            )
            render_candidates = state.fetch_render_candidates(now_mono_ms=3000, limit=10)
            srt_ready = state.fetch_srt_ready()
            suppressed_row = state.fetch_cue_by_key(cues[0].source_key)
            shown_row = state.fetch_cue_by_key(cues[1].source_key)
            state.close()

            self.assertEqual([c.source_key for c in render_candidates], [cues[1].source_key])
            self.assertEqual([c.source_key for c in srt_ready], [cues[1].source_key])
            assert suppressed_row is not None
            assert shown_row is not None
            self.assertEqual(suppressed_row.display_suppressed, True)
            self.assertEqual(suppressed_row.join_target_source_key, cues[1].source_key)
            self.assertEqual(shown_row.display_suppressed, False)

    def test_soft_delete_uses_effective_due_with_delay_adjust(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "state.sqlite3"
            state = StateStore(db)
            cue = SourceCue("srt:1:0:1000", "srt", 1, 0, 1000, "原文")
            state.upsert_source_cues([cue], now_mono_ms=1000, delay_ms=5000)  # due=6000
            batch = state.fetch_new_batch(5)
            state.save_pipeline_results(
                cues=batch,
                corrected_texts={cue.source_key: "纠错"},
                translated_texts={cue.source_key: "翻译"},
                translated_at_mono_ms=2000,
                fallback_mode="jp_raw",
                llm_latency_ms=100,
            )
            row = state.fetch_cue_by_key(cue.source_key)
            assert row is not None

            # Effective due becomes 5000 after -1000ms adjustment, so at now=5500 it is already past due.
            deleted = state.soft_delete_cue(
                source_key=cue.source_key,
                now_mono_ms=5500,
                delay_adjust_ms=-1000,
                expected_revision=row.revision,
            )
            state.close()

            self.assertEqual(deleted.get("ok"), False)
            self.assertEqual(deleted.get("code"), "past_due")

    def test_window_uses_asymmetric_effective_due(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "state.sqlite3"
            state = StateStore(db)
            cues = [
                SourceCue("srt:1:0:1000", "srt", 1, 0, 1000, "a"),  # seen 1000 -> due 6000
                SourceCue("srt:2:0:1000", "srt", 2, 0, 1000, "b"),  # seen 3000 -> due 8000
            ]
            state.upsert_source_cues([cues[0]], now_mono_ms=1000, delay_ms=5000)
            state.upsert_source_cues([cues[1]], now_mono_ms=3000, delay_ms=5000)

            # now=9000, D_eff window past=2000 future=6000 means [7000, 15000].
            # with adjust=-500ms, due_effective are 5500 and 7500 -> only second remains in window.
            rows = state.fetch_cues_window(
                now_mono_ms=9000,
                past_window_ms=2000,
                future_window_ms=6000,
                delay_adjust_ms=-500,
                limit=20,
            )
            state.close()

            self.assertEqual([r.source_key for r in rows], ["srt:2:0:1000"])

    def test_mark_due_unshown_as_displayed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "state.sqlite3"
            state = StateStore(db)
            cues = [
                SourceCue("srt:1:0:1000", "srt", 1, 0, 1000, "a"),
                SourceCue("srt:2:0:1000", "srt", 2, 0, 1000, "b"),
            ]
            state.upsert_source_cues([cues[0]], now_mono_ms=1000, delay_ms=1000)  # due 2000
            state.upsert_source_cues([cues[1]], now_mono_ms=3000, delay_ms=1000)  # due 4000
            batch = state.fetch_new_batch(10)
            state.save_pipeline_results(
                cues=batch,
                corrected_texts={c.source_key: c.jp_raw for c in batch},
                translated_texts={c.source_key: c.jp_raw for c in batch},
                translated_at_mono_ms=1500,
                fallback_mode="jp_raw",
                llm_latency_ms=100,
            )
            changed = state.mark_due_unshown_as_displayed(now_mono_ms=3500, delay_adjust_ms=0)
            r1 = state.fetch_cue_by_key(cues[0].source_key)
            r2 = state.fetch_cue_by_key(cues[1].source_key)
            state.close()

            self.assertEqual(changed, 1)
            assert r1 is not None and r2 is not None
            self.assertIsNotNone(r1.displayed_at_mono_ms)
            self.assertIsNone(r2.displayed_at_mono_ms)

    def test_recover_inflight_on_startup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "state.sqlite3"
            state = StateStore(db)
            cue = SourceCue("srt:1:0:1000", "srt", 1, 0, 1000, "a")
            state.upsert_source_cues([cue], now_mono_ms=1000, delay_ms=1000)
            claimed = state.fetch_and_claim_batch(limit=1, owner="main", now_mono_ms=1100)
            self.assertEqual(len(claimed), 1)
            state.close()

            state2 = StateStore(db)
            row = state2.fetch_cue_by_key(cue.source_key)
            stats = state2.fetch_new_queue_stats(now_mono_ms=1200)
            state2.close()

            assert row is not None
            self.assertEqual(row.status, "NEW")
            self.assertIsNone(row.inflight_owner)
            self.assertEqual(stats["inflight_count"], 0)

    def test_fetch_latest_ready_unshown_prefers_latest_translated(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "state.sqlite3"
            state = StateStore(db)
            c1 = SourceCue("srt:1:0:1000", "srt", 1, 0, 1000, "a")
            c2 = SourceCue("srt:2:0:1000", "srt", 2, 0, 1000, "b")
            state.upsert_source_cues([c1], now_mono_ms=1000, delay_ms=5000)
            state.upsert_source_cues([c2], now_mono_ms=2000, delay_ms=5000)
            batch = state.fetch_new_batch(10)
            state.save_pipeline_results(
                cues=batch,
                corrected_texts={c.source_key: c.jp_raw for c in batch},
                translated_texts={c.source_key: c.jp_raw for c in batch},
                translated_at_mono_ms=3000,
                fallback_mode="jp_raw",
                llm_latency_ms=100,
            )
            # Make c1 look older translated, c2 newer translated.
            state._conn.execute("UPDATE cues SET translated_mono_ms=? WHERE source_key=?", (2500, c1.source_key))  # type: ignore[attr-defined]
            state._conn.execute("UPDATE cues SET translated_mono_ms=? WHERE source_key=?", (3500, c2.source_key))  # type: ignore[attr-defined]
            state._conn.commit()  # type: ignore[attr-defined]
            latest = state.fetch_latest_ready_unshown(delay_adjust_ms=0)
            state.close()

            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest["source_key"], c2.source_key)

    def test_cleanup_runtime_scope_archives_non_active_source_kind(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "state.sqlite3"
            state = StateStore(db)
            srt = SourceCue("srt:1:0:1000", "srt", 1, 0, 1000, "a")
            speech = SourceCue("speechmatics:1:1000", "speechmatics", None, None, None, "b")
            state.upsert_source_cues([srt], now_mono_ms=1000, delay_ms=1000)
            state.upsert_source_cues([speech], now_mono_ms=1000, delay_ms=1000)
            res = state.cleanup_runtime_scope(
                now_mono_ms=2000,
                allowed_source_kinds=["speechmatics"],
                stale_unfinished_sec=3600.0,
                monotonic_guard_sec=600.0,
                anomaly_remaining_max_sec=300.0,
            )
            srt_row = state.fetch_cue_by_key(srt.source_key)
            speech_row = state.fetch_cue_by_key(speech.source_key)
            state.close()

            self.assertGreaterEqual(int(res.get("excluded_source_count", 0)), 1)
            assert srt_row is not None and speech_row is not None
            self.assertEqual(srt_row.deleted_soft, True)
            self.assertEqual(speech_row.deleted_soft, False)

    def test_fetch_latest_ready_unshown_respects_scope_and_freshness(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "state.sqlite3"
            state = StateStore(db)
            old = SourceCue("speechmatics:old:1000", "speechmatics", None, None, None, "old")
            new = SourceCue("speechmatics:new:1001", "speechmatics", None, None, None, "new")
            state.upsert_source_cues([old], now_mono_ms=1000, delay_ms=1000)
            state.upsert_source_cues([new], now_mono_ms=2000, delay_ms=1000)
            batch = state.fetch_new_batch(10)
            state.save_pipeline_results(
                cues=batch,
                corrected_texts={c.source_key: c.jp_raw for c in batch},
                translated_texts={c.source_key: c.jp_raw for c in batch},
                translated_at_mono_ms=3000,
                fallback_mode="jp_raw",
                llm_latency_ms=100,
            )
            # Simulate stale "old" row from long-past session.
            state._conn.execute("UPDATE cues SET updated_at=? WHERE source_key=?", (1.0, old.source_key))  # type: ignore[attr-defined]
            state._conn.commit()  # type: ignore[attr-defined]
            latest = state.fetch_latest_ready_unshown(
                delay_adjust_ms=0,
                allowed_source_kinds=["speechmatics"],
                updated_after_unix=1000.0,
            )
            state.close()

            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest["source_key"], new.source_key)


if __name__ == "__main__":
    unittest.main()
