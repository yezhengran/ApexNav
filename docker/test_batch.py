#!/usr/bin/env python3
"""Batch accounting/video tests; never start Habitat navigation or a planner."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basic_utils.record_episode.experiment import (
    EpisodeVideo, episode_plan, read_rows, resume_state, save_result,
    record_episode_identity,
)
from run_batch import evaluation_command, stop_owned
from compare_runs import compare


class BatchTest(unittest.TestCase):
    def row(self, number):
        return dict(episode_number=number, scene_id="scene-a", episode_id=str(number - 1),
                    target="chair", steps=10, success=number % 2, spl=0.123456789,
                    soft_spl=0.75, distance_to_goal=0.1, distance_to_goal_reward=0.2,
                    result="success" if number % 2 else "failure", elapsed_seconds=1.0,
                    cumulative_seconds=float(number), video_path="")

    def test_exact_first_50_and_resume_without_duplicates(self):
        target, skip, remaining = episode_plan(1000, limit=50)
        self.assertEqual((target, skip, remaining), (50, 0, 50))
        iterator = iter(range(1000))
        self.assertEqual([next(iterator) for _ in range(remaining)], list(range(50)))
        target, skip, remaining = episode_plan(1000, completed=7, limit=50)
        iterator = iter(range(1000))
        for _ in range(skip):
            next(iterator)
        self.assertEqual([next(iterator) for _ in range(remaining)], list(range(7, 50)))
        self.assertEqual(episode_plan(1000, completed=50, limit=50), (50, 50, 0))

    def test_single_index_is_not_a_count(self):
        self.assertEqual(episode_plan(1000, single=50), (1, 50, 1))
        with self.assertRaises(ValueError):
            episode_plan(1000, single=1000)
        with self.assertRaises(ValueError):
            episode_plan(1000, limit=0)

    def test_incremental_csv_and_exact_resume_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(resume_state(directory))
            for number in range(1, 51):
                save_result(directory, self.row(number), 50)
            self.assertEqual(len(read_rows(directory)), 50)
            state = resume_state(directory)
            self.assertEqual(state[:2], (50, 25))
            self.assertAlmostEqual(state[2], 50 * 0.123456789)
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["success_rate"], 0.5)
            self.assertEqual(summary["mean_steps"], 10)
            self.assertEqual(summary["result_counts"], {"success": 25, "failure": 25})
            self.assertEqual(len(list(Path(directory).glob("checkpoint-*.json"))), 5)
            self.assertEqual((Path(directory) / "summary.json").stat().st_mode & 0o777, 0o644)
            with self.assertRaises(ValueError):
                save_result(directory, self.row(50), 50)

    def test_command_caps_count_and_disables_ollama(self):
        command = evaluation_command("hm3dv2", 50, Path("/tmp/experiment"), True)
        self.assertIn("+max_episodes=50", command)
        self.assertIn("test_epi_num=-1", command)
        self.assertIn("llm.llm_client.llm_client=none", command)
        self.assertIn("need_video=true", command)
        self.assertFalse(any("max_episode_steps" in item for item in command))
        diagnostic = evaluation_command("hm3dv2", 1, Path("/tmp/diagnostic"), True, max_steps=3)
        self.assertIn("habitat.environment.max_episode_steps=3", diagnostic)
        adaptive = evaluation_command("hm3dv2", 50, Path("/tmp/a"), False,
                                      method="selected_review", seed=100, confirm_threshold=0.55,
                                      reject_threshold=0.10, uncertain_after_steps=400)
        self.assertIn("active_view.mode=selected_review", adaptive)
        self.assertIn("active_view.itm_confirm_threshold=0.55", adaptive)
        self.assertIn("active_view.itm_reject_threshold=0.1", adaptive)
        self.assertIn("active_view.uncertain_after_steps=400", adaptive)
        self.assertIn("habitat.seed=100", adaptive)
        self.assertFalse(any("active_view" in item for item in command))

    def test_episode_manifest_rejects_changed_order_and_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            record_episode_identity(directory, 1, "a", "0", "chair")
            record_episode_identity(directory, 1, "a", "0", "chair")
            record_episode_identity(directory, 2, "a", "1", "bed")
            with self.assertRaises(ValueError):
                record_episode_identity(directory, 2, "b", "1", "bed")
            with self.assertRaises(ValueError):
                record_episode_identity(directory, 4, "a", "3", "bed")

    def test_stop_only_owned_worker(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   start_new_session=True)
        try:
            stop_owned(process)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_comparison_matches_prefix_and_rejects_mismatched_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("baseline", "adaptive")]
            for path, method in zip(paths, ("baseline", "adaptive_view")):
                path.mkdir()
                (path / "experiment.json").write_text(json.dumps({
                    "method": method, "episodes": 50, "seed": 100, "dataset": "hm3dv2",
                    "video": False, "llm_mode": "cached"}))
                (path / "provenance.json").write_text(json.dumps({
                    "code_sha": "example", "source_sha256": "same-source"}))
                save_result(path, self.row(1), 50)
            save_result(paths[0], self.row(2), 50)
            report = compare(*paths)
            self.assertEqual(report["paired_episodes"], 1)
            self.assertFalse(report["complete"])
            self.assertEqual(report["delta_treatment_minus_baseline"]["spl"], 0)
            csv_path = paths[1] / "episodes.csv"
            csv_path.write_text(csv_path.read_text().replace("scene-a", "scene-b"))
            with self.assertRaises(ValueError):
                compare(*paths)

    def test_streamed_video_is_readable_and_has_no_frame_list(self):
        import imageio.v2 as imageio
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            video = EpisodeVideo(directory, 1)
            for value in (0, 100, 200):
                video.append(np.full((480, 1280, 3), value, dtype=np.uint8))
            destination = video.finish(Path(directory) / "success", "example")
            with imageio.get_reader(destination) as reader:
                self.assertEqual(reader.count_frames(), 3)
                self.assertEqual(reader.get_data(0).shape[1], 960)
            self.assertFalse(video.temporary.exists())
            self.assertFalse(any(isinstance(v, list) for v in vars(video).values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
