# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Non-vacuous full-vocabulary quality comparisons, without loading a model."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quant_quality as quality  # noqa: E402


class PromptComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ref, self.run = self.root / "ref", self.root / "candidate"
        self.index = {"requests": [{"segment": "code", "tokens": 256}],
                      "ids_sha256": hashlib.sha256(b"fixed ids").hexdigest()}
        for p in (self.ref, self.run):
            p.mkdir()
            self.write_index(p, self.index)
            np.array([2.0, 1.0, 0.0], dtype=np.float32).tofile(p / "first.0")

    @staticmethod
    def write_index(path, index):
        (path / "index.json").write_text(json.dumps(index))

    def compare(self, max_kl=None, min_top1=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return quality.cmd_pcompare(SimpleNamespace(ref=str(self.ref), runs=[str(self.run)],
                                                        max_mean_kl=max_kl, min_top1=min_top1))

    def test_identical_full_vocabulary_passes(self):
        self.assertEqual(self.compare(0.0, 1.0), 0)

    def test_missing_candidate_logits_is_not_zero_kl(self):
        (self.run / "first.0").unlink()
        with self.assertRaises(FileNotFoundError):
            self.compare()

    def test_missing_reference_logits_is_refused(self):
        (self.ref / "first.0").unlink()
        with self.assertRaises(FileNotFoundError):
            self.compare()

    def test_empty_request_set_is_refused(self):
        self.write_index(self.ref, {**self.index, "requests": []})
        with self.assertRaisesRegex(ValueError, "no prompt requests"):
            self.compare()

    def test_different_requests_are_refused(self):
        self.write_index(self.run, {**self.index, "requests": [{"segment": "docs", "tokens": 256}]})
        with self.assertRaisesRegex(ValueError, "identity differ"):
            self.compare()

    def test_different_or_missing_ids_hash_is_refused(self):
        for h in (None, hashlib.sha256(b"different ids").hexdigest()):
            with self.subTest(hash=h):
                self.write_index(self.run, {**self.index, "ids_sha256": h})
                with self.assertRaisesRegex(ValueError, "identity differ"):
                    self.compare()

    def test_legacy_reference_requires_regeneration(self):
        self.write_index(self.ref, {"requests": self.index["requests"]})
        with self.assertRaisesRegex(ValueError, "missing fixed-input"):
            self.compare()

    def test_empty_truncated_or_nonfinite_logits_are_refused(self):
        for raw in (b"", b"x", np.array([np.nan], dtype=np.float32).tobytes(),
                    np.array([np.inf], dtype=np.float32).tobytes()):
            with self.subTest(raw=raw):
                (self.run / "first.0").write_bytes(raw)
                with self.assertRaisesRegex(ValueError, "logits"):
                    self.compare()

    def test_vocabulary_mismatch_is_not_broadcast(self):
        np.array([2.0], dtype=np.float32).tofile(self.run / "first.0")
        with self.assertRaisesRegex(ValueError, "vocabulary differs"):
            self.compare()

    def test_every_request_is_required(self):
        for p in (self.ref, self.run):
            self.write_index(p, {**self.index, "requests": self.index["requests"] * 2})
        with self.assertRaises(FileNotFoundError):
            self.compare()

    def test_mean_full_vocabulary_kl_gate_rejects(self):
        np.array([0.0, 1.0, 2.0], dtype=np.float32).tofile(self.run / "first.0")
        self.assertEqual(self.compare(0.03), 1)
        self.assertEqual(self.compare(10), 0)

    def test_next_token_agreement_gate_rejects(self):
        np.array([0.0, 1.0, 2.0], dtype=np.float32).tofile(self.run / "first.0")
        self.assertEqual(self.compare(min_top1=1), 1)
        self.assertEqual(self.compare(min_top1=0), 0)

    def test_invalid_thresholds_are_refused(self):
        for kl, top1 in ((-1, None), (float("nan"), None), (None, 2), (None, float("inf"))):
            with self.subTest(kl=kl, top1=top1):
                with self.assertRaises(ValueError):
                    self.compare(kl, top1)

    def test_prompt_identity_hashes_consumed_snapshot(self):
        ids = self.root / "ids.json"
        source = json.dumps({"seg_tokens": 1000, "segments": [{"name": "code", "ids": list(range(1000))}]})
        ids.write_text(source)
        config = self.root / "config.json"
        config.write_text(json.dumps({"exe": "never-executed", "args": []}))
        out = self.root / "prompt-out"
        fake = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO("READY\n" + "DONE\n" * 4),
                               wait=lambda timeout=None: 0)

        def start(*args, **kwargs):
            ids.write_text("changed after input was consumed")
            return fake

        args = SimpleNamespace(config=str(config), ids=str(ids), out=str(out), env=[],
                               gpu_experts=False, min_free_gb=0, wrap=None)
        with patch.object(quality.subprocess, "Popen", side_effect=start), \
                patch.object(quality, "mem_available_gb", return_value=1), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(quality.cmd_prompt(args), 0)
        index = json.loads((out / "index.json").read_text())
        self.assertEqual(index["ids_sha256"], hashlib.sha256(source.encode()).hexdigest())

    def test_full_distribution_not_topk_renormalization(self):
        # Same top-1 but a changed low-probability tail must still contribute to KL.
        np.array([2.0, 1.0, -8.0], dtype=np.float32).tofile(self.run / "first.0")
        self.assertEqual(self.compare(0.03, 1), 1)


if __name__ == "__main__":
    unittest.main()
