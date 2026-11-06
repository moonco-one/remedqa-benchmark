import importlib.util
import io
import json
import os
import tempfile
import unittest
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("remedqa_eval", Path(__file__).parents[1] / "remedqa_eval.py")
eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval)


def reply(content="(A)"):
    return {"object": "chat.completion", "id": "response-1", "model": "public-model",
            "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}


def items_per_task(count=2):
    items = []
    for task in eval.TASKS:
        subset = task if task in {"medqa", "medmcqa"} else "mmlu"
        for index in range(count):
            for mode in eval.MODES:
                items.append({"task": task, "subset": subset, "id": f"{task}-{index:03}",
                              "mode": mode, "prompt": f"Question {task} {index} {mode}", "gold": "A"})
    return items


def settings(items, **overrides):
    values = {"base_url": "https://example.com/v1", "model": "comparison-model", "percent": Decimal(100),
              "seed": 20260723, "concurrency": 1}
    values.update(overrides)
    return eval.manifest(items, **values)


def record(item, content="(A)"):
    return {"key": eval.key(item), "at": "2026-11-06T00:00:00+00:00",
            "prompt_sha256": eval.digest(item["prompt"].encode()), "response": eval.check_response(reply(content))}


class ResponseTests(unittest.TestCase):
    def test_other_provider_does_not_need_moonco_cost(self):
        result = eval.check_response(reply())
        self.assertEqual(result["usage"], {})
        self.assertEqual(result["choices"][0]["message"]["content"], "(A)")

    def test_only_public_final_fields_are_saved(self):
        data = reply()
        data["provider"] = "private-provider"
        data["choices"][0]["message"]["reasoning_content"] = "private draft"
        data["usage"] = {"total_tokens": 3, "billable_cost_microusd": 2, "provider_cost": 7}
        clean = eval.check_response(data)
        text = json.dumps(clean)
        self.assertNotIn("private", text)
        self.assertNotIn("provider", text)
        self.assertEqual(clean["usage"], {"total_tokens": 3, "billable_cost_microusd": 2})

    def test_missing_answer_stays_as_an_empty_final_answer(self):
        data = reply()
        del data["choices"][0]["message"]["content"]
        self.assertIsNone(eval.check_response(data)["choices"][0]["message"]["content"])

    def test_invalid_usage_and_multiple_choices_are_rejected(self):
        for bad in (True, -1, "4"):
            data = reply(); data["usage"] = {"total_tokens": bad}
            with self.assertRaises(ValueError): eval.check_response(data)
        data = reply(); data["choices"] *= 2
        with self.assertRaises(ValueError): eval.check_response(data)

    def test_base_url_rules(self):
        self.assertEqual(eval.api_url("https://example.com/v1/"), "https://example.com/v1/chat/completions")
        self.assertEqual(eval.api_url("http://127.0.0.1:9999/v1"), "http://127.0.0.1:9999/v1/chat/completions")
        for url in ("http://example.com/v1", "https://user:secret@example.com/v1", "https://example.com/v1?key=x",
                    "https://example.com/v1#fragment", "file:///tmp/api"):
            with self.assertRaises(ValueError): eval.api_url(url)

    def test_request_uses_configured_endpoint_model_and_never_logs_key(self):
        item = items_per_task(1)[0]
        config = settings([item], base_url="https://other.example/api/v1", model="other-model")
        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return json.dumps(reply()).encode()
        class Opener:
            def open(self, call, timeout):
                self.call = call
                return FakeResponse()
        opener = Opener()
        with tempfile.TemporaryDirectory() as tmp, patch.object(eval.urllib.request, "build_opener", return_value=opener), patch.object(eval, "wait_for_rate_limit"):
            log = Path(tmp) / "attempts.jsonl"
            eval.request(item, "test-key-must-not-leak", log, config)
            self.assertEqual(opener.call.full_url, "https://other.example/api/v1/chat/completions")
            self.assertEqual(json.loads(opener.call.data)["model"], "other-model")
            self.assertNotIn("test-key-must-not-leak", log.read_text())
            self.assertNotIn("test-key-must-not-leak", json.dumps(config))


class SelectionTests(unittest.TestCase):
    def test_percentage_is_question_based_nested_and_order_independent(self):
        items = items_per_task(21)
        small = eval.select_items(items, 5)
        medium = eval.select_items(list(reversed(items)), 20)
        self.assertEqual(len(small), 8 * 2 * 6)  # ceil(21 * .05) for each task.
        self.assertEqual(len(medium), 8 * 5 * 6)
        self.assertTrue({eval.key(item) for item in small} <= {eval.key(item) for item in medium})
        self.assertEqual(small, eval.select_items(list(reversed(items)), 5))
        groups = defaultdict(set)
        for item in small: groups[(item["task"], item["id"])].add(item["mode"])
        self.assertTrue(all(modes == set(eval.MODES) for modes in groups.values()))

    def test_invalid_percent_seed_and_partial_modes_fail(self):
        items = items_per_task(1)
        for percent in (0, -1, 101, "NaN", "Infinity"):
            with self.assertRaises(ValueError): eval.select_items(items, percent)
        with self.assertRaises(ValueError): eval.select_items(items, 100, -1)
        with self.assertRaises(ValueError): eval.select_items(items[:-1])

    def test_smoke_is_three_questions_and_eighteen_requests(self):
        selected = eval.select_items(items_per_task(2), smoke=True)
        self.assertEqual(len(selected), 18)
        self.assertEqual({item["subset"] for item in selected}, set(eval.DATA_SHA256))


class ScoreTests(unittest.TestCase):
    def test_macro_weights_do_not_pool_questions(self):
        cells = {}
        for mode in eval.MODES:
            for task in eval.TASKS:
                cells[(mode, task)] = [100, 100] if task == "medqa" else [0, 1]
        self.assertEqual(eval.closed_avg(cells), Decimal("33.333333333333"))

    def test_bootstrap_keeps_all_modes_of_a_question_together(self):
        groups = {}
        for task in eval.TASKS:
            for index in range(2):
                correct = 0 if task == "medqa" and index == 0 else 1
                groups[(task, task, str(index))] = dict.fromkeys(eval.MODES, correct)
        self.assertEqual(eval.confidence_interval(groups, draws=400), ["66.666666666667", "100.000000000000"])

    def test_incomplete_run_has_no_benchmark_score(self):
        items = items_per_task(1)
        records = {eval.key(items[0]): record(items[0])}
        with tempfile.TemporaryDirectory() as tmp, patch.object(eval, "strict_parser", side_effect=AssertionError("must not score")):
            report = eval.write_reports(items, records, Path("unused"), Path(tmp), False)
            self.assertFalse(report["complete"])
            self.assertNotIn("strict_closed_avg_percent", report)
            self.assertIn("No benchmark score", (Path(tmp) / "report.md").read_text())

    def test_strict_answer_rules_and_empty_answers(self):
        items = items_per_task(1)
        records = {eval.key(item): record(item, None) for item in items}
        with patch.object(eval, "strict_parser", return_value=lambda content, mode: None):
            cells, groups, verdicts, summary = eval.inspect(items, records, Path("unused"))
        self.assertEqual(eval.closed_avg(cells), 0)
        self.assertEqual(summary["empty_answers"], len(items))
        self.assertEqual(summary["invalid_answers"], len(items))
        self.assertTrue(all(not verdict["correct"] for verdict in verdicts))

    def test_smoke_never_reports_a_benchmark_score(self):
        items = eval.select_items(items_per_task(1), smoke=True)
        records = {eval.key(item): record(item) for item in items}
        with tempfile.TemporaryDirectory() as tmp, patch.object(eval, "strict_parser", return_value=lambda content, mode: "A"):
            report = eval.write_reports(items, records, Path("unused"), Path(tmp), True)
            self.assertTrue(report["complete"])
            self.assertNotIn("strict_closed_avg_percent", report)
            self.assertNotIn("confidence_interval_95_percent", report)


class ResumeTests(unittest.TestCase):
    def test_changed_model_data_code_and_unknown_response_keys_fail(self):
        items = items_per_task(1)
        config = settings(items)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            eval.prepare_run(Path("unused"), out, config)
            altered = dict(config, model="changed")
            with self.assertRaises(ValueError): eval.prepare_run(Path("unused"), out, altered)
            altered = dict(config, script_sha256="changed")
            with self.assertRaises(ValueError): eval.prepare_run(Path("unused"), out, altered)
        bad = record(items[0]); bad["response"]["extra"] = "private"
        with self.assertRaises(ValueError): eval.validate_records(items, {eval.key(items[0]): bad})
        bad = record(items[0]); bad["prompt_sha256"] = "changed"
        with self.assertRaises(ValueError): eval.validate_records(items, {eval.key(items[0]): bad})
        with self.assertRaises(ValueError): eval.validate_records(items, {"other": record(items[0])})

    def test_resume_does_not_call_completed_items(self):
        items = items_per_task(1)[:2]
        config = settings(items)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"TEST_EVAL_KEY": "key"}), patch.object(eval, "write_reports", return_value={"complete": True}) as report:
            out = Path(tmp)
            eval.prepare_run(Path("unused"), out, config)
            eval.append_jsonl(out / "responses.jsonl", record(items[0]))
            with patch.object(eval, "request", return_value=eval.check_response(reply())) as call:
                eval.run(items, Path("unused"), out, config, "TEST_EVAL_KEY")
            self.assertEqual(call.call_count, 1)
            self.assertEqual(call.call_args.args[0], items[1])
            self.assertEqual(len(eval.saved_responses(out / "responses.jsonl")), 2)

    def test_failure_keeps_completed_replies_and_no_new_calls_are_started(self):
        items = items_per_task(1)[:3]
        config = settings(items)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"TEST_EVAL_KEY": "key"}), patch.object(eval, "write_reports"):
            out = Path(tmp)
            with patch.object(eval, "request", side_effect=[eval.check_response(reply()), RuntimeError("test failure")]) as call:
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    eval.run(items, Path("unused"), out, config, "TEST_EVAL_KEY")
            self.assertEqual(call.call_count, 2)
            self.assertEqual(len(eval.saved_responses(out / "responses.jsonl")), 1)

    def test_partial_and_duplicate_ledger_lines_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "responses.jsonl"
            path.write_text('{"key":"one"}')
            with self.assertRaises(ValueError): eval.saved_responses(path)
            path.write_text('{"key":"one"}\n{"key":"one"}\n')
            with self.assertRaises(ValueError): eval.saved_responses(path)

    def test_offline_score_does_not_read_credential_or_call_api(self):
        items = eval.select_items(items_per_task(1))
        config = settings(items)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); eval.write_json(out / "manifest.json", config)
            with patch.object(eval, "load_items", return_value=items), patch.object(eval, "write_reports", return_value={"complete": False}), patch.object(eval.os, "getenv", side_effect=AssertionError("no credentials")), patch.object(eval, "request", side_effect=AssertionError("no API")):
                self.assertEqual(eval.score(Path("unused"), out), {"complete": False})


class CliTests(unittest.TestCase):
    def test_api_commands_require_an_explicit_target_before_loading_inputs(self):
        for command in ("plan", "run", "smoke"):
            targets = ([], ["--base-url", "https://other.example/v1"], ["--model", "other-model"])
            for target in targets:
                with (
                    self.subTest(command=command, target=target),
                    patch.object(eval, "load_items", return_value=items_per_task(1)) as load,
                    patch.object(eval, "run", return_value={"complete": True}) as run,
                    patch("sys.stdout", new_callable=io.StringIO),
                    patch("sys.stderr", new_callable=io.StringIO),
                ):
                    with self.assertRaises(SystemExit) as error:
                        eval.main([command, *target])
                    self.assertEqual(error.exception.code, 2)
                    load.assert_not_called()
                    run.assert_not_called()

    def test_api_commands_use_the_explicit_endpoint_and_model(self):
        for command in ("plan", "run", "smoke"):
            with (
                self.subTest(command=command),
                patch.object(eval, "load_items", return_value=items_per_task(1)),
                patch.object(eval, "run", return_value={"complete": True}) as run,
                patch("sys.stdout", new_callable=io.StringIO) as output,
            ):
                result = eval.main([
                    command, "--base-url", "https://other.example/v1",
                    "--model", "other-model", "--api-key-env", "OTHER_API_KEY",
                ])
                self.assertEqual(result, 0)
                if command == "plan":
                    config = json.loads(output.getvalue())
                    run.assert_not_called()
                else:
                    config = run.call_args.args[3]
                    self.assertEqual(run.call_args.args[4], "OTHER_API_KEY")
                self.assertEqual(config["api_url"], "https://other.example/v1/chat/completions")
                self.assertEqual(config["model"], "other-model")

    def test_score_uses_the_saved_run_without_target_flags(self):
        with (
            patch.object(eval, "score", return_value={"complete": True}) as score,
            patch.object(eval, "load_items") as load,
            patch.object(eval, "run") as run,
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            result = eval.main(["score", "--repo", "upstream", "--out", "results/example"])
        self.assertEqual(result, 0)
        score.assert_called_once_with(Path("upstream"), Path("results/example"))
        load.assert_not_called()
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
