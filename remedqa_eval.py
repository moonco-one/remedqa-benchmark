"""Evaluate the pinned ReMedQA closed modes through a compatible API. Python 3.10+."""

import argparse
import hashlib
import ipaddress
import json
import math
import os
import random
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_EVEN
from email.utils import parsedate_to_datetime
from pathlib import Path

COMMIT = "2caa66392f5d7c9632d8c24d6a9391df79c6555a"
DATA_SHA256 = {
    "medqa": "cbcdb0143571e873d4764e5c9d0ee4928b50c331ba732462a84c88ef479643c4",
    "medmcqa": "55ac3efe57bb278e6a7a16f618e0f593017935bc2df6613a3abf069db4d14732",
    "mmlu": "41a0c4aaee0ac9c2322c31ca5f2cca80fc4d78244cee7fa0a4f95ec88fcf9622",
}
PARSER_SHA256 = "cefcec012dbba372dd698e69f31fdcad48137e872c78c3ff6bc7c23ef24a2225"
MODES = ("mcq", "incorrect", "roman_numeral", "none_of_the_provided", "fixed_pos", "no_symbols")
MMLU = ("anatomy", "clinical_knowledge", "college_biology", "college_medicine",
        "medical_genetics", "professional_medicine")
TASKS = ("medqa", "medmcqa", *MMLU)
ALGORITHM = "remedqa-stratified-sha256-v1"
MAX_TOKENS, TIMEOUT, MAX_ATTEMPTS, MIN_INTERVAL = 8192, 900, 3, 2.0
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 10_000, 42
LOG_LOCK, START_LOCK = threading.Lock(), threading.Lock()
NEXT_START = 0.0


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def now():
    return datetime.now(timezone.utc).isoformat()


def key(item):
    return f"{item['subset']}|{item['mode']}|{item['id']}"


def api_url(base_url):
    parts = urllib.parse.urlsplit(base_url)
    try:
        loopback = parts.hostname == "localhost" or ipaddress.ip_address(parts.hostname).is_loopback
    except ValueError:
        loopback = False
    if (not parts.hostname or parts.username is not None or parts.password is not None
            or parts.query or parts.fragment
            or parts.scheme not in ({"http", "https"} if loopback else {"https"})):
        raise ValueError("Use an HTTPS base URL without credentials, query, or fragment; HTTP is allowed on loopback only")
    _ = parts.port  # Reject an invalid port before a run starts.
    return base_url.rstrip("/") + "/chat/completions"


def load_items(repo):
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if revision != COMMIT:
        raise ValueError(f"ReMedQA must be at commit {COMMIT}")
    if digest((repo / "src/utils/parser.py").read_bytes()) != PARSER_SHA256:
        raise ValueError("ReMedQA parser differs from the pinned version")
    items = []
    for subset, expected_hash in DATA_SHA256.items():
        raw = (repo / "data/bench" / f"{subset}_all_think.json").read_bytes()
        if digest(raw) != expected_hash:
            raise ValueError(f"ReMedQA {subset} data differs from the pinned version")
        data = json.loads(raw)
        identities = set(data["mcq"])
        if any(set(data[mode]) != identities for mode in MODES):
            raise ValueError("ReMedQA question modes are incomplete")
        for question_id in sorted(identities):
            task = subset if subset != "mmlu" else question_id.rsplit("-", 1)[0]
            for mode in MODES:
                row = data[mode][question_id]
                items.append({"subset": subset, "task": task, "mode": mode,
                              "id": question_id, "prompt": row["prompt"], "gold": row["answer"]})
    if len(items) != 18_924 or {item["task"] for item in items} != set(TASKS):
        raise ValueError("ReMedQA does not contain the expected closed questions")
    return items


def select_items(items, percent=Decimal(100), seed=20260723, smoke=False):
    percent = Decimal(str(percent))
    if not percent.is_finite() or not 0 < percent <= 100 or seed < 0:
        raise ValueError("Percent must be greater than zero and at most 100; seed must be non-negative")
    questions = defaultdict(dict)
    for item in items:
        questions[item["task"]][item["id"]] = item["subset"]
    if smoke:
        first_id = {subset: min(row["id"] for row in items if row["subset"] == subset)
                    for subset in DATA_SHA256}
        selected = {(item["task"], item["id"]) for item in items
                    if item["id"] == first_id[item["subset"]]}
    else:
        selected = set()
        for task, identities in sorted(questions.items()):
            ranked = sorted(identities, key=lambda question_id: (digest(canonical({
                "algorithm": ALGORITHM, "seed": seed, "task": task, "id_question": question_id,
            })), question_id))
            count = int((Decimal(len(ranked)) * percent / 100).to_integral_value(rounding=ROUND_CEILING))
            selected.update((task, question_id) for question_id in ranked[:count])
    result = sorted((item for item in items if (item["task"], item["id"]) in selected),
                    key=lambda item: (item["task"], item["subset"], item["id"], MODES.index(item["mode"])))
    grouped = defaultdict(list)
    for item in result:
        grouped[(item["task"], item["subset"], item["id"])].append(item["mode"])
    if any(tuple(modes) != MODES for modes in grouped.values()) or (smoke and len(result) != 18):
        raise ValueError("Each selected question must have all six closed modes")
    return result


def manifest(items, *, base_url, model, percent, seed, concurrency, smoke=False):
    if not model.strip() or concurrency < 1:
        raise ValueError("Model must be non-empty and concurrency must be positive")
    identities = sorted({(item["task"], item["subset"], item["id"]) for item in items})
    return {
        "schema_version": 1, "upstream_commit": COMMIT, "data_sha256": DATA_SHA256,
        "parser_sha256": PARSER_SHA256, "script_sha256": digest(Path(__file__).read_bytes()),
        "selection": "smoke" if smoke else ALGORITHM, "percent": format(Decimal(str(percent)).normalize(), "f"),
        "seed": seed, "questions": len(identities), "requests": len(items),
        "question_ids": [{"task": task, "subset": subset, "id": question_id}
                         for task, subset, question_id in identities],
        "input_sha256": digest(canonical(items)), "selected_keys": [key(item) for item in items],
        "api_url": api_url(base_url), "model": model, "max_completion_tokens": MAX_TOKENS,
        "timeout_seconds": TIMEOUT, "max_attempts": MAX_ATTEMPTS,
        "minimum_start_interval_seconds": MIN_INTERVAL, "concurrency": concurrency,
        "scorer": "upstream-frozen-strict", "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path, value):
    with LOG_LOCK, path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def saved_responses(path):
    records = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
            if not line.endswith("\n"):
                raise ValueError("Saved response log has an incomplete line; preserve it before repair")
            record = json.loads(line)
            if record["key"] in records:
                raise ValueError("Saved response log has duplicate keys")
            records[record["key"]] = record
    return records


def check_response(data):
    """Keep public final-answer fields only; provider extensions are not saved."""
    if not isinstance(data, dict) or data.get("object") != "chat.completion":
        raise ValueError("API did not return a chat completion")
    if any(not isinstance(data.get(field), str) or not data[field] for field in ("id", "model")):
        raise ValueError("API response has no ID or model")
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("API response must contain one choice")
    message = choices[0].get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("API response has no assistant message")
    content, finish = message.get("content"), choices[0].get("finish_reason")
    if content is not None and not isinstance(content, str):
        raise ValueError("API response has invalid content")
    if finish is not None and not isinstance(finish, str):
        raise ValueError("API response has invalid finish reason")
    usage = {}
    original = data.get("usage") or {}
    if not isinstance(original, dict):
        raise ValueError("API response has invalid usage")
    for field in ("prompt_tokens", "completion_tokens", "total_tokens", "billable_cost_microusd"):
        if field in original:
            if type(original[field]) is not int or original[field] < 0:
                raise ValueError("API response has invalid usage count")
            usage[field] = original[field]
    return {"object": "chat.completion", "id": data["id"], "model": data["model"],
            "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish}],
            "usage": usage}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def wait_for_rate_limit():
    global NEXT_START
    with START_LOCK:
        time.sleep(max(0.0, NEXT_START - time.monotonic()))
        NEXT_START = time.monotonic() + MIN_INTERVAL


def retry_after(value, attempt):
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
    return 5.0 * attempt


def request(item, api_key, attempts_path, settings):
    body = {"model": settings["model"], "messages": [{"role": "user", "content": item["prompt"]}],
            "max_completion_tokens": settings["max_completion_tokens"], "stream": False}
    http = urllib.request.build_opener(NoRedirect)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        wait_for_rate_limit()
        call = urllib.request.Request(settings["api_url"], json.dumps(body).encode(),
                                      {"Authorization": f"Bearer {api_key}",
                                       "Content-Type": "application/json"}, method="POST")
        try:
            with http.open(call, timeout=TIMEOUT) as response:
                data = check_response(json.loads(response.read()))
            append_jsonl(attempts_path, {"key": key(item), "attempt": attempt, "status": 200})
            return data
        except urllib.error.HTTPError as error:
            status = error.code
            append_jsonl(attempts_path, {"key": key(item), "attempt": attempt, "status": status})
            if status != 429 and not 500 <= status < 600:
                raise RuntimeError(f"HTTP {status}; check the API key and settings") from None
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"HTTP {status} after {MAX_ATTEMPTS} attempts") from None
            time.sleep(max(MIN_INTERVAL, retry_after(error.headers.get("Retry-After"), attempt)))
        except (urllib.error.URLError, TimeoutError):
            append_jsonl(attempts_path, {"key": key(item), "attempt": attempt, "status": "network_error"})
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError("Network error; use the same command to resume") from None
            time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def valid_answer(parsed, mode):
    if mode == "incorrect":
        return (isinstance(parsed, list) and len(parsed) == 3
                and len(set(parsed)) == 3 and set(parsed) <= set("ABCD"))
    if mode == "no_symbols":
        return isinstance(parsed, str) and bool(parsed.strip())
    allowed = {"I", "II", "III", "IV"} if mode == "roman_numeral" else set("ABCD")
    return isinstance(parsed, str) and parsed in allowed


def strict_parser(repo):
    path = repo / "src/utils/parser.py"
    raw = path.read_bytes()
    if digest(raw) != PARSER_SHA256:
        raise ValueError("ReMedQA parser differs from the pinned version")
    namespace = {"__name__": "remedqa_frozen_parser", "__file__": str(path)}
    exec(compile(raw, str(path), "exec"), namespace)
    return namespace["parse_final_answer_think"]


def inspect(items, records, repo):
    parse = strict_parser(repo)
    cells, groups, verdicts = defaultdict(lambda: [0, 0]), defaultdict(dict), []
    models, finishes = Counter(), Counter()
    invalid = empty = cost = cost_count = 0
    for item in items:
        record = records[key(item)]
        if record["prompt_sha256"] != digest(item["prompt"].encode()):
            raise ValueError("Saved prompt hash differs from the pinned input")
        response = check_response(record["response"])
        choice = response["choices"][0]
        content = choice["message"]["content"] or ""
        parsed = parse(content, item["mode"])
        correct = (parsed == item["gold"] if item["mode"] != "no_symbols" else
                   isinstance(parsed, str) and parsed.replace(".", "") == item["gold"].replace(".", ""))
        cells[(item["mode"], item["task"])][0] += int(correct)
        cells[(item["mode"], item["task"])][1] += 1
        groups[(item["task"], item["subset"], item["id"])][item["mode"]] = int(correct)
        valid = valid_answer(parsed, item["mode"])
        invalid += int(not valid)
        empty += int(not content.strip())
        if "billable_cost_microusd" in response["usage"]:
            cost += response["usage"]["billable_cost_microusd"]
            cost_count += 1
        models[response["model"]] += 1
        finishes[str(choice["finish_reason"])] += 1
        verdicts.append({"key": key(item), "task": item["task"], "mode": item["mode"],
                         "parsed_answer": parsed, "valid_answer": valid, "correct": bool(correct)})
    return cells, groups, verdicts, {"invalid_answers": invalid, "empty_answers": empty,
        "reported_cost_usd": str(Decimal(cost) / 1_000_000) if cost_count else None,
        "responses_with_reported_cost": cost_count, "returned_models": dict(models),
        "finish_reasons": dict(finishes)}


def closed_avg(cells):
    scores = []
    for mode in MODES:
        def mean(task):
            correct, total = cells[(mode, task)]
            if not total:
                raise ValueError("ClosedAvg requires every task and mode")
            return Decimal(correct) / total
        mmlu = sum((mean(task) for task in MMLU), Decimal(0)) / 6
        scores.append((mean("medqa") + mean("medmcqa") + mmlu) / 3)
    return (sum(scores, Decimal(0)) / 6 * 100).quantize(Decimal("0.000000000001"), rounding=ROUND_HALF_EVEN)


def confidence_interval(groups, draws=BOOTSTRAP_DRAWS):
    # The score is linear: resampling each six-mode vector is equivalent to
    # resampling its count of correct answers. Keep task/question order fixed.
    by_task = defaultdict(list)
    for (task, subset, question_id), values in sorted(groups.items()):
        if set(values) != set(MODES):
            raise ValueError("Confidence interval requires all six modes per question")
        by_task[task].append(sum(values.values()))
    if set(by_task) != set(TASKS):
        raise ValueError("Confidence interval requires every task")
    rng, estimates = random.Random(BOOTSTRAP_SEED), []
    for _ in range(draws):
        rates = {}
        for task in TASKS:
            values = by_task[task]
            correct = sum(values[rng.randrange(len(values))] for _ in values)
            rates[task] = Decimal(correct) / (len(values) * 6)
        mmlu = sum((rates[task] for task in MMLU), Decimal(0)) / 6
        points = (rates["medqa"] + rates["medmcqa"] + mmlu) / 3 * 100
        estimates.append(points.quantize(Decimal("0.000000000001"), rounding=ROUND_HALF_EVEN))
    estimates.sort()
    return [str(estimates[math.floor(Decimal("0.025") * (draws - 1))]),
            str(estimates[math.ceil(Decimal("0.975") * (draws - 1))])]


def validate_records(items, records):
    expected = {key(item): item for item in items}
    if set(records) - set(expected):
        raise ValueError("Output folder has responses from another run")
    for item_key, record in records.items():
        if record["prompt_sha256"] != digest(expected[item_key]["prompt"].encode()):
            raise ValueError("Saved prompt hash differs from the pinned input")
        if check_response(record["response"]) != record["response"]:
            raise ValueError("Saved response has unsupported fields; preserve the original before repair")


def write_reports(items, records, repo, out, smoke):
    validate_records(items, records)
    complete = len(records) == len(items)
    times = sorted(record["at"] for record in records.values())
    result = {"complete": complete, "questions": len(items) // 6, "responses": len(records),
              "expected_responses": len(items), "kind": "connection_check" if smoke else "closed_evaluation",
              "earliest_saved_response_at": times[0] if times else None,
              "latest_saved_response_at": times[-1] if times else None}
    if not complete:
        write_json(out / "report.json", result)
        (out / "report.md").write_text("# Incomplete run\n\nNo benchmark score is available. Resume the same run.\n")
        return result
    cells, groups, verdicts, summary = inspect(items, records, repo)
    result.update(summary)
    result["correct_total_by_cell"] = {f"{mode}|{task}": value for (mode, task), value in sorted(cells.items())}
    if not smoke:
        result["strict_closed_avg_percent"] = str(closed_avg(cells))
        result["confidence_interval_95_percent"] = confidence_interval(groups)
        result["bootstrap"] = {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                               "unit": "question_with_all_six_modes", "stratification": "task"}
    write_json(out / "report.json", result)
    (out / "verdicts.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in verdicts))
    lines = ["# ReMedQA closed result" if not smoke else "# Connection and parser check", "",
             "This is a six-mode closed evaluation, not the paper's open-answer result.", "",
             f"- Questions: {len(items) // 6}", f"- Responses: {len(records)}", f"- Invalid answers: {summary['invalid_answers']}"]
    if smoke:
        lines += ["", "This 18-request connection check is not a benchmark score."]
    else:
        score = Decimal(result["strict_closed_avg_percent"])
        low, high = map(Decimal, result["confidence_interval_95_percent"])
        lines += ["", f"**Strict ClosedAvg: {score:.2f}%**", f"95% confidence interval: {low:.2f}–{high:.2f}%", "",
                  "Each mode gives equal weight to MedQA, MedMCQA, and the mean of six MMLU subjects.",
                  "The final score averages all six modes. Whole questions are resampled within each task.",
                  "Fresh API runs can differ. These intervals do not cover all model-generation variation."]
    lines += ["", "See manifest.json for the exact question set and settings; verdicts.jsonl records each strict verdict.", ""]
    (out / "report.md").write_text("\n".join(lines))
    return result


def prepare_run(repo, out, settings):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != settings:
            raise ValueError("Output folder uses different data, code, or settings; use a new folder")
    elif any(out.iterdir()):
        raise ValueError("Output folder has files but no manifest; use a new folder")
    else:
        write_json(path, settings)
    return saved_responses(out / "responses.jsonl")


def run(items, repo, out, settings, api_key_env):
    records = prepare_run(repo, out, settings)
    validate_records(items, records)
    smoke = settings["selection"] == "smoke"
    remaining = len(items) - len(records)
    print(f"{remaining} of {len(items)} requests left; no key is written to the results")
    if remaining:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Set the {api_key_env} environment variable")
        write_reports(items, records, repo, out, smoke)
        pending_items = iter(item for item in items if key(item) not in records)
        with ThreadPoolExecutor(max_workers=settings["concurrency"]) as pool:
            pending, failure = {}, None
            def submit_next():
                item = next(pending_items, None)
                if item is not None:
                    pending[pool.submit(request, item, api_key, out / "attempts.jsonl", settings)] = item
            for _ in range(settings["concurrency"]):
                submit_next()
            while pending:
                future = next(as_completed(pending))
                item = pending.pop(future)
                try:
                    reply = future.result()
                except (OSError, ValueError, RuntimeError) as error:
                    failure = failure or error
                    continue
                record = {"key": key(item), "at": now(),
                          "prompt_sha256": digest(item["prompt"].encode()), "response": reply}
                append_jsonl(out / "responses.jsonl", record)
                records[key(item)] = record
                if len(records) % 100 == 0 or len(records) == len(items):
                    print(f"Saved {len(records)}/{len(items)}", flush=True)
                if failure is None:
                    submit_next()
            if failure is not None:
                write_reports(items, records, repo, out, smoke)
                raise failure
    return write_reports(items, records, repo, out, smoke)


def score(repo, out):
    saved = json.loads((out / "manifest.json").read_text())
    smoke = saved["selection"] == "smoke"
    items = select_items(load_items(repo), saved["percent"], saved["seed"], smoke)
    base_url = saved["api_url"].removesuffix("/chat/completions")
    expected = manifest(items, base_url=base_url, model=saved["model"], percent=saved["percent"],
                        seed=saved["seed"], concurrency=saved["concurrency"], smoke=smoke)
    if saved != expected:
        raise ValueError("Saved manifest does not match the pinned data, code, and settings")
    return write_reports(items, saved_responses(out / "responses.jsonl"), repo, out, smoke)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "smoke", "run", "score"))
    parser.add_argument("--repo", type=Path, default=Path("remedqa"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key-env", default="MOONCO_API_KEY")
    parser.add_argument("--percent", type=Decimal, default=Decimal(100))
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command != "score" and (not args.base_url or not args.model):
        parser.error("plan, run, and smoke require --base-url and --model")
    try:
        if args.command == "score":
            result = score(args.repo, args.out)
        else:
            items = select_items(load_items(args.repo), args.percent, args.seed, args.command == "smoke")
            settings = manifest(items, base_url=args.base_url, model=args.model, percent=args.percent,
                                seed=args.seed, concurrency=args.concurrency, smoke=args.command == "smoke")
            if args.command == "plan":
                print(json.dumps(settings, indent=2))
                return 0
            result = run(items, args.repo, args.out, settings, args.api_key_env)
        print(f"Report: {args.out / 'report.md'}")
        return 0 if result["complete"] else 1
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Stopped: {type(error).__name__}; check the pinned input and settings\n")
    except RuntimeError as error:
        parser.exit(1, f"Stopped: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
