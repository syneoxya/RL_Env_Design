from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pm_env.schemas.evaluation_run_config import EvaluationRunConfig


@dataclass(frozen=True)
class RolloutRecord:
    group_id: str
    sample_id: str
    transcript_file: str
    prompt: str
    messages: list[dict[str, Any]]
    assistant_text: str
    reward: float
    advantage: float
    metadata: dict[str, Any]


def make_rollout_configs(
    base_config: EvaluationRunConfig,
    *,
    output_dir: Path,
    n_groups: int,
    group_size: int,
    strip_api_key: bool = True,
) -> list[Path]:
    config_dir = output_dir / "configs"
    transcript_dir = output_dir / "transcripts"
    config_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    config_paths = []

    base_run_id = base_config.run_id or "llm-grpo"

    for group_idx in range(n_groups):
        group_id = f"group_{group_idx:04d}"
        for sample_idx in range(group_size):
            sample_id = f"sample_{sample_idx:04d}"
            run_id = f"{base_run_id}-{group_id}-{sample_id}"
            transcript_file = transcript_dir / group_id / f"{sample_id}.json"
            transcript_file.parent.mkdir(parents=True, exist_ok=True)

            config = base_config.model_copy(
                update={
                    "run_id": run_id,
                    "transcript_file": transcript_file.as_posix(),
                    "model_api_key": None
                    if strip_api_key
                    else base_config.model_api_key,
                }
            )
            config_path = config_dir / f"{group_id}_{sample_id}.json"
            config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
            config_paths.append(config_path)

    run_script = output_dir / "run_rollouts.sh"
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Run these commands from the repository root.",
    ]
    commands.extend(
        f"uv run pm_env run --config {path.as_posix()} --containerized false"
        for path in config_paths
    )
    run_script.write_text("\n".join(commands) + "\n", encoding="utf-8")
    run_script.chmod(0o755)

    return config_paths


def _event_type(event: dict[str, Any]) -> str | None:
    return event.get("type")


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif item.get("type") == "text" and "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return ""


def load_transcript(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_config_file(path: Path) -> EvaluationRunConfig:
    return EvaluationRunConfig.model_validate_json(path.read_text(encoding="utf-8"))


def transcript_reward(transcript: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    scoring_events = [
        event for event in transcript.get("events", []) if _event_type(event) == "scoring"
    ]
    if not scoring_events:
        return 0.0, {"error": "No scoring event found."}

    scoring = scoring_events[-1].get("scoring", {})
    reward = float(scoring.get("score", 0.0))
    metadata = scoring.get("metadata") or {}
    return reward, metadata


def transcript_messages(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    messages = []
    for event in transcript.get("events", []):
        if _event_type(event) == "message_added":
            message = event.get("message")
            if isinstance(message, dict):
                messages.append(message)
    return messages


def transcript_prompt_and_completion(
    messages: list[dict[str, Any]],
) -> tuple[str, str]:
    user_messages = [m for m in messages if m.get("role") == "user"]
    assistant_messages = [m for m in messages if m.get("role") == "assistant"]

    prompt = _message_text(user_messages[0]) if user_messages else ""
    assistant_text = "\n\n".join(
        text for text in (_message_text(m) for m in assistant_messages) if text
    )
    return prompt, assistant_text


def discover_transcripts(rollout_dir: Path) -> dict[str, list[Path]]:
    transcript_root = rollout_dir / "transcripts"
    if not transcript_root.exists():
        raise FileNotFoundError(f"Missing transcript directory: {transcript_root}")

    grouped: dict[str, list[Path]] = {}
    for path in sorted(transcript_root.glob("group_*/sample_*.json")):
        group_id = path.parent.name
        grouped.setdefault(group_id, []).append(path)
    return grouped


def export_grpo_records(rollout_dir: Path, output_path: Path) -> list[RolloutRecord]:
    grouped = discover_transcripts(rollout_dir)
    records = []

    for group_id, paths in grouped.items():
        group_payloads = []
        for path in paths:
            transcript = load_transcript(path)
            messages = transcript_messages(transcript)
            prompt, assistant_text = transcript_prompt_and_completion(messages)
            reward, metadata = transcript_reward(transcript)
            group_payloads.append(
                {
                    "path": path,
                    "sample_id": path.stem,
                    "messages": messages,
                    "prompt": prompt,
                    "assistant_text": assistant_text,
                    "reward": reward,
                    "metadata": metadata,
                }
            )

        rewards = [float(payload["reward"]) for payload in group_payloads]
        mean_reward = sum(rewards) / max(len(rewards), 1)
        variance = sum((reward - mean_reward) ** 2 for reward in rewards) / max(
            len(rewards), 1
        )
        std_reward = variance**0.5
        denom = max(std_reward, 1e-6)

        for payload in group_payloads:
            advantage = (float(payload["reward"]) - mean_reward) / denom
            records.append(
                RolloutRecord(
                    group_id=group_id,
                    sample_id=str(payload["sample_id"]),
                    transcript_file=str(payload["path"]),
                    prompt=str(payload["prompt"]),
                    messages=payload["messages"],
                    assistant_text=str(payload["assistant_text"]),
                    reward=float(payload["reward"]),
                    advantage=float(advantage),
                    metadata=payload["metadata"],
                )
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.__dict__) + "\n")

    summary = {
        "num_groups": len(grouped),
        "num_records": len(records),
        "mean_reward": sum(r.reward for r in records) / max(len(records), 1),
        "max_reward": max((r.reward for r in records), default=0.0),
        "min_reward": min((r.reward for r in records), default=0.0),
        "output_path": output_path.as_posix(),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect/export LLM GRPO rollouts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser(
        "create-configs",
        help="Create grouped run configs for LLM GRPO rollout sampling.",
    )
    create_parser.add_argument("--base-config", type=Path, default=Path("run_config.json"))
    create_parser.add_argument("--output-dir", type=Path, default=Path("out/llm_grpo"))
    create_parser.add_argument("--n-groups", type=int, default=1)
    create_parser.add_argument("--group-size", type=int, default=8)
    create_parser.add_argument(
        "--keep-api-key",
        action="store_true",
        help="Keep model_api_key in generated configs instead of using environment variables.",
    )

    export_parser = subparsers.add_parser(
        "export-records",
        help="Export completed rollout transcripts to GRPO JSONL records.",
    )
    export_parser.add_argument("--rollout-dir", type=Path, default=Path("out/llm_grpo"))
    export_parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/llm_grpo/grpo_records.jsonl"),
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.command == "create-configs":
        base_config = parse_config_file(args.base_config)
        config_paths = make_rollout_configs(
            base_config,
            output_dir=args.output_dir,
            n_groups=args.n_groups,
            group_size=args.group_size,
            strip_api_key=not args.keep_api_key,
        )
        print(f"Wrote {len(config_paths)} rollout configs to {args.output_dir / 'configs'}")
        print(f"Run script: {args.output_dir / 'run_rollouts.sh'}")
        return

    if args.command == "export-records":
        records = export_grpo_records(args.rollout_dir, args.output)
        print(f"Wrote {len(records)} GRPO records to {args.output}")
        print(f"Summary: {args.output.with_suffix('.summary.json')}")
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
