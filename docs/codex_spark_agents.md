# Astra parent with Spark roles

Verified on 2026-09-06 using Linux, Codex CLI 0.153.4, and ChatGPT sign-in.
The standing workflow is recorded in [AGENTS.md](../AGENTS.md). Project defaults
are in [`.codex/config.toml`](../.codex/config.toml).

## Verified result

A new CLI session selected the custom agent with these routing parameters:

```json
{
  "task_name": "routing_test",
  "agent_type": "spark_explorer",
  "fork_turns": "none"
}
```

The spawn call did not supply a model override. Codex's persisted thread records
and `turn_context` events recorded:

| Thread | Model | Reasoning | Sandbox |
| --- | --- | --- | --- |
| Parent | `gpt-6-astra` | `medium` | `read-only` |
| Child, role `spark_explorer` | `gpt-5.3-codex-spark` | `medium` | `read-only` |

Parent thread ID: `01a07559-01ca-7f23-8788-38e94d001286`.
Child thread ID: `01a07559-2d34-7ba3-8693-c84bd15d75ad`.

The child read `tools/train_response_control.py` and correctly identified
`main(argv=None)` at line 107 and the module guard at line 136. Both turns
completed successfully; the CLI exited with code 0. No project tests,
validation, or training were run. The model check used runtime records rather
than the agent's self-description.

## Reproduce

Start a new session from this project directory with the intended sandbox:

```bash
codex -m gpt-6-astra -s workspace-write
```

Use `-s read-only` for inspection-only sessions.

Then ask:

> Use the custom agent spark_explorer to inspect tools/train_response_control.py
> and report its main entry-point function and line number. Spawn exactly one
> subagent using the agent type selector, without a runtime model override.
> Wait for its result. Do not modify files or run tests, validation, or training.

For independent code searches, the same custom agent can be requested multiple
times with separate bounded tasks. This validation exercised one child only.
The parent default was already Astra during the historical test; it is now also
set explicitly in the project configuration.

The chat that created this file did not expose a custom-agent selector or Spark
in its spawn tool. The fresh local CLI session did expose `agent_type` and
loaded this project agent. This test establishes the CLI path; it does not
establish hot reload in an existing chat or behavior in Windows Desktop.

The read-only sandbox was verified with a read-only parent. Parent runtime
permission overrides can take precedence over custom-agent sandbox defaults.

## Persistent workflow

Project defaults select an Astra parent and enable up to three concurrent Spark
children, with Spark/medium as the subagent defaults. Fixed custom roles remain
the required dispatch route; do not supply a runtime model override.

| Role | Task | Configuration |
| --- | --- | --- |
| `spark_explorer` | Code search and call-chain tracing | [spark-explorer.toml](../.codex/agents/spark-explorer.toml) |
| `spark_log_analyzer` | Existing logs and metric evidence | [spark-log-analyzer.toml](../.codex/agents/spark-log-analyzer.toml) |
| `spark_test_runner` | Assigned focused CPU checks | [spark-test-runner.toml](../.codex/agents/spark-test-runner.toml) |
| `spark_implementation_worker` | Bounded edits to assigned files | [spark-implementation-worker.toml](../.codex/agents/spark-implementation-worker.toml) |

All four files select `gpt-5.3-codex-spark` with `medium` reasoning. The explorer
and log analyzer default to read-only; the tester and implementation worker
use workspace-write. Parent runtime permission overrides may take precedence.

Astra handles global reasoning, task decomposition, architecture, integration,
and final review. Spark handles most routine searches, log analysis, focused
tests, and local edits. Give each child a self-contained task using
`fork_turns="none"`, explicit file ownership, and expected evidence. Parallelize
independent read tasks and serialize edits to shared files. Reuse same-task
children for follow-ups; Spark children must not delegate or start nested Codex.

Start a fresh Codex session to load changed agent files. When the active chat
cannot select custom roles, its Astra parent can launch a fresh local
`codex exec -m gpt-6-astra` session, instruct that session to dispatch through
`agent_type`, and collect its result. Avoid creating another orchestrator per
tiny subtask. Check routing in runtime records when establishing a new path or
debugging routing failures. Report unavailable Spark routing without silently
substituting another model.

Only run training or simulation validation when the user explicitly requests
it. Assign focused tests according to the actual change.

## Implementation worker verification

The 2026-09-06 configuration task also dispatched
`spark_implementation_worker` through a fresh Astra CLI session. Persisted
thread records identify parent `01a0755f-1376-7f11-874a-4808470a707c` as
`gpt-6-astra` and child `01a0755f-955a-7920-9625-d9bd411ec23e` as
`gpt-5.3-codex-spark`, with medium reasoning. The worker edited AGENTS.md and
this document and supplied the proposed TOML contents. The CLI exited with
code 0.

The worker's workspace-write sandbox protected `.codex` from edits. The outer
parent reviewed and applied the configuration within its existing permissions.
For future protected-path edits, return a concrete patch to the parent instead
of retrying writes. Application code, training, and simulation validation were
outside this configuration task. Log analyzer and test runner execution have
not been separately exercised; their configuration is installed.

## Example task prompt

> Use spark_explorer and spark_log_analyzer in parallel for the assigned code
> path and relevant existing logs. Synthesize their evidence yourself, then
> assign any bounded fix to spark_implementation_worker and focused checks to
> spark_test_runner where authorized. Select each through agent_type with
> fork_turns="none" and no runtime model override. Keep architecture decisions
> and final review in Astra. Do not run training or simulation validation.

Official documentation:
[Subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents)
documents project agents, per-agent model settings, permission inheritance,
and a Spark explorer example.
