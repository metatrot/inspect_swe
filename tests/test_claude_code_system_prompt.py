import asyncio
import subprocess
from pathlib import Path
from typing import cast

import pytest
from inspect_ai.util import ExecResult, SandboxEnvironment
from inspect_swe import claude_code
from inspect_swe._claude_code.agentbinary import (
    claude_code_binary_version,
    claude_code_supports_system_prompt_files,
)
from inspect_swe._claude_code.claude_code import (
    _system_prompt_args,
    _system_prompt_file_args,
)


class LocalSandbox:
    def __init__(self) -> None:
        self.users: list[str | None] = []

    async def exec(
        self,
        cmd: list[str],
        input: str | bytes | None = None,
        user: str | None = None,
        **kwargs: object,
    ) -> ExecResult[str]:
        self.users.append(user)
        completed = subprocess.run(cmd, input=input, capture_output=True, text=True)
        return ExecResult(
            success=completed.returncode == 0,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def fake_claude(tmp_path: Path, script: str) -> str:
    binary = tmp_path / "claude"
    binary.write_text(f"#!/bin/sh\n{script}\n")
    binary.chmod(0o755)
    return str(binary)


def test_system_prompt_appends_to_default() -> None:
    assert _system_prompt_args(
        ["Task prompt", "Agent prompt"], None, is_resume=False
    ) == [
        "--append-system-prompt",
        "Task prompt\n\nAgent prompt",
    ]


def test_system_prompt_can_replace_default() -> None:
    assert _system_prompt_args([], "Replacement prompt", is_resume=False) == [
        "--system-prompt",
        "Replacement prompt",
    ]


def test_task_prompt_is_appended_to_replacement() -> None:
    assert _system_prompt_args(
        ["Task prompt"], "Replacement prompt", is_resume=False
    ) == [
        "--system-prompt",
        "Replacement prompt",
        "--append-system-prompt",
        "Task prompt",
    ]


def test_empty_system_prompts_add_no_cli_flags() -> None:
    assert _system_prompt_args([], None, is_resume=False) == []


def test_resume_reapplies_replacement_without_appended_messages() -> None:
    assert _system_prompt_args(
        ["Round-tripped prompt"], "Replacement prompt", is_resume=True
    ) == [
        "--system-prompt",
        "Replacement prompt",
    ]


def test_append_and_replace_system_prompts_are_mutually_exclusive() -> None:
    with pytest.raises(
        ValueError,
        match="system_prompt and replace_system_prompt cannot both be specified",
    ):
        claude_code(
            system_prompt="Additional prompt",
            replace_system_prompt="Replacement prompt",
        )


def test_claude_code_binary_version_parses_the_version_line(tmp_path: Path) -> None:
    binary = fake_claude(tmp_path, 'echo "2.1.221 (Claude Code)"')
    version = asyncio.run(
        claude_code_binary_version(cast(SandboxEnvironment, LocalSandbox()), binary)
    )
    assert version == (2, 1, 221)


def test_claude_code_binary_version_probes_as_the_agent_user(tmp_path: Path) -> None:
    binary = fake_claude(tmp_path, 'echo "2.1.221 (Claude Code)"')
    sbox = LocalSandbox()
    asyncio.run(
        claude_code_binary_version(cast(SandboxEnvironment, sbox), binary, "agent")
    )
    assert sbox.users == ["agent"]


def test_claude_code_binary_version_rejects_unparseable_output(
    tmp_path: Path,
) -> None:
    binary = fake_claude(tmp_path, 'echo "Claude Code"')
    with pytest.raises(RuntimeError, match="Unable to parse claude code version"):
        asyncio.run(
            claude_code_binary_version(cast(SandboxEnvironment, LocalSandbox()), binary)
        )


def test_claude_code_binary_version_raises_when_the_probe_fails(
    tmp_path: Path,
) -> None:
    binary = fake_claude(tmp_path, "echo broken >&2; exit 1")
    with pytest.raises(RuntimeError, match="broken"):
        asyncio.run(
            claude_code_binary_version(cast(SandboxEnvironment, LocalSandbox()), binary)
        )


HELP_HIDING_THE_FILE_FLAGS = """  --append-system-prompt <prompt>       Append a system prompt to the default
  --system-prompt <prompt>              System prompt to use for the session
"""


def fake_claude_with_hidden_file_flags(tmp_path: Path, version: str) -> str:
    return fake_claude(
        tmp_path,
        f'case "$1" in --version) echo "{version} (Claude Code)";; '
        f"--help) cat <<'EOF'\n{HELP_HIDING_THE_FILE_FLAGS}EOF\n;; esac",
    )


def test_system_prompt_files_supported_from_2_0_33_even_when_help_hides_them(
    tmp_path: Path,
) -> None:
    binary = fake_claude_with_hidden_file_flags(tmp_path, "2.0.33")
    assert asyncio.run(
        claude_code_supports_system_prompt_files(
            cast(SandboxEnvironment, LocalSandbox()), binary
        )
    )


def test_system_prompt_files_unsupported_below_2_0_33(tmp_path: Path) -> None:
    binary = fake_claude_with_hidden_file_flags(tmp_path, "2.0.32")
    assert not asyncio.run(
        claude_code_supports_system_prompt_files(
            cast(SandboxEnvironment, LocalSandbox()), binary
        )
    )


def test_system_prompt_files_keep_the_prompt_out_of_argv(tmp_path: Path) -> None:
    args = asyncio.run(
        _system_prompt_file_args(
            LocalSandbox(),
            _system_prompt_args(["Task prompt"], "Replacement prompt", is_resume=False),
            None,
            str(tmp_path),
        )
    )

    assert args[0] == "--system-prompt-file"
    assert args[2] == "--append-system-prompt-file"
    assert "Task prompt" not in "\0".join(args)
    assert "Replacement prompt" not in "\0".join(args)
    assert Path(args[1]).parent == tmp_path
    assert Path(args[1]).read_text() == "Replacement prompt"
    assert Path(args[3]).read_text() == "Task prompt"


def test_system_prompt_files_are_written_as_the_agent_user(tmp_path: Path) -> None:
    sbox = LocalSandbox()
    args = asyncio.run(
        _system_prompt_file_args(
            sbox, ["--append-system-prompt", "Task prompt"], "agent", str(tmp_path)
        )
    )

    assert sbox.users == ["agent"]
    assert Path(args[1]).read_text() == "Task prompt"


def test_system_prompt_file_write_failure_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Error writing system prompt"):
        asyncio.run(
            _system_prompt_file_args(
                LocalSandbox(),
                ["--append-system-prompt", "Task prompt"],
                None,
                str(tmp_path / "missing"),
            )
        )


def test_no_system_prompt_writes_no_files(tmp_path: Path) -> None:
    sbox = LocalSandbox()
    args = asyncio.run(_system_prompt_file_args(sbox, [], None, str(tmp_path)))

    assert args == []
    assert list(tmp_path.iterdir()) == []
    assert sbox.users == []
