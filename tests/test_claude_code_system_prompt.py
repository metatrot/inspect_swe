import asyncio
from types import SimpleNamespace
from typing import cast

import pytest
from inspect_ai.util import SandboxEnvironment
from inspect_swe import claude_code
from inspect_swe._claude_code.agentbinary import (
    claude_code_supports_system_prompt_files,
)
from inspect_swe._claude_code.claude_code import (
    _remove_system_prompt_files,
    _system_prompt_args,
    _write_system_prompt_files,
)


class RecordingSandbox:
    """Minimal sandbox stand-in that records writes and shell commands."""

    async def write_file(self, file: str, contents: str) -> None:
        self.files[file] = contents

    def __init__(self, help_output: str = "") -> None:
        self.files: dict[str, str] = {}
        self.execs: list[str] = []
        self.help_output = help_output

    async def exec(self, cmd: list[str], **kwargs: object) -> object:
        self.execs.append(cmd[-1])
        stdout = self.help_output if "--help" in cmd[-1] else ""
        return SimpleNamespace(success=True, stdout=stdout, stderr="", returncode=0)


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


# how 2.1.221 spells the flags in --help (it mentions rather than lists them)
HELP_WITH_FILE_FLAGS = """  --append-system-prompt <prompt>       Append a system prompt to the default
                                        via: --system-prompt[-file],
                                        --append-system-prompt[-file], --add-dir
"""

# the same section in 2.0.32, which rejects the flags as an unknown option
HELP_WITHOUT_FILE_FLAGS = """  --append-system-prompt <prompt>       Append a system prompt to the default
  --system-prompt <prompt>              System prompt to use for the session
"""


def test_system_prompt_files_detected_from_help() -> None:
    sbox = RecordingSandbox(HELP_WITH_FILE_FLAGS)
    assert asyncio.run(
        claude_code_supports_system_prompt_files(
            cast(SandboxEnvironment, sbox), "claude"
        )
    )


def test_system_prompt_files_absent_from_older_help() -> None:
    sbox = RecordingSandbox(HELP_WITHOUT_FILE_FLAGS)
    assert not asyncio.run(
        claude_code_supports_system_prompt_files(
            cast(SandboxEnvironment, sbox), "claude"
        )
    )


def test_system_prompt_files_detected_when_help_lists_the_flag() -> None:
    sbox = RecordingSandbox("  --append-system-prompt-file <file>\n")
    assert asyncio.run(
        claude_code_supports_system_prompt_files(
            cast(SandboxEnvironment, sbox), "claude"
        )
    )


def test_system_prompt_files_keep_the_prompt_out_of_argv() -> None:
    sbox = RecordingSandbox()
    args, paths = asyncio.run(
        _write_system_prompt_files(
            sbox,
            _system_prompt_args(["Task prompt"], "Replacement prompt", is_resume=False),
            None,
            "/tmp/prompts",
        )
    )

    assert args[0] == "--system-prompt-file"
    assert args[2] == "--append-system-prompt-file"
    assert [args[1], args[3]] == paths
    assert "Task prompt" not in "\0".join(args)
    assert "Replacement prompt" not in "\0".join(args)
    assert sbox.files[paths[0]] == "Replacement prompt"
    assert sbox.files[paths[1]] == "Task prompt"


def test_system_prompt_files_are_not_world_readable() -> None:
    sbox = RecordingSandbox()
    _, paths = asyncio.run(
        _write_system_prompt_files(
            sbox, ["--append-system-prompt", "Task prompt"], "agent", "/tmp/prompts"
        )
    )

    assert f"chmod 600 {paths[0]}" in sbox.execs[0]
    assert f"chown agent {paths[0]}" in sbox.execs[0]


def test_system_prompt_files_are_removed_after_the_agent_exits() -> None:
    sbox = RecordingSandbox()
    _, paths = asyncio.run(
        _write_system_prompt_files(
            sbox, ["--append-system-prompt", "Task prompt"], None, "/tmp/prompts"
        )
    )
    asyncio.run(_remove_system_prompt_files(sbox, paths))

    assert f"rm -f {paths[0]}" in sbox.execs[-1]


def test_no_system_prompt_writes_no_files() -> None:
    sbox = RecordingSandbox()
    args, paths = asyncio.run(
        _write_system_prompt_files(sbox, [], None, "/tmp/prompts")
    )

    assert args == []
    assert paths == []
    assert sbox.files == {}
    asyncio.run(_remove_system_prompt_files(sbox, paths))
    assert sbox.execs == []
