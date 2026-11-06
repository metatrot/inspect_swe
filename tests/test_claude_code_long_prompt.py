"""claude_code launches Claude Code with a prompt longer than one argv entry may be.

The prompt was the last argv entry of `claude --print`, and Linux caps a single
argv entry at 128 KiB (MAX_ARG_STRLEN), so a longer prompt failed to launch with
`OSError(7, 'Argument list too long')`. Claude Code now reads the prompt from
stdin. mockllm answers, so the test needs a sandbox but no model API.
"""

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import resolve_sample_attachments
from inspect_ai.model import ChatMessageUser
from inspect_swe import claude_code

from tests.conftest import assert_eval_completed, skip_if_no_docker

PROMPT = '-x "quoted" $HOME `tick` ünïcødé 😀\n\nReply with: done.\n' + "x" * 200_000


@pytest.mark.slow
@skip_if_no_docker
def test_claude_code_delivers_a_prompt_longer_than_an_argv_entry() -> None:
    task = Task(dataset=[Sample(input=PROMPT)], solver=claude_code(), sandbox="docker")

    log = eval(task, model="mockllm/model", time_limit=300)[0]

    sample = resolve_sample_attachments(assert_eval_completed(log)[0])
    sent = [
        message.text
        for event in sample.events
        if event.event == "model"
        for message in event.input
        if isinstance(message, ChatMessageUser)
    ]
    assert any(PROMPT in text for text in sent)
