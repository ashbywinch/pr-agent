"""A malformed prediction is handed back to the model with the parser's error."""

import asyncio

# the package's import order: pr_agent.log imports config_loader, which imports
# back into it — importing config_loader first is how every entry point does it
import pr_agent.config_loader  # noqa: F401

from pr_agent.algo.prediction_repair import repair_prediction

MALFORMED = """review:
  security_concerns: |
    No
  key_issues_to_review:
    - relevant_file: |
        a.py
      issue_content: The fixture demonstrates this: two singletons are scattered
"""

REPAIRED = """review:
  security_concerns: |
    No
  key_issues_to_review:
    - relevant_file: |
        a.py
      issue_content: |
        The fixture demonstrates this: two singletons are scattered
"""


class _FakeHandler:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat_completion(self, model, system, user, temperature=0.2, img_path=None):
        self.calls.append(user)
        return self.responses.pop(0), "stop"


def _parse(text):
    """The reviewer's parse, reduced to what the loop needs: data, error."""
    import yaml

    try:
        data = yaml.safe_load(text)
    except Exception as e:  # the parser's message names line and column
        return None, str(e)
    return data, ""


def _valid(data):
    return isinstance(data, dict) and bool(data.get("review"))


def _run(responses, response):
    handler = _FakeHandler(responses)
    out = asyncio.run(
        repair_prediction(handler, "m", response, "system", parse=_parse, valid=_valid)
    )
    return out, handler


def test_a_valid_prediction_is_not_sent_back():
    out, handler = _run([], REPAIRED)
    assert out == REPAIRED
    assert handler.calls == []


def test_a_malformed_prediction_is_repaired_with_the_parser_error():
    out, handler = _run([REPAIRED], MALFORMED)
    assert out == REPAIRED
    assert len(handler.calls) == 1
    prompt = handler.calls[0]
    assert "could not be parsed as YAML" in prompt
    assert "line" in prompt and "column" in prompt  # the parser's verdict
    assert "demonstrates this" in prompt  # the model's own answer
    assert "SAME content" in prompt


def test_a_repeated_answer_ends_the_loop():
    out, handler = _run([MALFORMED], MALFORMED)
    assert out == MALFORMED
    assert len(handler.calls) == 1  # no second draw


def test_a_second_bad_answer_still_stops_after_the_bound():
    # a DIFFERENT malformed answer each time: the loop draws twice, then stops
    other_bad = MALFORMED.replace("a.py", "b.py")
    out, handler = _run([other_bad, other_bad + "\n"], MALFORMED)
    assert len(handler.calls) == 2
    assert out != REPAIRED
