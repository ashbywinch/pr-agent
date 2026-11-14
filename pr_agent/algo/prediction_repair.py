"""Repair a prediction the parser rejected, by handing the model its own answer.

The model is asked to serialize its answer (YAML for the review), and it
sometimes writes a plain scalar where the format needs a block scalar — an
unquoted colon inside prose is the usual case. Parsing then fails on text the
model itself produced, and re-rolling the same request just draws again.

The parser's message names the line, the column and the offending text, which
is exactly the correction the model needs. This module owns that loop: parse,
hand back the answer and the error, parse again, bounded.
"""

from typing import Any, Callable, Tuple


def _logger():
    """pr_agent.log imports config_loader, which imports back into it — a
    top-level import here fails when this module is the first one loaded."""
    from pr_agent.log import get_logger

    return get_logger()

# Two attempts: the first usually corrects it, and a model that has not by the
# second is not going to.
MAX_REPAIR_ATTEMPTS = 2


def repair_prompt(previous: str, error: str) -> str:
    """The message that hands the model its own answer and the parser's verdict."""
    return (
        "Your previous answer could not be parsed as YAML.\n"
        f"The parser said: {error}\n"
        "Here is your previous answer:\n"
        f"{previous}\n"
        "Return the SAME content as valid YAML. Keep every finding, header and value identical — "
        "only the serialization changes. Wrap every prose value in a `|` block scalar: a plain "
        "value containing a colon is what failed. No commentary, no code fences."
    )


async def repair_prediction(
    handler,
    model: str,
    response: str,
    system_prompt: str,
    parse: Callable[[str], Tuple[Any, str]],
    valid: Callable[[Any], bool],
    temperature: float = 0.2,
    max_attempts: int = MAX_REPAIR_ATTEMPTS,
) -> str:
    """Return `response`, or the model's re-serialized version of it.

    `parse` returns (data, error) with an empty error when the text parsed;
    `valid` says whether the parsed data carries what the caller needs. The loop
    stops on success, on a repeated answer (the model has nothing new to say),
    or when the attempts run out — the caller then reports the failure it would
    have reported anyway.
    """
    data, error = parse(response)
    if not error and valid(data):
        return response
    for _attempt in range(max_attempts):
        repaired, _finish = await handler.chat_completion(
            model=model,
            temperature=temperature,
            system=system_prompt,
            user=repair_prompt(response, error),
        )
        if repaired.strip() == response.strip():
            _logger().warning("Prediction repair returned the same text; stopping")
            break
        data, error = parse(repaired)
        if not error and valid(data):
            _logger().info("Recovered a malformed prediction by repairing its YAML")
            return repaired
        response = repaired
    _logger().warning("Prediction still unparseable after repair",
                         artifact={'response_text': response})
    return response
