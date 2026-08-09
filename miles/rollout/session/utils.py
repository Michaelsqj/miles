from typing import Any

from miles.utils.chat_template_utils import message_matches


def preserve_ordered_content_blocks(
    stored_messages: list[dict[str, Any]],
    request_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Retain server-owned Inkling block order when clients replay history.

    ``content_blocks`` is deliberately absent from the client response, so the
    next request can only replay the compatible flattened OpenAI fields. Copy
    the stored side channel solely across messages already proven equivalent by
    ``message_matches``; never infer or synthesize it for new messages.
    """
    preserved = list(request_messages)
    for index, request_message in enumerate(preserved):
        if index >= len(stored_messages):
            break
        stored_message = stored_messages[index]
        if (
            stored_message.get("role") == "assistant"
            and "content_blocks" in stored_message
            and message_matches(stored_message, request_message)
        ):
            preserved[index] = {
                **request_message,
                "content_blocks": stored_message["content_blocks"],
            }
    return preserved
