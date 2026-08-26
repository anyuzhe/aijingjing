from __future__ import annotations


def codex_http_transport_args(provider_id: str = "openai-http") -> list[str]:
    """Use Codex login over HTTPS when the local network cannot sustain WebSockets."""

    return [
        "--config",
        f'model_provider="{provider_id}"',
        "--config",
        f'model_providers.{provider_id}.name="OpenAI HTTP"',
        "--config",
        f"model_providers.{provider_id}.requires_openai_auth=true",
        "--config",
        f"model_providers.{provider_id}.supports_websockets=false",
        "--config",
        f'model_providers.{provider_id}.wire_api="responses"',
    ]
