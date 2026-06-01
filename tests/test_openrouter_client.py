import json
from unittest.mock import patch

from site4drug_inference.common.openrouter_client import ApproxChatRenderer, OpenRouterChatClient


class _FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps({"choices": [{"message": {"content": "{\"ok\": true}"}}]}).encode("utf-8")


def test_openrouter_client_posts_chat_completion_payload():
    captured = {}

    def fake_urlopen(request, timeout):
        headers = {key.lower(): value for key, value in request.header_items()}
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["authorization"] = headers.get("authorization")
        captured["referer"] = headers.get("http-referer")
        captured["title"] = headers.get("x-openrouter-title")
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse()

    client = OpenRouterChatClient(
        api_key="or-test",
        model="openai/test-model",
        referer="https://example.org",
        title="Site4Drug Test",
        timeout=7,
    )
    with patch("urllib.request.urlopen", fake_urlopen):
        text = client.sample_messages(
            messages=[{"role": "user", "content": "Return JSON"}],
            max_tokens=32,
            temperature=0.0,
            sampling_seed=123,
        )

    assert text == "{\"ok\": true}"
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["timeout"] == 7
    assert captured["authorization"] == "Bearer or-test"
    assert captured["referer"] == "https://example.org"
    assert captured["title"] == "Site4Drug Test"
    assert captured["payload"]["model"] == "openai/test-model"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "Return JSON"}]
    assert captured["payload"]["seed"] == 123


def test_approx_chat_renderer_exposes_budget_length():
    renderer = ApproxChatRenderer()
    prompt = renderer.build_generation_prompt(
        [
            {"role": "system", "content": "You are Site4Drug."},
            {"role": "user", "content": "Find a site."},
        ]
    )
    assert prompt.length > 0
    assert "system: You are Site4Drug." in prompt.text
