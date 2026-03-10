from unittest.mock import MagicMock, patch

from src.models import (
    ALL_CLAUDE,
    ALL_DEEPSEEK,
    COST_OPTIMIZED,
    PRESET_CONFIGS,
    create_chat_model,
)

EXPECTED_KEYS = {"classifier", "billing", "technical", "general", "quality_check"}


class TestCreateChatModel:
    def test_claude_model_returns_chat_anthropic(self):
        mock_cls = MagicMock()
        with patch("langchain_anthropic.ChatAnthropic", mock_cls):
            create_chat_model("claude-sonnet-4-20250514")
        mock_cls.assert_called_once_with(model="claude-sonnet-4-20250514")

    def test_deepseek_model_returns_chat_openai(self):
        mock_cls = MagicMock()
        with patch("langchain_openai.ChatOpenAI", mock_cls), patch.dict(
            "os.environ", {"DEEPSEEK_API_KEY": "test-key"}
        ):
            create_chat_model("deepseek-chat")
        mock_cls.assert_called_once_with(
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key="test-key",
        )

    def test_unknown_prefix_defaults_to_anthropic(self):
        mock_cls = MagicMock()
        with patch("langchain_anthropic.ChatAnthropic", mock_cls):
            create_chat_model("some-other-model")
        mock_cls.assert_called_once_with(model="some-other-model")


class TestPresetConfigs:
    def test_all_claude_has_all_keys(self):
        assert set(ALL_CLAUDE.keys()) == EXPECTED_KEYS

    def test_cost_optimized_has_all_keys(self):
        assert set(COST_OPTIMIZED.keys()) == EXPECTED_KEYS

    def test_all_deepseek_has_all_keys(self):
        assert set(ALL_DEEPSEEK.keys()) == EXPECTED_KEYS

    def test_cost_optimized_uses_deepseek_for_classifier_and_qc(self):
        assert COST_OPTIMIZED["classifier"] == "deepseek-chat"
        assert COST_OPTIMIZED["quality_check"] == "deepseek-chat"

    def test_cost_optimized_uses_claude_for_agents(self):
        assert COST_OPTIMIZED["billing"].startswith("claude")
        assert COST_OPTIMIZED["technical"].startswith("claude")
        assert COST_OPTIMIZED["general"].startswith("claude")

    def test_all_deepseek_uses_deepseek_everywhere(self):
        for key in EXPECTED_KEYS:
            assert ALL_DEEPSEEK[key] == "deepseek-chat"

    def test_preset_configs_dict_has_all_presets(self):
        assert set(PRESET_CONFIGS.keys()) == {"all_claude", "cost_optimized", "all_deepseek"}
