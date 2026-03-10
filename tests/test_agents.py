from src.agents import billing_agent, general_agent, technical_agent


class TestStubAgents:
    def test_billing_agent_returns_response(self, sample_state):
        result = billing_agent(sample_state)
        assert "Billing Agent" in result["response"]
        assert len(result["messages"]) == 1

    def test_technical_agent_returns_response(self, sample_state):
        result = technical_agent(sample_state)
        assert "Technical Agent" in result["response"]
        assert len(result["messages"]) == 1

    def test_general_agent_returns_response(self, sample_state):
        result = general_agent(sample_state)
        assert "General Agent" in result["response"]
        assert len(result["messages"]) == 1
