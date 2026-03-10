from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.tools.calculator import calculator
from src.tools.code_analysis import analyze_code
from src.tools.search import (
    _format_results,
    create_search_billing_docs,
    create_search_general_docs,
    create_search_technical_docs,
)


class TestSearchTools:
    def test_format_results_with_docs(self):
        docs = [
            Document(
                page_content="Refund policy text",
                metadata={"source": "refund_policy.md", "directory": "billing_docs"},
            )
        ]
        result = _format_results(docs)
        assert "Refund policy text" in result
        assert "billing_docs/refund_policy.md" in result

    def test_format_results_empty(self):
        assert "No relevant" in _format_results([])

    def test_search_billing_docs_calls_retriever(self):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(
                page_content="30-day refund",
                metadata={"source": "refund.md", "directory": "billing_docs"},
            )
        ]
        tool = create_search_billing_docs(retriever=mock_retriever)
        result = tool.invoke("refund policy")
        mock_retriever.invoke.assert_called_once_with("refund policy")
        assert "30-day refund" in result

    def test_search_technical_docs_calls_retriever(self):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(
                page_content="Error 0x8007",
                metadata={"source": "errors.md", "directory": "technical_docs"},
            )
        ]
        tool = create_search_technical_docs(retriever=mock_retriever)
        result = tool.invoke("error code")
        assert "0x8007" in result

    def test_search_general_docs_calls_retriever(self):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(
                page_content="Hours: 9am-5pm",
                metadata={"source": "company.md", "directory": "general_docs"},
            )
        ]
        tool = create_search_general_docs(retriever=mock_retriever)
        result = tool.invoke("business hours")
        assert "9am-5pm" in result


class TestCalculator:
    def test_basic_addition(self):
        assert "15" in calculator.invoke("10 + 5")

    def test_multiplication(self):
        assert "359.88" in calculator.invoke("29.99 * 12")

    def test_division(self):
        result = calculator.invoke("100 / 3")
        assert "33.33" in result

    def test_subtraction(self):
        assert "70.01" in calculator.invoke("100 - 29.99")

    def test_complex_expression(self):
        assert "110" in calculator.invoke("(100 + 10 * 2) - 10")

    def test_division_by_zero(self):
        result = calculator.invoke("10 / 0")
        assert "Error" in result

    def test_rejects_function_calls(self):
        result = calculator.invoke("__import__('os').system('rm -rf /')")
        assert "Error" in result

    def test_rejects_variable_access(self):
        result = calculator.invoke("x + 1")
        assert "Error" in result


class TestCodeAnalysis:
    def test_valid_python(self):
        result = analyze_code.invoke("def hello():\n    print('hi')")
        assert "Python" in result
        assert "Valid" in result
        assert "Lines of code: 2" in result

    def test_python_syntax_error(self):
        result = analyze_code.invoke("def hello(\n    print('hi')")
        assert "Syntax error" in result

    def test_bare_except_warning(self):
        result = analyze_code.invoke("try:\n    x = 1\nexcept:\n    pass")
        assert "Bare" in result

    def test_eval_warning(self):
        result = analyze_code.invoke("result = eval(user_input)")
        assert "security risk" in result

    def test_javascript_detection(self):
        result = analyze_code.invoke("const x = 5;\nfunction foo() { return x; }")
        assert "JavaScript" in result
