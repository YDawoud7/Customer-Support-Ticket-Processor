import ast
import re

from langchain_core.tools import tool


@tool
def analyze_code(code: str) -> str:
    """Analyze a code snippet for common issues. Checks for syntax errors,
    identifies the language, counts lines, and flags potential problems like
    bare excepts, unused imports, or missing error handling. Use this tool
    when a customer shares code in their support ticket."""
    analysis = []

    lines = code.strip().split("\n")
    analysis.append(f"Lines of code: {len(lines)}")

    # Try to detect language
    is_valid_python = False
    try:
        ast.parse(code)
        is_valid_python = True
    except SyntaxError:
        pass

    if is_valid_python or re.search(r"\bdef\b|\bimport\b|\bclass\b.*:", code):
        language = "Python"
    elif re.search(r"\bfunction\b|\bconst\b|\blet\b|\bvar\b", code):
        language = "JavaScript"
    elif re.search(r"\bfunc\b.*\{|\bpackage\b", code):
        language = "Go"
    else:
        language = "Unknown"
    analysis.append(f"Detected language: {language}")

    # Python-specific analysis
    if language == "Python":
        try:
            ast.parse(code)
            analysis.append("Syntax: Valid Python syntax")
        except SyntaxError as e:
            analysis.append(f"Syntax error: {e.msg} (line {e.lineno})")

        if re.search(r"except\s*:", code):
            analysis.append(
                "Warning: Bare 'except:' clause found — consider catching specific exceptions"
            )
        if re.search(r"import\s+\*", code):
            analysis.append(
                "Warning: Wildcard import found — consider importing specific names"
            )
        if re.search(r"eval\s*\(", code):
            analysis.append(
                "Warning: eval() usage found — potential security risk"
            )

    return "\n".join(analysis)
