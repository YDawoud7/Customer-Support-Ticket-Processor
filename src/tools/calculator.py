import ast
import operator

from langchain_core.tools import tool

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    """Recursively evaluate an AST node using only safe arithmetic operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return SAFE_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval(node.operand)
    else:
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports addition, subtraction,
    multiplication, division, modulo, and exponentiation. Use this tool for
    calculating refund amounts, pro-rated charges, subscription costs, etc.

    Examples: "29.99 * 12", "100 - 29.99", "299 / 12"
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        return f"{expression} = {result}"
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return f"Error evaluating '{expression}': {e}"
