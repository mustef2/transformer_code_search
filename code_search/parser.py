import ast
from pathlib import Path

IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules",
                ".mypy_cache", ".pytest_cache", "site-packages", "dist", "build"}

def extract_python_functions(repository: Path) -> list[dict]:
    """Extract functions, async functions and class methods with source locations."""
    repository = Path(repository).resolve()
    documents = []
    for file_path in repository.rglob("*.py"):
        if any(part in IGNORED_DIRS for part in file_path.relative_to(repository).parts):
            continue
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file_path))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        relative_path = file_path.relative_to(repository).as_posix()
        lines = source.splitlines()
        class_stack = []

        class Visitor(ast.NodeVisitor):
            def visit_ClassDef(self, node):
                class_stack.append(node.name)
                self.generic_visit(node)
                class_stack.pop()

            def visit_FunctionDef(self, node):
                add_function(node, False)
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node):
                add_function(node, True)
                self.generic_visit(node)

        def add_function(node, is_async):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            end = getattr(node, "end_lineno", node.lineno)
            snippet = "\n".join(lines[start - 1:end])
            is_method = bool(class_stack)
            kind = ("async_" if is_async else "") + ("method" if is_method else "function")
            qualified = ".".join(class_stack + [node.name])
            docstring = ast.get_docstring(node) or ""
            args = list(getattr(node.args, "posonlyargs", [])) + list(node.args.args)
            arg_names = [arg.arg for arg in args] + [arg.arg for arg in node.args.kwonlyargs]
            if node.args.vararg:
                arg_names.append("*" + node.args.vararg.arg)
            if node.args.kwarg:
                arg_names.append("**" + node.args.kwarg.arg)
            signature = f"{node.name}({', '.join(arg_names)})"
            documents.append({
                "qualified_name": qualified, "name": node.name, "signature": signature,
                "symbol_type": kind, "file_path": relative_path,
                "start_line": start, "end_line": end, "docstring": docstring,
                "source": snippet,
                "search_text": f"Name: {qualified}\nSignature: {signature}\nType: {kind}\n"
                               f"File: {relative_path}\nDocstring: {docstring}\nPython code:\n{snippet}"
            })
        Visitor().visit(tree)
    return documents
