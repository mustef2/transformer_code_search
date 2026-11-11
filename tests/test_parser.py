from code_search.parser import extract_python_functions

def test_extracts_functions_methods_and_async(tmp_path):
    (tmp_path / "sample.py").write_text(
        "def greet(name):\n    return 'hello ' + name\n\nclass User:\n"
        "    @property\n    def active(self):\n        return True\n"
        "    async def load(self, user_id):\n        return user_id\n",
        encoding="utf-8")
    found = {d["qualified_name"]: d for d in extract_python_functions(tmp_path)}
    assert set(found) == {"greet", "User.active", "User.load"}
    assert found["User.active"]["symbol_type"] == "method"
    assert found["User.load"]["symbol_type"] == "async_method"
    assert found["User.active"]["start_line"] == 5

def test_skips_invalid_python(tmp_path):
    (tmp_path / "bad.py").write_text("def broken(:\n", encoding="utf-8")
    assert extract_python_functions(tmp_path) == []
