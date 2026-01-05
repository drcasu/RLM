"""Comprehensive tests for LocalREPL environment."""

import os

from rlm.environments.local_repl import LocalREPL


class TestLocalREPLBasic:
    """Basic functionality tests for LocalREPL."""

    def test_simple_execution(self):
        """Test basic code execution."""
        repl = LocalREPL()
        result = repl.execute_code("x = 1 + 2")
        assert result.stderr == ""
        assert repl.locals["x"] == 3
        repl.cleanup()

    def test_print_output(self):
        """Test that print statements are captured."""
        repl = LocalREPL()
        result = repl.execute_code("print('Hello, World!')")
        assert "Hello, World!" in result.stdout
        repl.cleanup()

    def test_error_handling(self):
        """Test that errors are captured in stderr."""
        repl = LocalREPL()
        result = repl.execute_code("1 / 0")
        assert "ZeroDivisionError" in result.stderr
        repl.cleanup()

    def test_syntax_error(self):
        """Test syntax error handling."""
        repl = LocalREPL()
        result = repl.execute_code("def broken(")
        assert "SyntaxError" in result.stderr
        repl.cleanup()


class TestLocalREPLPersistence:
    """Tests for state persistence across executions."""

    def test_variable_persistence(self):
        """Test that variables persist across multiple code executions."""
        repl = LocalREPL()

        result1 = repl.execute_code("x = 42")
        assert result1.stderr == ""
        assert repl.locals["x"] == 42

        result2 = repl.execute_code("y = x + 8")
        assert result2.stderr == ""
        assert repl.locals["y"] == 50

        result3 = repl.execute_code("print(y)")
        assert "50" in result3.stdout

        repl.cleanup()

    def test_function_persistence(self):
        """Test that defined functions persist."""
        repl = LocalREPL()

        repl.execute_code(
            """
def greet(name):
    return f"Hello, {name}!"
"""
        )

        result = repl.execute_code("print(greet('World'))")
        assert "Hello, World!" in result.stdout
        repl.cleanup()

    def test_list_comprehension(self):
        """Test that list comprehensions work."""
        repl = LocalREPL()

        repl.execute_code("squares = [x**2 for x in range(5)]")
        assert repl.locals["squares"] == [0, 1, 4, 9, 16]

        result = repl.execute_code("print(sum(squares))")
        assert "30" in result.stdout
        repl.cleanup()


class TestLocalREPLBuiltins:
    """Tests for safe builtins and blocked functions."""

    def test_safe_builtins_available(self):
        """Test that safe builtins are available."""
        repl = LocalREPL()

        # Test various safe builtins
        _ = repl.execute_code("x = len([1, 2, 3])")
        assert repl.locals["x"] == 3

        _ = repl.execute_code("y = sum([1, 2, 3, 4])")
        assert repl.locals["y"] == 10

        _ = repl.execute_code("z = sorted([3, 1, 2])")
        assert repl.locals["z"] == [1, 2, 3]

        repl.cleanup()

    def test_imports_work(self):
        """Test that imports work."""
        repl = LocalREPL()
        result = repl.execute_code("import math\nx = math.pi")
        assert result.stderr == ""
        assert abs(repl.locals["x"] - 3.14159) < 0.001
        repl.cleanup()


class TestLocalREPLContextManager:
    """Tests for context manager usage."""

    def test_context_manager(self):
        """Test using LocalREPL as context manager."""
        with LocalREPL() as repl:
            _ = repl.execute_code("x = 100")
            assert repl.locals["x"] == 100


class TestLocalREPLHelpers:
    """Tests for helper functions (FINAL_VAR, etc.)."""

    def test_final_var_existing(self):
        """Test FINAL_VAR with existing variable."""
        repl = LocalREPL()
        repl.execute_code("answer = 42")
        _ = repl.execute_code("result = FINAL_VAR('answer')")
        assert repl.locals["result"] == "42"
        repl.cleanup()

    def test_final_var_missing(self):
        """Test FINAL_VAR with non-existent variable."""
        repl = LocalREPL()
        _ = repl.execute_code("result = FINAL_VAR('nonexistent')")
        assert "Error" in repl.locals["result"]
        repl.cleanup()

    def test_llm_query_no_handler(self):
        """Test llm_query without handler configured."""
        repl = LocalREPL()
        _ = repl.execute_code("response = llm_query('test')")
        assert "Error" in repl.locals["response"]
        repl.cleanup()


class TestLocalREPLContext:
    """Tests for context loading."""

    def test_string_context(self):
        """Test loading string context."""
        repl = LocalREPL(context_payload="This is the context data.")
        assert "context" in repl.locals
        assert repl.locals["context"] == "This is the context data."
        repl.cleanup()

    def test_dict_context(self):
        """Test loading dict context."""
        repl = LocalREPL(context_payload={"key": "value", "number": 42})
        assert "context" in repl.locals
        assert repl.locals["context"]["key"] == "value"
        assert repl.locals["context"]["number"] == 42
        repl.cleanup()

    def test_list_context(self):
        """Test loading list context."""
        repl = LocalREPL(context_payload=[1, 2, 3, "four"])
        assert "context" in repl.locals
        assert repl.locals["context"] == [1, 2, 3, "four"]
        repl.cleanup()


class TestLocalREPLCleanup:
    """Tests for cleanup behavior."""

    def test_cleanup_clears_state(self):
        """Test that cleanup clears the namespace."""
        repl = LocalREPL()
        repl.execute_code("x = 42")
        assert "x" in repl.locals
        repl.cleanup()
        assert len(repl.locals) == 0

    def test_temp_dir_created_and_cleaned(self):
        """Test that temp directory is created and cleaned up."""
        repl = LocalREPL()
        temp_dir = repl.temp_dir
        assert os.path.exists(temp_dir)
        repl.cleanup()
        assert not os.path.exists(temp_dir)


class TestLocalREPLSecuritySafeImport:
    """Tests for _safe_import security control."""

    def test_safe_import_blocks_os(self):
        """Test that importing os is blocked at runtime."""
        repl = LocalREPL()
        result = repl.execute_code("import os")
        # Should fail at static analysis
        assert "Security" in result.stderr or "ImportError" in result.stderr
        repl.cleanup()

    def test_safe_import_blocks_subprocess(self):
        """Test that importing subprocess is blocked."""
        repl = LocalREPL()
        result = repl.execute_code("import subprocess")
        assert "Security" in result.stderr or "ImportError" in result.stderr
        repl.cleanup()

    def test_safe_import_blocks_socket(self):
        """Test that importing socket is blocked."""
        repl = LocalREPL()
        result = repl.execute_code("import socket")
        assert "Security" in result.stderr or "ImportError" in result.stderr
        repl.cleanup()

    def test_safe_import_allows_math(self):
        """Test that safe modules like math are allowed."""
        repl = LocalREPL()
        result = repl.execute_code("import math\nx = math.pi")
        assert result.stderr == ""
        assert abs(repl.locals["x"] - 3.14159) < 0.001
        repl.cleanup()

    def test_safe_import_allows_json(self):
        """Test that safe modules like json are allowed."""
        repl = LocalREPL()
        result = repl.execute_code("import json\nx = json.dumps({'a': 1})")
        assert result.stderr == ""
        assert repl.locals["x"] == '{"a": 1}'
        repl.cleanup()

    def test_safe_import_blocks_from_import(self):
        """Test that from imports of dangerous modules are blocked."""
        repl = LocalREPL()
        result = repl.execute_code("from os import path")
        assert "Security" in result.stderr or "ImportError" in result.stderr
        repl.cleanup()


class TestLocalREPLSecuritySafeOpen:
    """Tests for _safe_open security control."""

    def test_safe_open_allows_temp_dir_access(self):
        """Test that files in temp_dir can be accessed."""
        repl = LocalREPL()
        # Write a file in temp_dir and read it back
        result = repl.execute_code("""
with open('test_file.txt', 'w') as f:
    f.write('hello')
with open('test_file.txt', 'r') as f:
    content = f.read()
""")
        assert result.stderr == ""
        assert repl.locals["content"] == "hello"
        repl.cleanup()

    def test_safe_open_blocks_parent_directory(self):
        """Test that accessing files outside temp_dir is blocked."""
        repl = LocalREPL()
        result = repl.execute_code("with open('../../../etc/passwd', 'r') as f: pass")
        assert "PermissionError" in result.stderr or "denied" in result.stderr.lower()
        repl.cleanup()

    def test_safe_open_blocks_absolute_path(self):
        """Test that absolute paths outside temp_dir are blocked."""
        repl = LocalREPL()
        # Try to access a file outside the temp directory
        if os.name == "nt":  # Windows
            result = repl.execute_code(
                "with open('C:\\\\Windows\\\\System32\\\\config\\\\sam', 'r') as f: pass"
            )
        else:  # Unix-like
            result = repl.execute_code("with open('/etc/passwd', 'r') as f: pass")
        assert "PermissionError" in result.stderr or "denied" in result.stderr.lower()
        repl.cleanup()

    def test_safe_open_blocks_path_traversal(self):
        """Test that path traversal attacks are blocked."""
        repl = LocalREPL()
        result = repl.execute_code(
            "with open('./subdir/../../../../../../etc/passwd', 'r') as f: pass"
        )
        assert "PermissionError" in result.stderr or "denied" in result.stderr.lower()
        repl.cleanup()


class TestLocalREPLSecurityTimeout:
    """Tests for execution timeout security control."""

    def test_timeout_configured_correctly(self):
        """Test that timeout is configured with default value."""
        repl = LocalREPL()
        assert repl.execution_timeout == 30
        repl.cleanup()

    def test_custom_timeout_value(self):
        """Test that custom timeout can be set."""
        repl = LocalREPL(execution_timeout=60)
        assert repl.execution_timeout == 60
        repl.cleanup()

    def test_timeout_disabled_with_zero(self):
        """Test that timeout can be disabled with 0."""
        repl = LocalREPL(execution_timeout=0)
        assert repl.execution_timeout == 0
        repl.cleanup()

    def test_short_timeout_catches_infinite_loop(self):
        """Test that infinite loops are caught by timeout."""
        # Use a very short timeout for testing
        repl = LocalREPL(execution_timeout=1)
        result = repl.execute_code("while True: pass")
        assert "ExecutionTimeoutError" in result.stderr or "timeout" in result.stderr.lower()
        repl.cleanup()

    def test_fast_code_completes_within_timeout(self):
        """Test that fast code completes successfully."""
        repl = LocalREPL(execution_timeout=5)
        result = repl.execute_code("x = sum(range(1000))")
        assert result.stderr == ""
        assert repl.locals["x"] == 499500
        repl.cleanup()


class TestLocalREPLSecurityMemoryLimit:
    """Tests for memory limit security control."""

    def test_memory_limit_configured_correctly(self):
        """Test that memory limit is configured with default value."""
        repl = LocalREPL()
        assert repl.memory_limit_mb == 512
        repl.cleanup()

    def test_custom_memory_limit(self):
        """Test that custom memory limit can be set."""
        repl = LocalREPL(memory_limit_mb=256)
        assert repl.memory_limit_mb == 256
        repl.cleanup()

    def test_memory_limit_disabled_with_zero(self):
        """Test that memory limit can be disabled with 0."""
        repl = LocalREPL(memory_limit_mb=0)
        assert repl.memory_limit_mb == 0
        repl.cleanup()


class TestLocalREPLSecuritySafeGetattr:
    """Tests for _safe_getattr security control."""

    def test_safe_getattr_allows_normal_attribute(self):
        """Test that normal attribute access via getattr is allowed."""
        repl = LocalREPL()
        result = repl.execute_code("""
x = getattr([1, 2, 3], '__len__')()
""")
        assert result.stderr == ""
        assert repl.locals["x"] == 3
        repl.cleanup()

    def test_safe_getattr_blocks_dunder_class(self):
        """Test that getattr blocks __class__ access."""
        repl = LocalREPL()
        result = repl.execute_code("x = getattr([], '__class__')")
        # Blocked by static analysis (Security) or runtime (_safe_getattr -> AttributeError)
        assert "Security" in result.stderr or "AttributeError" in result.stderr
        assert "__class__" in result.stderr
        repl.cleanup()

    def test_safe_getattr_blocks_dunder_globals(self):
        """Test that getattr blocks __globals__ access."""
        repl = LocalREPL()
        result = repl.execute_code("""
def f(): pass
x = getattr(f, '__globals__')
""")
        # Blocked by static analysis (Security) or runtime (_safe_getattr -> AttributeError)
        assert "Security" in result.stderr or "AttributeError" in result.stderr
        assert "__globals__" in result.stderr
        repl.cleanup()

    def test_safe_getattr_blocks_dunder_code(self):
        """Test that getattr blocks __code__ access."""
        repl = LocalREPL()
        result = repl.execute_code("""
def f(): pass
x = getattr(f, '__code__')
""")
        # Blocked by static analysis (Security) or runtime (_safe_getattr -> AttributeError)
        assert "Security" in result.stderr or "AttributeError" in result.stderr
        assert "__code__" in result.stderr
        repl.cleanup()

    def test_safe_getattr_blocks_dunder_subclasses(self):
        """Test that getattr blocks __subclasses__ access."""
        repl = LocalREPL()
        result = repl.execute_code("x = getattr(object, '__subclasses__')")
        # Blocked by static analysis (Security) or runtime (_safe_getattr -> AttributeError)
        assert "Security" in result.stderr or "AttributeError" in result.stderr
        assert "__subclasses__" in result.stderr
        repl.cleanup()

    def test_safe_getattr_blocks_dunder_builtins(self):
        """Test that getattr blocks __builtins__ access."""
        repl = LocalREPL()
        result = repl.execute_code("x = getattr({}, '__builtins__', None)")
        # Blocked by static analysis (Security) or runtime (_safe_getattr -> AttributeError)
        assert "Security" in result.stderr or "AttributeError" in result.stderr
        assert "__builtins__" in result.stderr
        repl.cleanup()

    def test_safe_getattr_with_default_on_blocked_attr(self):
        """Test that getattr with default still blocks dangerous attributes."""
        repl = LocalREPL()
        result = repl.execute_code("x = getattr({}, '__class__', 'safe_default')")
        # Blocked by static analysis (Security) or runtime (_safe_getattr -> AttributeError)
        assert "Security" in result.stderr or "AttributeError" in result.stderr
        assert "__class__" in result.stderr
        repl.cleanup()

    def test_safe_getattr_runtime_with_dynamic_attr(self):
        """Test that _safe_getattr blocks dynamic dangerous attribute access at runtime."""
        repl = LocalREPL()
        # Use dynamic attribute name to bypass static analysis - runtime must catch this
        result = repl.execute_code("""
attr_name = '__' + 'class' + '__'
x = getattr([], attr_name)
""")
        assert "AttributeError" in result.stderr
        assert "blocked" in result.stderr.lower()
        repl.cleanup()

    def test_safe_setattr_runtime_with_dynamic_attr(self):
        """Test that _safe_setattr blocks dynamic dangerous attribute setting at runtime."""
        repl = LocalREPL()
        # Use dynamic attribute name to bypass static analysis - runtime must catch this
        result = repl.execute_code("""
import types
obj = types.SimpleNamespace()
attr_name = '__' + 'dict' + '__'
setattr(obj, attr_name, {})
""")
        assert "AttributeError" in result.stderr
        assert "blocked" in result.stderr.lower()
        repl.cleanup()


class TestLocalREPLSecuritySafeSetattr:
    """Tests for _safe_setattr security control."""

    def test_safe_setattr_allows_normal_attribute(self):
        """Test that normal attribute setting via setattr is allowed."""
        repl = LocalREPL()
        # Use an object with __dict__ that allows setting attributes
        result = repl.execute_code("""
import types
obj = types.SimpleNamespace()
setattr(obj, 'name', 'test')
x = obj.name
""")
        assert result.stderr == ""
        assert repl.locals["x"] == "test"
        repl.cleanup()

    def test_safe_setattr_blocks_dunder_class(self):
        """Test that setattr blocks __class__ modification."""
        repl = LocalREPL()
        result = repl.execute_code("""
import types
obj = types.SimpleNamespace()
setattr(obj, '__class__', object)
""")
        assert "AttributeError" in result.stderr
        assert "blocked" in result.stderr.lower() or "__class__" in result.stderr
        repl.cleanup()

    def test_safe_setattr_blocks_dunder_dict(self):
        """Test that setattr blocks __dict__ modification."""
        repl = LocalREPL()
        result = repl.execute_code("""
import types
obj = types.SimpleNamespace()
setattr(obj, '__dict__', {})
""")
        assert "AttributeError" in result.stderr
        assert "blocked" in result.stderr.lower() or "__dict__" in result.stderr
        repl.cleanup()


class TestLocalREPLSecurityNetworkBlocking:
    """Tests for network module import blocking."""

    def test_blocks_urllib_import(self):
        """Test that urllib import is blocked."""
        repl = LocalREPL()
        result = repl.execute_code("import urllib")
        assert "Security" in result.stderr or "ImportError" in result.stderr
        repl.cleanup()

    def test_blocks_http_import(self):
        """Test that http module import is blocked."""
        repl = LocalREPL()
        result = repl.execute_code("import http")
        assert "Security" in result.stderr or "ImportError" in result.stderr
        repl.cleanup()

    def test_blocks_urllib_request_from_import(self):
        """Test that from urllib.request import is blocked."""
        repl = LocalREPL()
        result = repl.execute_code("from urllib.request import urlopen")
        assert "Security" in result.stderr or "ImportError" in result.stderr
        repl.cleanup()

    def test_blocks_http_client_from_import(self):
        """Test that from http.client import is blocked."""
        repl = LocalREPL()
        result = repl.execute_code("from http.client import HTTPConnection")
        assert "Security" in result.stderr or "ImportError" in result.stderr
        repl.cleanup()


class TestLocalREPLSecurityTmpfs:
    """Tests for tmpfs temp directory support."""

    def test_tmpfs_disabled_by_default(self):
        """Test that tmpfs is disabled by default."""
        repl = LocalREPL()
        assert repl.use_tmpfs is False
        assert repl._tmpfs_mounted is False
        repl.cleanup()

    def test_tmpfs_can_be_enabled(self):
        """Test that tmpfs can be enabled via parameter."""
        repl = LocalREPL(use_tmpfs=True)
        assert repl.use_tmpfs is True
        # On non-Linux or without privileges, it may not be mounted
        # but the parameter should be set
        repl.cleanup()

    def test_tmpfs_size_configurable(self):
        """Test that tmpfs size can be configured."""
        repl = LocalREPL(use_tmpfs=True, tmpfs_size_mb=128)
        assert repl.tmpfs_size_mb == 128
        repl.cleanup()

    def test_tmpfs_default_size(self):
        """Test that tmpfs has a default size."""
        repl = LocalREPL()
        assert repl.tmpfs_size_mb == 64  # DEFAULT_TMPFS_SIZE_MB
        repl.cleanup()

    def test_temp_dir_still_works_with_tmpfs_enabled(self):
        """Test that temp dir operations work when tmpfs is requested."""
        repl = LocalREPL(use_tmpfs=True)
        # Should still be able to write files regardless of tmpfs mount status
        result = repl.execute_code("""
with open('test.txt', 'w') as f:
    f.write('hello')
with open('test.txt', 'r') as f:
    content = f.read()
""")
        assert result.stderr == ""
        assert repl.locals["content"] == "hello"
        repl.cleanup()
