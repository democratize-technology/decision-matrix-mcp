"""
Test __main__.py module execution
"""

import subprocess
import sys
from pathlib import Path


def test_main_module_execution():
    """Test that __main__.py can be executed as a module"""
    # Get the path to the package
    package_path = Path(__file__).parent.parent / "src"

    # Run the module as a script with a timeout
    try:
        result = subprocess.run(
            [sys.executable, "-m", "decision_matrix_mcp"],
            cwd=package_path.parent,
            capture_output=True,
            text=True,
            timeout=2,  # Timeout after 2 seconds
        )
    except subprocess.TimeoutExpired:
        # Server started successfully and ran until timeout
        return

    # If we get here, the process exited early
    # Check that no Python errors occurred
    assert "Traceback" not in result.stderr
    assert "ImportError" not in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_main_file_execution():
    """Test that __main__.py fails gracefully when run directly"""
    # Get the path to __main__.py
    main_file = Path(__file__).parent.parent / "src" / "decision_matrix_mcp" / "__main__.py"

    # Run the file directly
    result = subprocess.run([sys.executable, str(main_file)], capture_output=True, text=True)

    # When run directly, __main__.py will fail with relative import error
    # This is expected behavior for a package module
    assert result.returncode != 0
    assert "ImportError" in result.stderr or "from . import main" in result.stderr
