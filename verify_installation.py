"""Verify AstridML installation and dependencies."""

import sys


def check_imports():
    """Check that all required packages can be imported."""
    print("Checking imports...")
    errors = []

    packages = [
        ("numpy", "NumPy"),
        ("pandas", "pandas"),
        ("scipy", "SciPy"),
        ("sklearn", "scikit-learn"),
        ("tensorflow", "TensorFlow"),
        ("fastapi", "FastAPI"),
        ("pydantic", "Pydantic"),
    ]

    for module_name, display_name in packages:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            print(f"  ✗ {display_name}: {e}")
            errors.append(display_name)

    return errors


def check_astridml():
    """Check that AstridML modules can be imported."""
    print("\nChecking AstridML modules...")
    errors = []

    modules = [
        ("astridml.sdg", "Synthetic Data Generator"),
        ("astridml.dpm", "Data Preprocessing Module"),
        ("astridml.models", "ML Models"),
        ("astridml.api", "API"),
    ]

    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            print(f"  ✗ {display_name}: {e}")
            errors.append(display_name)

    return errors


def check_version():
    """Check Python version."""
    print("\nChecking Python version...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("  ✗ Python 3.9+ required")
        return False

    print("  ✓ Python version OK")
    return True


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("AstridML Installation Verification")
    print("=" * 60)

    version_ok = check_version()
    import_errors = check_imports()
    module_errors = check_astridml()

    print("\n" + "=" * 60)

    if not version_ok or import_errors or module_errors:
        print("INSTALLATION INCOMPLETE")
        if import_errors:
            print(f"\nMissing dependencies: {', '.join(import_errors)}")
            print("Run: uv pip install -e '.[dev]'")
        if module_errors:
            print(f"\nMissing modules: {', '.join(module_errors)}")
        sys.exit(1)
    else:
        print("INSTALLATION SUCCESSFUL!")
        print("\nYou can now:")
        print("  - Run example.py to see the pipeline in action")
        print("  - Run tests with: pytest")
        print("  - Start API with: uvicorn astridml.api.server:app")

    print("=" * 60)


if __name__ == "__main__":
    main()
