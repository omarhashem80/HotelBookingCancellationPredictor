import sys
import pytest

def run_tests():
    print("Running tests via pytest...")
    sys.exit(pytest.main(["tests"]))

if __name__ == "__main__":
    run_tests()