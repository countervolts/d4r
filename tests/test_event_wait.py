"""Completion/error contracts for the sleeping CUDA event wait (no GPU needed)."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class EventWaitTests(unittest.TestCase):
    def test_completion_pending_and_error(self):
        source = r'''
#include "d4r_event_wait.h"
#include <cassert>
int main() {
    int sleeps = 0, queries = 0;
    assert(d4r_wait_event([] { return 0; }, [&] { ++sleeps; }) == 0);
    assert(sleeps == 0);
    assert(d4r_wait_event([&] { return ++queries <= 7 ? 600 : 0; },
                         [&] { ++sleeps; }) == 0);
    assert(queries == 8 && sleeps == 7);
    sleeps = 0;
    assert(d4r_wait_event([] { return 719; }, [&] { ++sleeps; }) == 719);
    assert(sleeps == 0);
    queries = 0;
    assert(d4r_wait_event([&] { return ++queries == 1 ? 600 : 700; },
                         [&] { ++sleeps; }) == 700);
    assert(sleeps == 1);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp) / "wait"
            subprocess.run(["c++", "-x", "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "tools"), "-o", str(runner), "-"],
                           input=source, text=True, check=True, capture_output=True)
            subprocess.run([str(runner)], check=True, timeout=5)
