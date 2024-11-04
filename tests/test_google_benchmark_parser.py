#!/usr/bin/env python3
from cryptanalysislib.parsers.parse_google_benchmark import read_google_benchmark_data

def test_simple():
    t = "tests/google_benchmark_test_data.json"
    a = read_google_benchmark_data(t)
    assert len(a) > 1
    for b in a:
        assert int(b["size"]) >= 32

    assert True

