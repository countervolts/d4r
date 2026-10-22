"""Arithmetic proof and fail-closed checks for exact prefetch division lowering."""
import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[1] / 'engine/rewrite_k_output_prefetch.py'
spec = importlib.util.spec_from_file_location('k_output_prefetch', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(count=3):
    statements = [f'    r{i} = uint(int(r{100+i}) / int(r7));' for i in range(count)]
    return 'void main() {\n' + '\n'.join(statements) + '\n    f1 = fma(f2, f3, f4);\n}\n'


def test_exhaustive_integer_quotient_proof():
    for divisor in range(1, 25):
        for numerator in range(1024):
            assert module.quotient(numerator, divisor) == numerator // divisor


def test_only_divisions_change_and_original_fallback_remains():
    source = fixture()
    transformed = module.rewrite(source)
    assert transformed.count('uint(D4R_PREFETCH_DIV(') == 3
    assert 'return int(n) / int(d);' in transformed
    assert 'if(n<1024u && d>=1u && d<=24u)' in transformed
    body = transformed[transformed.index('void main()'):]
    restored = module.re.sub(r'uint\(D4R_PREFETCH_DIV\((r\d+), r7\)\)',
                             r'uint(int(\1) / int(r7))', body)
    assert restored == source


@pytest.mark.parametrize('count', [0, 2, 4])
def test_unexpected_shader_shape_fails_closed(count):
    with pytest.raises(ValueError):
        module.rewrite(fixture(count))


def test_reapplying_or_ambiguous_entry_fails_closed():
    with pytest.raises(ValueError):
        module.rewrite(module.rewrite(fixture()))
    with pytest.raises(ValueError):
        module.rewrite(fixture() + 'void main() {}')


@pytest.mark.parametrize('numerator,divisor', [(-1, 1), (1024, 1), (0, 0), (1, 25)])
def test_proof_helper_rejects_outside_domain(numerator, divisor):
    with pytest.raises(ValueError):
        module.quotient(numerator, divisor)
