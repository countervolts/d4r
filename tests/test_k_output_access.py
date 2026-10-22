"""Fail-closed provenance/alignment gates for the optional translated K output pass."""
import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[1] / 'engine/rewrite_k_output_access.py'
spec = importlib.util.spec_from_file_location('k_output_access', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    s = '''buffer LutBits { uint16_t bits[]; } lut;
uint PARAM64(uint o) {
 if(o==280u) return 0x80000000u+((param.w[36]>>8)&15u)*32768u;
 return 0u;
}
uint PARAM16(uint o) { return 0u; }
uint G16(uint a) {
 return a>=0x80000000u ? uint(lut.bits[(a-0x80000000u)>>1]) : uint(head.bits[a>>1]);
}
precise float preserve = fma(x, y, z);
'''
    for group, base in enumerate(['rd128', 'rd50']):
        s += f'''        rd{100+group} = uint(PARAM64(280u));
        rd{110+group} = uint(uint(int(int(r99)) * int(16)));
        {base} = uint(rd{100+group} + rd{110+group});
'''
        for i in range(4):
            s += f'        r{1000+10*group+i} = uint(G32({base} + uint({4*i}u)));\n'
    s += '        rd1 = uint(PARAM64(272u));\n        rd2 = uint(PARAM64(272u));\n'
    s += '        r1 = uint(G32(rd1 + uint(0u)));\n        r2 = uint(G32(rd2 + uint(0u)));\n'
    return s


def test_specializes_only_authenticated_records():
    result = module.rewrite(fixture())
    assert result.count('uvec4 block = lut.blocks[') == 2
    assert 'precise float preserve = fma(x, y, z);' in result
    assert 'r1 = uint(G32(rd1 + uint(0u)));' in result
    assert 'return uint(head.bits[a>>1]);' in result


@pytest.mark.parametrize('old,new', [
    ('* int(16)', '* int(8)'),
    ('rd100 = uint(PARAM64(280u))', 'rd100 = uint(PARAM64(272u))'),
    ('0x80000000u', '0x40000000u'),
    ('uint16_t bits[]', 'uint bits[]'),
])
def test_rejects_unproven_address_layouts(old, new):
    with pytest.raises(ValueError):
        module.rewrite(fixture().replace(old, new, 1))


def test_missing_or_extra_access_is_rejected():
    with pytest.raises(ValueError):
        module.rewrite(fixture() + '        r3 = uint(G32(rd3 + uint(0u)));\n')


def test_nonzero_head_base_is_rejected():
    with pytest.raises(ValueError):
        module.rewrite(fixture().replace(' return 0u;', ' if(o==272u) return 32u; return 0u;', 1))


def test_scalar_tagged_access_is_rejected():
    with pytest.raises(ValueError):
        module.rewrite(fixture() + '        rd9 = uint(PARAM64(280u));\n        r9 = uint(G16(rd9 + uint(0u)));\n')
