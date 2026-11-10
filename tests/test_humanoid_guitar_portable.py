import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def invoke(*args):
    return subprocess.run(
        [sys.executable, '-S', '-m', 'demo.humanoid_guitar.run', *args],
        cwd=ROOT, capture_output=True, text=True,
    )


def test_guitar_help_needs_no_site_packages():
    for args in [(), ('prepare',), ('run',), ('audit',), ('render',)]:
        result = invoke(*args, '--help')
        assert result.returncode == 0, result.stderr
        assert 'usage:' in result.stdout
    assert '--out' in invoke('--help').stdout


def test_missing_guitar_assets_fail_before_runtime(tmp_path):
    assets, output = tmp_path / 'missing assets', tmp_path / 'fresh output'
    result = invoke('--assets', str(assets), '--out', str(output))
    assert result.returncode == 2
    assert 'prepare' in result.stderr and 'contact_profile.json' in result.stderr
    assert not output.exists()


def test_existing_guitar_run_is_never_overwritten(tmp_path):
    assets, output = tmp_path / 'assets', tmp_path / 'existing run'
    (assets / 'contact-v4').mkdir(parents=True)
    (assets / 'contact-v4' / 'contact_profile.json').write_text('{}')
    output.mkdir()
    sentinel = output / 'old result'
    sentinel.write_text('keep me')
    result = invoke('--assets', str(assets), '--out', str(output))
    assert result.returncode == 2
    assert 'already exists' in result.stderr
    assert sentinel.read_text() == 'keep me'


def test_render_requires_audit_before_starting_isaac(tmp_path):
    out = tmp_path / 'physics only'
    (out / 'case').mkdir(parents=True)
    (out / 'case' / 'report.json').write_text('{}')
    result = invoke('render', '--out', str(out))
    assert result.returncode == 2
    assert 'run audit --out' in result.stderr
    assert not (out / 'native').exists()


def test_render_refuses_to_overwrite_existing_capture(tmp_path):
    out = tmp_path / 'completed run'
    (out / 'case').mkdir(parents=True)
    (out / 'case' / 'report.json').write_text('{}')
    (out / 'audit.json').write_text('{}')
    (out / 'native').mkdir()
    sentinel = out / 'native' / 'video.mp4'
    sentinel.write_bytes(b'prior media')
    result = invoke('render', '--out', str(out))
    assert result.returncode == 2
    assert 'already exists' in result.stderr
    assert sentinel.read_bytes() == b'prior media'
