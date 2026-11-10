import subprocess
import sys
import pytest


def test_catalog_works_without_site_packages():
    base = [sys.executable, '-S', '-m', 'demo.examples']
    result = subprocess.run(base + ['--list'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for name in ['hanging', 'endpoints', 'obstacle', 'self-contact', 'materials', 'tip-target', 'braid', 'winch', 'contact-arena', 'guitar', 'rack-insertion', 'cascade']:
        assert name in result.stdout
    result = subprocess.run(base + ['--info', 'tip-target'], capture_output=True, text=True)
    assert result.returncode == 0 and 'reset' in result.stdout
    assert 'demo.basics.task' in result.stdout and 'demo/basics/README.md' in result.stdout
    result = subprocess.run(base + ['not-a-case'], capture_output=True, text=True)
    assert result.returncode != 0 and 'Unknown example' in result.stderr


@pytest.mark.parametrize('name',['hanging','endpoints','obstacle','self-contact','materials','tip-target','braid','winch','contact-arena','guitar','rack-insertion','cascade'])
def test_all_delegated_help_without_site_packages(name):
    result=subprocess.run([sys.executable,'-S','-m','demo.examples',name,'--help'],capture_output=True,text=True)
    assert result.returncode==0, result.stderr
    assert 'usage:' in result.stdout.lower() and '--out' in result.stdout


@pytest.mark.parametrize('name,runtime,required_option',[
    ('guitar','CUDA','--assets'),
    ('rack-insertion','Isaac Lab','--profile'),
    ('cascade','CUDA','--out'),
])
def test_showcase_info_describes_runtime_and_required_inputs(name,runtime,required_option):
    result=subprocess.run([sys.executable,'-S','-m','demo.examples','--info',name],
                          capture_output=True,text=True)
    assert result.returncode==0, result.stderr
    assert runtime in result.stdout and required_option in result.stdout
    assert 'README.md' in result.stdout
