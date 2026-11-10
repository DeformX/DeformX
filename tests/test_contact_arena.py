import json
import subprocess
import sys
import numpy as np
import pytest


def test_last_world_must_participate_in_contact():
    from demo.contact_arena.model import contact_summary
    counts = np.array([12, 13, 0])
    result = contact_summary(counts, 20)
    assert not result['every_world_contacted']
    assert result['active_worlds'] == 2
    assert result['minimum_contact_fraction'] == 0
    assert contact_summary(np.array([5, 20]), 20)['minimum_contact_fraction'] == .25


def test_scene_and_guide_have_independent_world_coordinates():
    from demo.contact_arena.model import build, guide_at, clearance
    s = build(65)
    np.testing.assert_array_equal(s['positions'][0], s['positions'][-1])
    assert s['pins'].sum() == 130
    assert np.isfinite(guide_at(s, 3)).all()
    assert clearance(np.array([[[0., 0., .552]]]), np.array([[0., 0., .36]])).min() > -.00001


def test_lean_help_and_invalid_inputs_preserve_outputs(tmp_path):
    command = [sys.executable, '-S', '-m', 'demo.contact_arena.run']
    help_result = subprocess.run(command + ['--help'], capture_output=True, text=True)
    assert help_result.returncode == 0, help_result.stderr
    for extra in (['--seconds', 'nan'], ['--worlds', '0'], ['--measure-steps', '1']):
        result = subprocess.run(command + ['--out', str(tmp_path/'new')] + extra, capture_output=True, text=True)
        assert result.returncode != 0
        assert not (tmp_path/'new').exists()
    marker = tmp_path/'keep'; marker.write_text('preserved')
    result = subprocess.run(command + ['--out', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0 and marker.read_text() == 'preserved'


def test_replay_rejects_retimed_or_modified_frame(tmp_path):
    from demo.contact_arena.render import validate_capture
    from demo.contact_arena.run import sha, write
    data=tmp_path/'data'; data.mkdir(); out=tmp_path/'frames'; out.mkdir()
    np.savez(data/'trajectory.npz',t=np.array([2.,2.+1/30]))
    write(data/'report.json',dict(complete=True,contact_qualified=True,trajectory_sha256=sha(data/'trajectory.npz')))
    frames=[]
    for i,t in enumerate([2.,2.+1/30]):
        f=out/f'{i}.png'; f.write_bytes(b'fixture')
        frames.append(dict(file=f.name,trajectory_index=i,actual_t_s=t,sha256=sha(f)))
    report=dict(complete=True,errors=[],app_close_invoked=True,capture_clock_delta_max_s=0,amplitude_scale=1,
        trajectory_sha256=sha(data/'trajectory.npz'),run_report_sha256=sha(data/'report.json'),frames=frames)
    write(out/'native_report.json',report); validate_capture(data,out)
    report['frames'][-1]['actual_t_s']+=.1; write(out/'native_report.json',report)
    with pytest.raises(ValueError,match='timestamp'): validate_capture(data,out)
    report['frames'][-1]['actual_t_s']-=.1; write(out/'native_report.json',report)
    (out/'1.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'): validate_capture(data,out)


def test_comparison_rejects_material_change_disguised_as_refinement():
    import copy
    from demo.contact_arena.analyze import matching_workload
    nominal=dict(settings=dict(out='one',worlds=64,substeps=80,EA_N=3000,contact='on'),
        backend_sha256='same',criteria_sha256='same',gpu_name='GPU',host='host',source_sha256={'run.py':'a','model.py':'b'})
    refined=copy.deepcopy(nominal); refined['settings'].update(out='two',substeps=160)
    matching_workload(nominal,refined,('substeps',))
    refined['settings']['EA_N']=6000
    with pytest.raises(ValueError,match='workload'): matching_workload(nominal,refined,('substeps',))
    refined['settings']['EA_N']=3000;refined['gpu_name']='another GPU'
    with pytest.raises(ValueError,match='gpu_name'): matching_workload(nominal,refined,('substeps',))


def test_native_entry_preserves_existing_report(tmp_path):
    data=tmp_path/'data'; data.mkdir(); out=tmp_path/'out'; out.mkdir()
    (data/'trajectory.npz').write_bytes(b'not used by rejected input')
    (data/'report.json').write_text('{"complete":false}')
    marker=out/'native_report.json'; marker.write_text('{"preserve":true}')
    command=[sys.executable,'-m','demo.contact_arena.render','--native','--data',str(data),
             '--assets',str(tmp_path/'assets'),'--out',str(out)]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode!=0
    assert marker.read_text()=='{"preserve":true}'
