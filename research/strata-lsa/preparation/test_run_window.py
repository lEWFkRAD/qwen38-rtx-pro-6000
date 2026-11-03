"""CPU-only launcher tests. GPU, service, subprocess and signal actions mocked."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock,patch
import run_window as r

def args(*extra):
    return r.parser().parse_args(list(extra))
def gpu(free=16000,utilization=0,capability='12.0',uuid='GPU-one'):
    return dict(uuid=uuid,name='test',capability=capability,total_mib=97887,free_mib=free,utilization_percent=utilization,sampled_at='test')
class GuardTests(unittest.TestCase):
    def test_plan_no_reads(self):
        with patch.object(r,'command',side_effect=AssertionError('read')),patch.object(r,'execute',side_effect=AssertionError('execute')),contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(r.main([]),0)
        self.assertFalse(json.loads(out.getvalue())['gpu_work_started'])
    def test_window_label(self):
        with self.assertRaises(r.Refusal):
            r.validate_args(args('--execute'))
    def test_resource_bounds(self):
        for key,value in [('--max-gpu-mib','4097'),('--max-gpu-mib','128'),('--max-seconds','1801'),('--query-batch','129'),('--repeats','11')]:
            with self.subTest(key=key,value=value),self.assertRaises(r.Refusal):
                r.validate_args(args(key,value))
    def test_duplicate_contexts(self):
        with self.assertRaises(r.Refusal):
            r.validate_args(args('--contexts','131072','131072'))
    def test_active_stops_before_gpu(self):
        with patch.object(r,'primary_state',return_value='active'),patch.object(r,'gpu_snapshot') as read:
            with self.assertRaises(r.Refusal):
                r.check_window(2048)
            read.assert_not_called()
    def test_unknown_state(self):
        for value in ('','failed','activating','inactive\nactive'):
            with self.subTest(value=value),patch.object(r,'primary_state',return_value=value),self.assertRaises(r.Refusal):
                r.check_window(2048)
    def test_resources_refused(self):
        for snapshot in (gpu(utilization=6),gpu(free=8191),gpu(capability='10.3')):
            with self.subTest(snapshot=snapshot),patch.object(r,'primary_state',return_value='inactive'),patch.object(r,'gpu_snapshot',return_value=snapshot),patch.object(r.time,'sleep'),self.assertRaises(r.Refusal):
                r.check_window(2048)
    def test_three_samples(self):
        with patch.object(r,'primary_state',return_value='inactive') as service,patch.object(r,'gpu_snapshot',return_value=gpu()) as read,patch.object(r.time,'sleep') as sleep:
            self.assertEqual(len(r.check_window(2048)),3)
        self.assertEqual(read.call_count,3)
        self.assertEqual(service.call_count,2)
        self.assertEqual(sleep.call_count,2)
    def test_identity_change(self):
        with patch.object(r,'primary_state',return_value='inactive'),patch.object(r,'gpu_snapshot',side_effect=[gpu(),gpu(uuid='other')]),patch.object(r.time,'sleep'),self.assertRaises(r.Refusal):
            r.check_window(2048)
    def test_primary_restart(self):
        with patch.object(r,'primary_state',side_effect=['inactive','active']),patch.object(r,'gpu_snapshot',return_value=gpu()),patch.object(r.time,'sleep'),self.assertRaises(r.Refusal):
            r.check_window(2048)
    def test_invalid_gpu_rows(self):
        for raw in ('','GPU,n,12.0,100,200,0','GPU,n,12.0,100,50,N/A','GPU,n,12.0,100,50,0\nGPU2,n,12.0,100,50,0'):
            with self.subTest(raw=raw),patch.object(r,'command',return_value=raw),self.assertRaises(r.Refusal):
                r.gpu_snapshot()
    def test_valid_gpu_parse(self):
        with patch.object(r,'command',return_value='GPU-1, RTX PRO 6000, 12.0, 97887, 90000, 0'):
            self.assertEqual(r.gpu_snapshot()['free_mib'],90000)
    def test_command_errors(self):
        bad=subprocess.CompletedProcess(['x'],1,'','denied')
        with patch.object(r.subprocess,'run',return_value=bad),self.assertRaises(r.Refusal):
            r.command(['x'])
        with patch.object(r.subprocess,'run',side_effect=subprocess.TimeoutExpired('x',1)),self.assertRaises(r.Refusal):
            r.command(['x'])
    def test_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaises(r.Refusal):
                r.within(root,root/'..'/'escape')
    def test_dirty_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch.object(r,'command',return_value='wrong'),self.assertRaises(r.Refusal):
                r.source_provenance(root)
            with patch.object(r,'command',side_effect=[r.PIN,' M src/a.py']),self.assertRaises(r.Refusal):
                r.source_provenance(root)
    def test_non_linux(self):
        with patch.object(r.platform,'system',return_value='Windows'),patch.object(r,'kit_lock') as lock,self.assertRaises(r.Refusal):
            r.execute(Path('.'),args('--execute','--window-label','test'))
        lock.assert_not_called()
    def test_result_requires_gpu_and_success(self):
        self.assertTrue(r.valid_result(dict(status='passed',mode='gpu',gpu_executed=True)))
        for value in ({},[],dict(status='failed',mode='gpu',gpu_executed=True),dict(status='passed',mode='cpu',gpu_executed=False),dict(status='planned_only',mode='gpu',gpu_executed=True)):
            self.assertFalse(r.valid_result(value))
class WorkerTests(unittest.TestCase):
    def test_success_no_kill(self):
        proc=Mock(pid=123,returncode=0)
        proc.wait.return_value=0
        with tempfile.TemporaryDirectory() as temp,patch.object(r.subprocess,'Popen',return_value=proc) as spawn,patch.object(r,'kill_owned_group') as kill:
            result=r.run_child(['python','x'],env={},cwd=temp,log_path=Path(temp)/'worker.log',timeout_seconds=10)
        self.assertEqual(result['returncode'],0)
        self.assertFalse(result['timed_out'])
        self.assertTrue(spawn.call_args.kwargs['start_new_session'])
        self.assertNotIn('shell',spawn.call_args.kwargs)
        kill.assert_not_called()
    def test_timeout_own_group(self):
        proc=Mock(pid=321,returncode=-9)
        proc.wait.side_effect=subprocess.TimeoutExpired('worker',10)
        with tempfile.TemporaryDirectory() as temp,patch.object(r.subprocess,'Popen',return_value=proc),patch.object(r,'kill_owned_group') as kill,patch.object(r.time,'monotonic',side_effect=[100,100,111]):
            result=r.run_child(['python','x'],env={},cwd=temp,log_path=Path(temp)/'worker.log',timeout_seconds=10)
        self.assertTrue(result['timed_out'])
        kill.assert_called_once_with(proc)
    def test_interrupt_own_group(self):
        proc=Mock(pid=321)
        proc.wait.side_effect=KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as temp,patch.object(r.subprocess,'Popen',return_value=proc),patch.object(r,'kill_owned_group') as kill:
            with self.assertRaises(KeyboardInterrupt):
                r.run_child(['python','x'],env={},cwd=temp,log_path=Path(temp)/'worker.log',timeout_seconds=10)
        kill.assert_called_once_with(proc)
    def test_cleanup_descendants(self):
        proc=Mock(pid=987)
        with patch.object(r.os,'killpg',create=True) as kill:
            r.kill_owned_group(proc)
        self.assertEqual([call.args[0] for call in kill.call_args_list],[987,987])
    def test_primary_definition_missing(self):
        with patch.object(r,'command',return_value='LoadState=not-found\nActiveState=inactive'),self.assertRaises(r.Refusal):
            r.primary_state()
    def test_worker_aborts_on_restart_or_unreadable_state(self):
        for failure in ('active',r.Refusal('state unreadable')):
            proc=Mock(pid=555,returncode=-15)
            proc.wait.side_effect=subprocess.TimeoutExpired('worker',2)
            state_patch=patch.object(r,'primary_state',side_effect=failure) if isinstance(failure,Exception) else patch.object(r,'primary_state',return_value=failure)
            with self.subTest(failure=str(failure)),tempfile.TemporaryDirectory() as temp,patch.object(r.subprocess,'Popen',return_value=proc),patch.object(r,'kill_owned_group') as kill,patch.object(r.time,'monotonic',return_value=100),state_patch:
                result=r.run_child(['python','x'],env={},cwd=temp,log_path=Path(temp)/'worker.log',timeout_seconds=10)
            self.assertTrue(result['aborted'])
            self.assertFalse(result['timed_out'])
            kill.assert_called_once_with(proc)
if __name__=='__main__':
    unittest.main()
