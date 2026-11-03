"""CPU-only harness tests: no torch import, GPU access or dependency installation."""
import contextlib
import io
import json
import unittest
from unittest import mock
import benchmark_lsa as b
class HarnessTests(unittest.TestCase):
    def args(self,*args):
        return b.parser().parse_args(args)
    def test_plan_no_import(self):
        with mock.patch.object(b,'load_reference',side_effect=AssertionError('import')), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(b.main([]),0)
        self.assertEqual(json.loads(out.getvalue())['status'],'planned_only')
    def test_gpu_double_optin(self):
        for argv in (['--mode','gpu'],['--allow-gpu']):
            with mock.patch.object(b,'load_reference',side_effect=AssertionError('import')),contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(b.main(argv),1)
            self.assertEqual(json.loads(out.getvalue())['error']['type'],'ValueError')
    def test_bounds(self):
        for argv in (['--contexts','1049088'],['--contexts','131073'],['--contexts','131072','131072'],['--query-batch','129'],['--max-gpu-mib','8193'],['--max-seconds','1801'],['--repeats','0'],['--warmups','0'],['--mode','gpu','--allow-gpu','--contexts','1024']):
            with self.subTest(argv=argv),self.assertRaises(ValueError):
                b.validate(self.args(*argv))
    def test_score_bound(self):
        p=b.plan(self.args())
        self.assertEqual(p['largest_score_matrix_elements'],64*1048576//8)
        self.assertEqual(p['query_batches'],[1,64])
    def test_corrupt_selection(self):
        good=[0,1,2,3,-1]
        self.assertTrue(b.check_rows([good],[4],[3]))
        for row,n in [([0,1,2,4,-1],4),([0,1,1,3,-1],4),([0,1,2,3,0],4),([0,1,3,-1,-1],3),(good,-1),(good,6)]:
            with self.subTest(row=row,n=n),self.assertRaises(AssertionError):
                b.check_rows([row],[n],[3])
    def test_deadline(self):
        with mock.patch.object(b.time,'monotonic',side_effect=[100,101]):
            timer=b.Deadline(1)
            with self.assertRaises(TimeoutError):
                timer.check()
    def test_timings(self):
        self.assertEqual(b.summary([3,1,2])['median_ms'],2)
        for values in ([],[float('nan')],[-1],[float('inf')]):
            with self.assertRaises(ValueError):
                b.summary(values)
    def test_failure_not_pass(self):
        with mock.patch.object(b,'cpu_smoke',side_effect=RuntimeError('broken oracle')),contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(b.main(['--mode','cpu']),1)
        self.assertEqual(json.loads(out.getvalue())['status'],'failed')
    def test_qualification_failure(self):
        def failed(a,r):
            r.update(status='qualification_failed',gpu_executed=True)
        with mock.patch.object(b,'gpu_run',side_effect=failed),contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(b.main(['--mode','gpu','--allow-gpu']),1)
        self.assertEqual(json.loads(out.getvalue())['status'],'qualification_failed')
if __name__=='__main__':
    unittest.main()
