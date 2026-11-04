"""CPU-only synthetic contract tests; no service, HTTP or GPU operations."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('draft_contract', ROOT / 'strata-contract.py')
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)


def config(context=262144, streams=8):
    return {'version':1,'unit':'qwen38-flash-next-strata.service','base_url':'http://127.0.0.1:8002',
            'model':'synthetic','aliases':['synthetic'],'context_length':context,'max_streams':streams,
            'user':'synthetic','group':'synthetic','exec_start':['/usr/bin/python3','-m','serve.server'],
            'environment':{'ONYX_STRATA_READINESS_GATE':'1'},'source_commit':'a'*40,
            'hashes':{'/etc/systemd/system/qwen38-flash-next-strata.service':'0'*64},
            'ready_path':'/run/qwen38-flash-next-strata/ready.json'}


def telemetry(context=262144, streams=8):
    h={'status':'ok','service':'strata','loaded':True,'max_context':context,'model':'synthetic',
       'images':True,'api_key':True,'qualified':True}
    m={'engine':{'batch_slots':streams,'context':context,'kv':'int8','expert_slots':24576},
       'live':{'admission_inflight':0,'running':0,'queued':0,'waiting':0,'state':'idle','slots':[]},
       'totals':{'requests':0,'output_tokens':0,'prompt_ms':0,'decode_ms':0,'prompt_tokens':0}}
    s={'context':{'native':context},'concurrency':{'serving':streams}}
    return h,m,s


class ContextContract(unittest.TestCase):
    def load(self, value):
        # Only ownership metadata is mocked; actual synthetic JSON parsing/hashing runs.
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'contract.json';path.write_text(json.dumps(value));path.chmod(0o600)
            original=Path.stat
            def stat(p,*args,**kwargs):
                real=original(p,*args,**kwargs)
                return types.SimpleNamespace(st_uid=0,st_mode=real.st_mode) if p==path else real
            with mock.patch.object(C.Path,'stat',stat):
                return C.load_config(path)

    def test_both_supported_integer_contexts_and_existing_stream_counts(self):
        for context in (131072,262144):
            for streams in (4,8):
                with self.subTest(context=context,streams=streams):
                    self.assertEqual(self.load(config(context,streams))['context_length'],context)

    def test_other_contexts_and_wrong_types_are_rejected(self):
        for context in (65536,196608,524288,0,-1,True,False,131072.0,262144.0,'262144',None):
            with self.subTest(context=context):
                with self.assertRaises(C.ContractViolation): self.load(config(context))

    def test_stream_policy_is_preserved(self):
        for streams in (1,6,16):
            with self.subTest(streams=streams):
                with self.assertRaises(C.ContractViolation): self.load(config(262144,streams))

    def test_matching_health_for_both_contexts(self):
        for context in (131072,262144):
            with self.subTest(context=context):
                result=C.validate_health(config(context),*telemetry(context))
                self.assertEqual(result['context_length'],context)
                self.assertEqual(result['streams']['max'],8)

    def test_engine_context_cannot_silently_fall_back_in_either_direction(self):
        for context,wrong in ((262144,131072),(131072,262144)):
            h,m,s=telemetry(context);m['engine']['context']=wrong
            with self.subTest(context=context):
                with self.assertRaises(C.Unavailable): C.validate_health(config(context),h,m,s)

    def test_health_context_must_also_match(self):
        h,m,s=telemetry();h['max_context']=131072
        with self.assertRaises(C.Unavailable): C.validate_health(config(),h,m,s)

    def test_slots_and_advertised_serving_count_remain_strict(self):
        for field in ('batch_slots','serving'):
            h,m,s=telemetry()
            if field=='batch_slots':m['engine'][field]=4
            else:s['concurrency'][field]=4
            with self.subTest(field=field):
                with self.assertRaises(C.Unavailable): C.validate_health(config(),h,m,s)

    def test_int8_and_all_gpu_experts_remain_required(self):
        h,m,s=telemetry();m['engine']['expert_slots']=24007
        with self.assertRaises(C.CapacityMismatch): C.validate_health(config(),h,m,s)
        h,m,s=telemetry();m['engine']['kv']='fp16'
        with self.assertRaises(C.Unavailable): C.validate_health(config(),h,m,s)

    def test_ready_receipt_still_binds_identity_and_contract_hash(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'ready.json';identity={'pid':123,'engine_pid':456,'boot_id':'synthetic'}
            c={'ready_path':str(path),'_config_sha256':'a'*64}
            good={'version':1,'identity':identity,'config_sha256':'a'*64,'canary':'SIDECAR_OK'}
            path.write_text(json.dumps(good));self.assertEqual(C.verify_ready(c,identity),good)
            for field,value in [('identity',{'pid':999}),('config_sha256','b'*64),('canary','WRONG')]:
                changed=copy.deepcopy(good);changed[field]=value;path.write_text(json.dumps(changed))
                with self.subTest(field=field):
                    with self.assertRaises(C.Unavailable): C.verify_ready(c,identity)


if __name__=='__main__': unittest.main(verbosity=2)
