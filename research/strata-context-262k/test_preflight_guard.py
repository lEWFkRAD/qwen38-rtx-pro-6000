#!/usr/bin/env python3
"""CPU-only safety tests; all infrastructure reads mocked, no process launches."""
import copy,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import preflight_guard as G

class GuardTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
  self.path=Path(self.tmp.name)/'baseline.json'
  self.hashes={'/etc/onyx-inference/strata-server.json':'a','/etc/onyx-inference/strata.json':'b',
   '/etc/onyx-inference/strata-server.shared-settings.json':'c','/etc/onyx-personal/config.json':'d'}
  self.path.write_text(json.dumps({'production_config_sha256':self.hashes,'recovery':{'attempts':[1,2]},
   'sunshine_user_unit':{'UnitFileState':'enabled'}}))
  inactive={'ActiveState':'inactive','MainPID':'0'}
  self.view={'hashes':dict(self.hashes),'hold':{'kind':'owned-strata-maintenance','version':1,'owner':G.OWNER},
   'production':inactive.copy(),'production_listener':'','competitors':{'other':inactive.copy()},
   'sunshine':{**inactive,'UnitFileState':'enabled'},'attempts':[1,2],'capacity_block':None,
   'host_available_kib':10*1024*1024,'gpu_free_mib':95000,'gpu':[],'requested_processes':{},
   'allowed_executables':{'/opt/engine':'pinned'}}
  self.guard=G.Guard(baseline_path=self.path,snapshot_reader=lambda pids:copy.deepcopy(self.view))
 def check(self,**kwargs):self.guard.check(force=True,**kwargs)
 def test_user_unit_query_retains_show_subcommand(self):
  with patch.object(G.pwd,'getpwnam',return_value=type('U',(),{'pw_uid':1000})()),patch.object(G,'command',return_value='ActiveState=inactive') as read:
   G.unit('sunshine.service','operator')
  argv=read.call_args.args[0]
  index=argv.index('systemctl')
  self.assertEqual(argv[index:index+4],['systemctl','--user','show','sunshine.service'])
 def test_paused_clean_baseline_passes_before_start(self):self.check(before_start=True)
 def test_wrong_hold_owner_refuses(self):
  self.view['hold']['owner']='another'; self.assertRaises(G.Refusal,self.check,before_start=True)
 def test_production_must_be_inactive(self):
  self.view['production']['ActiveState']='active';self.assertRaises(G.Refusal,self.check,before_start=True)
 def test_shared_or_personal_hash_change_refuses(self):
  for key in list(self.hashes)[2:]:
   with self.subTest(key=key):
    saved=self.view['hashes'][key];self.view['hashes'][key]='changed'
    self.assertRaises(G.Refusal,self.check,before_start=True);self.view['hashes'][key]=saved
 def test_listener_or_competitor_refuses(self):
  self.view['production_listener']='LISTEN';self.assertRaises(G.Refusal,self.check,before_start=True)
  self.view['production_listener']='';self.view['competitors']['other']['MainPID']='55'
  self.assertRaises(G.Refusal,self.check,before_start=True)
 def test_sunshine_start_or_enablement_change_refuses(self):
  for field,value in [('ActiveState','active'),('UnitFileState','disabled')]:
   with self.subTest(field=field):
    saved=self.view['sunshine'][field];self.view['sunshine'][field]=value
    self.assertRaises(G.Refusal,self.check,before_start=True);self.view['sunshine'][field]=saved
 def test_recovery_history_or_capacity_latch_change_refuses(self):
  self.view['attempts'].append(3);self.assertRaises(G.Refusal,self.check,before_start=True)
  self.view['attempts']=[1,2];self.view['capacity_block']={'reason':'capacity'}
  self.assertRaises(G.Refusal,self.check,before_start=True)
 def identity(self,pid=500):
  return {'pid':pid,'ppid':os.getpid(),'start_ticks':'100','pgid':500,'sid':500,'uid':os.geteuid(),'exe':'/opt/engine'}
 def bind(self):
  process=type('FakePopen',(),{'pid':500,'poll':lambda self:None})()
  self.guard.bind_engine(type('Engine',(),{'proc':process})())
  self.view['requested_processes']['500']=self.identity()
 def test_owned_direct_session_gpu_passes(self):
  self.bind();self.view['gpu']=[self.identity()];self.check()
 def test_gpu_without_explicit_private_binding_refuses(self):
  self.view['gpu']=[self.identity()];self.assertRaises(G.Refusal,self.check)
 def test_foreign_gpu_session_or_executable_refuses(self):
  self.bind()
  for field,value in [('sid',999),('exe','/unapproved')]:
   with self.subTest(field=field):
    gpu=self.identity();gpu[field]=value;self.view['gpu']=[gpu]
    self.assertRaises(G.Refusal,self.check)
 def test_reused_root_pid_refuses(self):
  self.bind();self.check();self.view['requested_processes']['500']['start_ticks']='101'
  self.assertRaises(G.Refusal,self.check)
 def test_existing_compute_before_start_always_refuses_even_if_ours(self):
  self.bind();self.view['gpu']=[self.identity()]
  self.assertRaises(G.Refusal,self.check,before_start=True)
 def test_low_host_or_gpu_memory_refuses(self):
  self.view['host_available_kib']=3*1024*1024;self.assertRaises(G.Refusal,self.check,before_start=True)
  self.view['host_available_kib']=10*1024*1024;self.view['gpu_free_mib']=79000
  self.assertRaises(G.Refusal,self.check,before_start=True)

if __name__=='__main__':unittest.main()
