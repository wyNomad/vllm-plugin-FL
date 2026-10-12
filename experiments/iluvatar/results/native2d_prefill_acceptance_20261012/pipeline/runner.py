import subprocess,json,time,traceback,hashlib,os
from pathlib import Path
personal=Path('/workspace/wyNomad');repo=personal/'vllm-plugin-FL'
root=personal/'results/native2d_prefill_pipeline_20261012'
screen=personal/'results/native2d_prefill_screen_20261012'
expected='197610d877f3bd50231239fef4dd6d1760638c01'
target='7b37dafda546c6e54e695a3db712e4e3a89e6ca8'
branch='perf/iluvatar-native2d-context-routing'
def run(args,timeout=60):
 return subprocess.run(args,check=True,capture_output=True,text=True,timeout=timeout).stdout.strip()
def state(value):
 (root/'current_stage.txt').write_text(value+'\n');print(value,flush=True)
try:
 state('waiting: frozen screen commit '+expected)
 deadline=time.monotonic()+14400
 while not (screen/'run.exit').exists():
  if time.monotonic()>deadline: raise TimeoutError('screen did not finish in 4h')
  time.sleep(10)
 assert not run(['git','-C',str(repo),'status','--porcelain'])
 assert run(['git','-C',str(repo),'rev-parse','HEAD'])==expected
 assert not run(['docker','ps','-a','--filter','name=^wynomad-minicpm-s2-gpu3$','--format','{{.Names}}']), 'personal GPU3 container not cleaned'
 state('screen finished; syncing archive and acceptance driver')
 protected=json.loads((personal/'backups/code-sync-20261011-c1fb81d/before.json').read_text())['protected_sha256']
 def checksum(path):
  return 'symlink:'+os.readlink(path) if path.is_symlink() else hashlib.sha256(path.read_bytes()).hexdigest()
 assert all(checksum(personal/name)==digest for name,digest in protected.items())
 run(['git','-C',str(repo),'fetch',str(personal/'native2d-prefill-acceptance.bundle'),'refs/heads/'+branch+':refs/remotes/origin/'+branch])
 run(['git','-C',str(repo),'merge','--ff-only','origin/'+branch])
 assert run(['git','-C',str(repo),'rev-parse','HEAD'])==target
 assert not run(['git','-C',str(repo),'status','--porcelain'])
 assert all(checksum(personal/name)==digest for name,digest in protected.items())
 (root/'git_sync_verified.json').write_text(json.dumps(dict(branch=branch,head=target,clean=True,protected_files=len(protected)),indent=2)+'\n')
 if (screen/'run.exit').read_text().strip()!='0':
  state('stopped: screen failed; code synced, formal not started')
 else:
  comparison=json.loads((screen/'comparison.json').read_text())
  checks=[row for row in comparison['changes'] if row['case']!='warmup']
  if len(checks)!=2 or not all(row['throughput_change_pct']>=-1 and row['ttft_change_pct']<=1 for row in checks):
   state('stopped: screen performance gate failed; formal not started')
  else:
   state('screen passed; starting Math500 and official four-round baseline/candidate acceptance')
   with (root/'acceptance_supervisor.log').open('w') as out:
    subprocess.run(['python3','-u',str(repo/'experiments/iluvatar/native2d_context/run_acceptance.py'),'--gpu','3'],stdout=out,stderr=subprocess.STDOUT,check=True,timeout=36000)
   state('complete: official acceptance passed')
 (root/'run.exit').write_text('0\n')
except Exception:
 traceback.print_exc();state('failed: inspect supervisor.log; no blind restart')
 (root/'run.exit').write_text('1\n')
