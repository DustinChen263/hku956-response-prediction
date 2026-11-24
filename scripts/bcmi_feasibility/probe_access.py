from pathlib import Path
import urllib.request,json
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'reports/bcmi_feasibility/sources'
checks=[]
for ds in ['ds002724','ds002722']:
 tree=json.loads((S/f'{ds}_tree.json').read_text())['tree']
 paths=[next(x['path'] for x in tree if x['path'].endswith('.edf')),next(x['path'] for x in tree if x['path'].startswith('stimuli/') and x['path'].endswith('.wav') and '__MACOSX' not in x['path'])]
 for path in paths:
  url=f'https://s3.amazonaws.com/openneuro.org/{ds}/{path}'
  try:
   req=urllib.request.Request(url,headers={'Range':'bytes=0-16383'})
   with urllib.request.urlopen(req,timeout=40) as f:b=f.read(16384);status=f.status;hdr=dict(f.headers)
   dest=S/ds/(path+'.header');dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(b)
   checks.append({'dataset':ds,'path':path,'url':url,'status':status,'headers':hdr,'downloaded_bytes':len(b)})
   print(ds,path,status,len(b),hdr.get('Content-Range'),b[:16],flush=True)
  except Exception as e:checks.append({'dataset':ds,'path':path,'url':url,'error':str(e)});print(str(e),flush=True)
(S/'access_probe.json').write_text(json.dumps(checks,indent=2))
