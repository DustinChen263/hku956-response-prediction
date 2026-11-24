"""Public read-only downloads, capped header probes and four representative runs."""
from pathlib import Path
import urllib.request,json,hashlib,re,concurrent.futures,time
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'reports/bcmi_feasibility/sources';D=ROOT/'data/raw/bcmi_feasibility'

def fetch(task):
 ds,path,full=task;url=f'https://s3.amazonaws.com/openneuro.org/{ds}/{path}'
 dest=(D/ds/path) if full else (S/ds/(path+'.header'))
 dest.parent.mkdir(parents=True,exist_ok=True)
 pointer=(S/ds/(path+'.annex_pointer')).read_text();match=re.search(r'MD5E-s(\d+)--([0-9a-f]{32})',pointer)
 size=int(match[1]);md5=match[2]
 if dest.exists() and (not full or dest.stat().st_size==size):
  data=dest.read_bytes();return {'dataset':ds,'path':path,'url':url,'full_download':full,'status':'cached','bytes':len(data),'expected_bytes':size,'md5_matches':hashlib.md5(data).hexdigest()==md5 if full else None}
 for attempt in range(3):
  try:
   limit=16384 if path.endswith('.edf') else 4096
   req=urllib.request.Request(url,headers={} if full else {'Range':f'bytes=0-{limit-1}'})
   with urllib.request.urlopen(req,timeout=60) as f:
    status=f.status;hdr=dict(f.headers);data=f.read(size+1 if full else limit)
   if full and (len(data)!=size or hashlib.md5(data).hexdigest()!=md5):raise ValueError('Size/MD5 mismatch')
   dest.write_bytes(data)
   return {'dataset':ds,'path':path,'url':url,'full_download':full,'status':status,'bytes':len(data),'expected_bytes':size,'md5_matches':True if full else None,'content_range':hdr.get('Content-Range')}
  except Exception as e:
   if attempt==2:return {'dataset':ds,'path':path,'url':url,'full_download':full,'status':'error','error':str(e),'expected_bytes':size}
   time.sleep(1)

tasks=[];samples=[]
for ds in ['ds002724','ds002722']:
 tree=json.loads((S/f'{ds}_tree.json').read_text())['tree']
 edfs=sorted(x['path'] for x in tree if x['path'].endswith('.edf'))
 wavs=sorted(x['path'] for x in tree if x['path'].startswith('stimuli/') and x['path'].endswith('.wav') and '__MACOSX' not in x['path'])
 tasks.extend((ds,p,False) for p in edfs+wavs)
 # First and last participant/run, chosen before inspecting signal quality.
 for p in [edfs[0],edfs[-1],wavs[0],wavs[-1]]:samples.append((ds,p,True))
print('Header probes',len(tasks),'full representative files',samples,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 results=list(pool.map(fetch,tasks))
(S/'header_download_log.json').write_text(json.dumps(results,indent=2))
print('Headers done; errors',sum(r['status']=='error' for r in results),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 full=list(pool.map(fetch,samples))
(S/'sample_download_log.json').write_text(json.dumps(full,indent=2))
print(json.dumps(full,indent=2),flush=True)
