import argparse,csv
from pathlib import Path
F=['Timestamp','Open','High','Low','Close','Volume']
def n(x):
 try:return float(x)
 except:return None
def load(p,y):
 with open(p,encoding='utf-8-sig',newline='') as f:
  r=list(csv.reader(f))
 h=set(x.lower() for x in r[0]) if r else set()
 if {'timestamp','open','high','low','close'}<=h:
  fs=[x.strip() for x in r[0]]; r=r[1:]
  z=[{k:(x[i] if i<len(x) else '') for i,k in enumerate(fs)} for x in r]
 else:z=[{k:(x[i] if i<len(x) else '') for i,k in enumerate(F)} for x in r]
 return [x for x in z if str(x.get('Timestamp','')).startswith(str(y))]
def run(rows,u,d,m,fee):
 pt=[]
 for i in range(len(rows)-1):
  x=rows[i];o=n(x.get('Open'));c=n(x.get('Close'));p=None
  if i>=5 and o is not None and c is not None:
   w=rows[i-5:i+1];cl=[n(a.get('Close')) for a in w];hi=[n(a.get('High')) for a in w]
   if all(v is not None for v in cl+hi):
    q=sum(cl)/6 if c>o else sum(hi)/6 if c<o else c;z=q-c;p=1 if z>u else -1 if z<-d else 0
  pt.append((c,p))
 pos=0;ep=eb=None;ts=[];forced=0;entries=exits=0
 for i,(c,p) in enumerate(pt):
  if c is None:continue
  if pos and i-eb>=m:
   g=(c-ep)/ep if pos==1 else (ep-c)/ep;ts.append((pos,g,g-2*fee,i-eb,1));exits+=1;forced+=1;pos=0;ep=eb=None
  des=p if p in (-1,1) else 0
  if pos==0:
   if des:pos=des;ep=c;eb=i;entries+=1
  elif des and des!=pos:
   g=(c-ep)/ep if pos==1 else (ep-c)/ep;ts.append((pos,g,g-2*fee,i-eb,0));exits+=1;pos=des;ep=c;eb=i;entries+=1
 gross=sum(t[1] for t in ts);net=sum(t[2] for t in ts)
 return dict(entries=entries,exits=exits,trades=len(ts),wins=sum(t[2]>0 for t in ts),losses=sum(t[2]<=0 for t in ts),gross=gross,commission=(entries+exits)*fee,net=net,forced=forced,open=pos)
def main():
 a=argparse.ArgumentParser();a.add_argument('--input',required=True);a.add_argument('--output',required=True);a.add_argument('--year',type=int,default=2026);a.add_argument('--max-duration',type=int,default=6);a.add_argument('--fee',type=float,default=.0013);a.add_argument('--step',type=int,default=50);a.add_argument('--max-threshold',type=int,default=1000);a=a.parse_args();rows=load(a.input,a.year);R=[]
 for u in range(0,a.max_threshold+1,a.step):
  for d in range(0,a.max_threshold+1,a.step):
   q=run(rows,u,d,a.max_duration,a.fee);q.update(ct_up=u,ct_down=d);R.append(q)
 R.sort(key=lambda x:x['net'],reverse=True);o=Path(a.output);o.parent.mkdir(parents=True,exist_ok=True)
 with o.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=R[0].keys());w.writeheader();w.writerows(R)
 print(f'output={o}\ninput_rows_{a.year}={len(rows)}\ncombinations={len(R)}\nMAX_DURATION={a.max_duration}\n\nTOP 20 BY NET_RETURN_SUM');print('ct_up,ct_down,entries,exits,trades,wins,losses,gross,commission,net,forced,open')
 for q in R[:20]:print(f"{q['ct_up']},{q['ct_down']},{q['entries']},{q['exits']},{q['trades']},{q['wins']},{q['losses']},{q['gross']:.10f},{q['commission']:.10f},{q['net']:.10f},{q['forced']},{q['open']}")
if __name__=='__main__':main()
