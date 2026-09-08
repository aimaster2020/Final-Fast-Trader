import argparse,csv,statistics
from collections import defaultdict
from pathlib import Path
SYMS=('BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT'); MONTHS=('2026-05','2026-06','2026-07','2026-08'); MIN_BODY=0.0001

def rows(p,s):
 o=[]
 with open(p,encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   if r.get('symbol')==s:
    try:o.append((r['month'],int(float(r['timestamp'])),float(r['open']),float(r['high']),float(r['low']),float(r['close'])))
    except:pass
 return sorted(o,key=lambda x:x[1])

def S(r):
 _,_,o,h,l,c=r;j=c-o;k=h-c;m=h-o;return int(k>j)+int(k>m)+int(l>j)

def audit(rs,mo):
 a=[r for r in rs if r[0]==mo]; H=defaultdict(list);G=[];diff=0;n=0;D=[];SV=defaultdict(list)
 pos=None;ei=-1
 for i in range(len(a)-1):
  r=a[i]; s=S(r); body=abs(r[5]-r[2]); valid=r[5]!=0 and body/abs(r[5])>=MIN_BODY
  gr=statistics.median(G) if G else 2.0; sr=statistics.median(H[s]) if H[s] else gr
  if pos is None:
   n+=1;diff+=sr!=gr;D.append(sr-gr);SV[s].append(sr);pos=(1 if s in(2,3) else -1,r[5],body*sr,body*sr*.5);ei=i
  side,en,tp,sl=pos
  if i>ei and ((side==1 and (r[5]>=en+tp or r[5]<=en-sl)) or (side==-1 and (r[5]<=en-tp or r[5]>=en+sl))):pos=None;ei=-1
  if valid:
   ratio=abs(a[i+1][5]-r[5])/body;H[s].append(ratio);G.append(ratio)
 return n,diff,SV,D

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',default='reports/prepared_price_action_1h.csv');p=Path(ap.parse_args().input)
 print('EXCEL_TPSL_AUDIT | exact S0-S3 | past_only | state_median vs global_median')
 for s in SYMS:
  te=td=0;sv=defaultdict(list);ds=[];print('\n'+s)
  for m in MONTHS:
   n,d,v,x=audit(rows(p,s),m);te+=n;td+=d;ds+=x
   for k,z in v.items():sv[k]+=z
   print(f'{m} entries={n} different={d} diff_pct={(100*d/n if n else 0):.1f}%')
  print(f'4M entries={te} different={td} diff_pct={(100*td/te if te else 0):.1f}% mean_abs_ratio_diff={statistics.mean(abs(x) for x in ds) if ds else 0:.3f}')
  for k in range(4):
   if sv[k]:print(f'  S{k} mean_selected_ratio={statistics.mean(sv[k]):.3f}')
if __name__=='__main__':main()
