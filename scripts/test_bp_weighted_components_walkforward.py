#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

def parse_args():
    p=argparse.ArgumentParser(description='Walk-forward weighted BP component test.')
    p.add_argument('--input',required=True); p.add_argument('--output',required=True)
    p.add_argument('--min-train-samples',type=int,default=50)
    return p.parse_args()

def main():
    a=parse_args(); source=Path(a.input); target=Path(a.output)
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from backtest_exact_excel_formulas import load_rows, calculate
    rows=load_rows(source); result=calculate(rows); n=len(result); split=n//2
    components={
      'Trend_Up':('bull','trend_up'),'Trend_Down':('bear','trend_down'),
      'K_Hammer':('bull','K'),'L_InvertedHammer':('bull','L'),'M_DragonflyDoji':('bull','M'),
      'N_BullishMarubozu':('bull','N'),'O_BullishEngulfing':('bull','O'),'P_BullishHarami':('bull','P'),
      'Q_PiercingLine':('bull','Q'),'R_TweezerBottom':('bull','R'),'S_MorningStar':('bull','S'),
      'T_ThreeWhiteSoldiers':('bull','T'),'U_BullishOutsideBar':('bull','U'),'V_BullishPinBar':('bull','V'),
      'W_BullishBreakout':('bull','W'),'X_BullishFalseBreakout':('bull','X'),'Y_BullishRetest':('bull','Y'),
      'AF_HangingManLike':('bear','AF'),'AG_ShootingStar':('bear','AG'),'AH_GravestoneDoji':('bear','AH'),
      'AI_BearishMarubozu':('bear','AI'),'AJ_BearishEngulfing':('bear','AJ'),'AK_BearishHarami':('bear','AK'),
      'AL_DarkCloud':('bear','AL'),'AM_TweezerTop':('bear','AM'),'AN_EveningStar':('bear','AN'),
      'AO_ThreeBlackCrows':('bear','AO'),'AP_BearishOutsideBar':('bear','AP'),'AQ_BearishPinBar':('bear','AQ'),
      'AR_BearishBreakdown':('bear','AR'),'AS_BearishFalseBreakdown':('bear','AS'),'AT_BearishRetest':('bear','AT')}
    def is_active(i,kind,key):
        if key=='trend_up': return result[i]['AW_Trend']=='صعودی'
        if key=='trend_down': return result[i]['AW_Trend']=='نزولی'
        return int(result[i][key])==1
    train={name:[0,0] for name in components}
    for i in range(split):
        a0=float(rows[i]['Close']); a1=float(rows[i+1]['Close']); actual=1 if a1>a0 else -1 if a1<a0 else 0
        if not actual: continue
        for name,(kind,key) in components.items():
            if is_active(i,kind,key):
                train[name][0]+=1
                if (1 if kind=='bull' else -1)==actual: train[name][1]+=1
    weights={}; polarity={}; summary=[]
    for name,(samples,correct) in train.items():
        if samples<a.min_train_samples: weights[name]=0.0; polarity[name]=1; continue
        acc=correct/samples; polarity[name]=1 if acc>=.5 else -1; weights[name]=abs(acc-.5)*2
        summary.append({'Component':name,'TrainSamples':samples,'TrainAccuracy_%':f'{acc*100:.4f}','Polarity':'DIRECT' if polarity[name]==1 else 'INVERSE','Weight':f'{weights[name]:.6f}'})
    tc=tt=lc=lt=sc=st=0; details=[]
    for i in range(split,n-1):
        c0=float(rows[i]['Close']); c1=float(rows[i+1]['Close']); actual=1 if c1>c0 else -1 if c1<c0 else 0
        if not actual: continue
        score=0.0; active=[]
        for name,(kind,key) in components.items():
            w=weights.get(name,0.0)
            if w<=0 or not is_active(i,kind,key): continue
            pred=(1 if kind=='bull' else -1)*polarity[name]; score+=w*pred; active.append(name)
        pred=1 if score>0 else -1 if score<0 else 0
        if not pred: continue
        ok=int(pred==actual); tt+=1; tc+=ok
        if pred==1: lt+=1; lc+=ok
        else: st+=1; sc+=ok
        details.append({'Timestamp':rows[i]['Timestamp'],'Current_Close':c0,'Next_Close':c1,'Weighted_Score':f'{score:.8f}','Predicted':pred,'Actual':actual,'Correct':ok,'Active_Components':'|'.join(active)})
    target.parent.mkdir(parents=True,exist_ok=True); dp=target.with_name(target.stem+'_details.csv')
    with target.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['Component','TrainSamples','TrainAccuracy_%','Polarity','Weight']); w.writeheader(); w.writerows(sorted(summary,key=lambda x:float(x['Weight']),reverse=True))
    with dp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['Timestamp','Current_Close','Next_Close','Weighted_Score','Predicted','Actual','Correct','Active_Components']); w.writeheader(); w.writerows(details)
    print(f'input={source}'); print(f'output={target}'); print(f'details={dp}'); print('TEST=WEIGHTED_BP_COMPONENTS_WALKFORWARD'); print(f'rows={n} split={split} min_train_samples={a.min_train_samples}'); print('WEIGHT=2*abs(train_accuracy-50%)'); print('POLARITY=direct if train accuracy >= 50%, inverse otherwise'); print('EVALUATION=second half only'); print('RESULT')
    print(f'all={tc}/{tt} accuracy={tc/tt*100 if tt else 0:.4f}%'); print(f'long={lc}/{lt} accuracy={lc/lt*100 if lt else 0:.4f}%'); print(f'short={sc}/{st} accuracy={sc/st*100 if st else 0:.4f}%')
if __name__=='__main__': main()
