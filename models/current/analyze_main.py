"""读取新模型，计算解释、运行情景和排放因子比较，不再调整模型。"""
from pathlib import Path
import json, sys, warnings, importlib.util
import numpy as np
import pandas as pd
import joblib
from train_model import CatBoostAdapter, TARGETS, PARTS, score
warnings.filterwarnings('ignore',message='X does not have valid feature names')
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True)
data=pd.read_csv(ROOT/'results_rate_only/daily_inputs.csv');data['date']=pd.to_datetime(data.date)
d=data[data.plant_id.eq('RE')].copy()
m=pd.read_csv(ROOT/'results_rate_only/model_inputs.csv');m['date']=pd.to_datetime(m.date)
tr=m[m.plant_id.eq('RE') & m.split.eq('train')].reset_index(drop=True)
te=m[m.plant_id.eq('RE') & m.split.eq('test')].reset_index(drop=True)
pack=joblib.load(ROOT/'results_rate_only/RE/model.joblib')
s=pack['summary'];rng=np.random.default_rng(42)
cols=pack['branches']['total']['features'];pipe=pack['branches']['total']['pipeline']
base=np.maximum(pipe.predict(te[cols]),0);base_rmse=score(te.y_total,base)['RMSE']
imp=[]
for c in cols:
    vals=[]
    for _ in range(5):
        x=te[cols].copy();x[c]=rng.permutation(x[c].to_numpy())
        vals.append(score(te.y_total,np.maximum(pipe.predict(x),0))['RMSE']-base_rmse)
    imp.append({'feature':c,'RMSE_increase':np.mean(vals),'sd':np.std(vals,ddof=1)})
imp=pd.DataFrame(imp).sort_values('RMSE_increase',ascending=False)
imp.to_csv(OUT/'permutation_importance.csv',index=False,encoding='utf-8-sig')
# 部分依赖以训练记录为背景。色带为背景样本预测分布，不是置信区间。
bg=tr[cols].sample(n=min(400,len(tr)),random_state=42);rows=[]
for c in imp.feature.head(6):
    for v in np.linspace(tr[c].quantile(.05),tr[c].quantile(.95),25):
        x=bg.copy();x[c]=v;p=np.maximum(pipe.predict(x),0)
        rows.append({'feature':c,'value':v,'mean':np.mean(p),'P10':np.quantile(p,.1),'P90':np.quantile(p,.9)})
pd.DataFrame(rows).to_csv(OUT/'partial_dependence.csv',index=False,encoding='utf-8-sig')
annual=[]
for yr,g in d.groupby(d.date.dt.year):
    annual.append({'year':yr,'days':len(g),'flow_m3':g.flow_m3_d.sum(),
      **{b+'_t':g['E_'+b].sum()/1000 for b in TARGETS},
      'flow_weighted_intensity':g.E_total.sum()/g.flow_m3_d.sum(),'mean_intensity':g.y_total.mean()})
pd.DataFrame(annual).to_csv(OUT/'annual_inventory.csv',index=False,encoding='utf-8-sig')
ef=[]
for f in [.0106,.035]:
    n=d.E_n2o*f/.0106;tot=d.E_total-d.E_n2o+n
    ef.append({'N2O_EF':f,'total_t':tot.sum()/1000,'n2o_t':n.sum()/1000,
       'electricity_t':d.E_electricity.sum()/1000,'ch4_t':d.E_ch4.sum()/1000,
       'external_co2_t':d.E_external_co2.sum()/1000,'mean_intensity':np.mean(tot/d.flow_m3_d),
       'weighted_intensity':tot.sum()/d.flow_m3_d.sum(),'n2o_share_pct':100*n.sum()/tot.sum()})
pd.DataFrame(ef).to_csv(OUT/'emission_factor_comparison.csv',index=False,encoding='utf-8-sig')
# 报告原始统计量和模型实际用到的不同特征，不人为规定变量或数据点总数。
stats=[]
for c in sorted(set(sum([z['features'] for z in pack['branches'].values()],[]))):
    x=m[m.plant_id.eq('RE')][c]
    stats.append({'feature':c,'nonmissing':x.notna().sum(),'missing':x.isna().sum(),
      'mean':x.mean(),'sd':x.std(),'min':x.min(),'P50':x.median(),'max':x.max()})
pd.DataFrame(stats).to_csv(OUT/'feature_statistics.csv',index=False,encoding='utf-8-sig')
print('Main model explanations and fixed-factor uncertainty comparison saved.')
