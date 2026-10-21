"""固定已保存的四项模型，重新计算三组光伏与进水增长情景。

光伏替代外购电，不改变设备总用电特征；仅将电力排放预测乘(1-r)。
上游控制使进水COD、TN降低5%、15%、25%，处理水量提高相同比例，出水浓度不变。
其他单位水量运行投入保持基准水平，日总量随水量变化，不设减量情景。
全部负荷、去除率和历史窗口重新生成，验证基准能重现保存的测试结果。
"""
from pathlib import Path
import sys,json,hashlib,warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings('ignore',message='X does not have valid feature names')
ROOT=Path(__file__).resolve().parent
MODEL=ROOT
sys.path.insert(0,str(MODEL))
from train_model import CatBoostAdapter,PARTS
R=MODEL/'results_rate_only'
raw=pd.read_csv(R/'daily_inputs.csv',float_precision='round_trip')
raw=raw[raw.plant_id.eq('RE')].copy().sort_values('date').reset_index(drop=True)
raw['date']=pd.to_datetime(raw.date)
inputs=pd.read_csv(R/'model_inputs.csv',float_precision='round_trip')
inputs=inputs[inputs.plant_id.eq('RE')].copy();inputs['date']=pd.to_datetime(inputs.date)
tr=inputs[inputs.split.eq('train')].copy()
te=inputs[inputs.split.eq('test')].copy().sort_values('date').reset_index(drop=True)
pack=joblib.load(R/'RE/model.joblib')
features=sorted(set(c for b in pack['branches'].values() for c in b['features']))
assert pack['weight_total']==0, 'Review the physical PV adjustment if weights change.'

def prepare(increase):
    d=raw.copy();factor=1+increase
    d['flow_m3_d']*=factor
    d['COD_in_mg_L']*=(1-increase);d['TN_in_mg_L']*=(1-increase)
    # 用户确认：氨氮随TN同倍率降低，维持台账氮组成比例，不新增独立情景水平。
    d['NH4_in_mg_L']*=(1-increase)
    # 运行投入按处理水量变化。单位水量投加和总用电强度不变。
    for c in ['electricity_kWh_d','acetate_total_kg_d','sodium_chlorate_kg_d','sodium_hypochlorite_kg_d']:
        d[c]*=factor
    for p in ['COD','BOD5','NH4','TN','TP','SS']:
        d[p+'_in_load_kg_d']=d.flow_m3_d*d[p+'_in_mg_L']/1000
        d[p+'_removal_rate']=(d[p+'_in_mg_L']-d[p+'_out_mg_L'])/d[p+'_in_mg_L'].replace(0,np.nan)
    d['electricity_kWh_m3']=d.electricity_kWh_d/d.flow_m3_d
    d['acetate_kg_m3']=d.acetate_total_kg_d/d.flow_m3_d
    for c in ['sodium_chlorate','sodium_hypochlorite']:d[c+'_kg_m3']=d[c+'_kg_d']/d.flow_m3_d
    for n,a,b in [('COD_TN_ratio','COD','TN'),('BOD5_TN_ratio','BOD5','TN'),('COD_NH4_ratio','COD','NH4')]:
        d[n]=d[a+'_in_mg_L']/d[b+'_in_mg_L'].replace(0,np.nan)
    for c in features:
        if c.endswith('_lag1'):d[c]=d[c[:-5]].shift(1)
        elif '_roll' in c:
            stem,w=c.rsplit('_roll',1);days=int(w.split('_')[0]);d[c]=d[stem].shift(1).rolling(days,min_periods=days).mean()
    return d.set_index('date').loc[te.date].reset_index()

def predict(d,pv):
    p={k:np.maximum(v['pipeline'].predict(d[v['features']]),0) for k,v in pack['branches'].items()}
    before=p['electricity'].copy()
    p['electricity']=p['electricity']*(1-pv)
    p['final']=sum(p[k] for k in PARTS)
    p['without_pv']=p['final']+before*pv
    p['pv_avoided']=before*pv
    return p

base=prepare(0)
assert np.allclose(base[features],te[features],rtol=1e-10,atol=1e-11,equal_nan=True),'Baseline feature reconstruction differs.'
pred0=predict(base,0)
saved=pd.read_csv(R/'RE/test_predictions.csv',float_precision='round_trip')
assert np.allclose(pred0['final'],saved.pred_final,rtol=1e-10,atol=1e-12),'Saved baseline not reproduced.'
base_total=np.sum(pred0['final']*base.flow_m3_d)
base_intensity=base_total/base.flow_m3_d.sum()
records=[];daily=[];components=[];input_records=[];predictions={};cases={}
for code,label,u in [('S0','基准',0),('S1','低',.05),('S2','中',.15),('S3','高',.25)]:
    d=prepare(u);p=predict(d,u);q=d.flow_m3_d.to_numpy()
    total=float(np.sum(p['final']*q));wp=total/q.sum();cases[code]=d;predictions[code]=p
    inv={
      'electricity':d.electricity_kWh_d.to_numpy()*.6*(1-u),
      'n2o':(d.flow_m3_d*(d.TN_in_mg_L-d.TN_out_mg_L).clip(lower=0)/1000).to_numpy()*.0106*44/28*273,
      'ch4':d.BOD5_in_load_kg_d.to_numpy()*.019*27.9,
      'external_co2':d.acetate_total_kg_d.to_numpy()*88/82}
    invtotal=sum(inv.values());invbase=base.E_total.sum()
    lo=tr[features].min();hi=tr[features].max()
    outside=((d[features]<lo)|(d[features]>hi)) & d[features].notna()
    row={'scenario':code,'level':label,'increase_pct':u*100,'influent_decrease_pct':u*100,'flow_increase_pct':u*100,'pv_share_pct':u*100,
      'dates':len(d),'water_m3':float(q.sum()),'mean_flow_m3_d':float(q.mean()),
      'COD_in_weighted_mg_L':float(np.sum(d.COD_in_mg_L*q)/q.sum()),
      'TN_in_weighted_mg_L':float(np.sum(d.TN_in_mg_L*q)/q.sum()),
      'NH4_in_weighted_mg_L':float(np.sum(d.NH4_in_mg_L*q)/q.sum()),
      'TN_below_NH4_days':int((d.TN_in_mg_L<d.NH4_in_mg_L).sum()),
      'COD_removal_pct':float(100*(d.COD_in_load_kg_d*d.COD_removal_rate).sum()/d.COD_in_load_kg_d.sum()),
      'TN_removal_pct':float(100*(d.TN_in_load_kg_d*d.TN_removal_rate).sum()/d.TN_in_load_kg_d.sum()),
      'total_power_kWh_m3':float(d.electricity_kWh_d.sum()/q.sum()),
      'grid_power_kWh_m3':float(d.electricity_kWh_d.sum()*(1-u)/q.sum()),
      'predicted_total_t':total/1000,'predicted_weighted_intensity':wp,
      'predicted_daily_mean_intensity':float(np.mean(p['final'])),
      'intensity_reduction_pct':float(100*(1-wp/base_intensity)),
      'total_change_pct':float(100*(total/base_total-1)),
      'pv_avoided_t':float(np.sum(p['pv_avoided']*q)/1000),
      'without_pv_total_t':float(np.sum(p['without_pv']*q)/1000),
      'inventory_total_t':float(invtotal.sum()/1000),
      'inventory_weighted_intensity':float(invtotal.sum()/q.sum()),
      'inventory_total_change_pct':float(100*(invtotal.sum()/invbase-1)),
      'outside_training_cells_pct':float(100*outside.sum().sum()/d[features].notna().sum().sum()),
      'dates_with_any_outside_training':int(outside.any(axis=1).sum())}
    records.append(row)
    for k in PARTS:components.append({'scenario':code,'component':k,'predicted_t':float(np.sum(p[k]*q)/1000),'inventory_t':float(inv[k].sum()/1000)})
    for i,dt in enumerate(d.date):
        daily.append({'scenario':code,'date':str(dt.date()),'water_m3':float(q[i]),'intensity_kgCO2e_m3':float(p['final'][i]),'predicted_kgCO2e':float(q[i]*p['final'][i]),'without_pv_kgCO2e':float(q[i]*p['without_pv'][i]),'pv_avoided_kgCO2e':float(q[i]*p['pv_avoided'][i]),'inventory_kgCO2e':float(invtotal[i])})
    view=d[['date',*features]].copy();view.insert(0,'scenario',code);input_records.append(view)
    # 保持所有出水浓度和单位水量运行投入不变。
    assert np.allclose(d.electricity_kWh_m3,base.electricity_kWh_m3,equal_nan=True)
    assert np.allclose(d.acetate_kg_m3,base.acetate_kg_m3,equal_nan=True)
    assert np.allclose(d[[p+'_out_mg_L' for p in ['COD','BOD5','NH4','TN','TP','SS']]],base[[p+'_out_mg_L' for p in ['COD','BOD5','NH4','TN','TP','SS']]],equal_nan=True)
    assert np.allclose(p['final'],sum(p[k] for k in PARTS))

# 周为单位配对重复抽样，只描述同一已训练模型下基准期记录差异。
rng=np.random.default_rng(42);assert len(base)==182
indices=np.arange(182).reshape(26,7)
draws=indices[rng.integers(0,26,size=(1000,26))].reshape(1000,182)
for row in records:
    code=row['scenario'];d=cases[code];p=predictions[code]
    q=d.flow_m3_d.to_numpy();qb=base.flow_m3_d.to_numpy()
    totals=(p['final'][draws]*q[draws]).sum(axis=1)
    w=totals/q[draws].sum(axis=1)
    btotals=(pred0['final'][draws]*qb[draws]).sum(axis=1)
    bw=btotals/qb[draws].sum(axis=1)
    row.update({'intensity_P025':float(np.quantile(w,.025)),'intensity_P975':float(np.quantile(w,.975)),
      'intensity_reduction_P025_pct':float(np.quantile(100*(1-w/bw),.025)),
      'intensity_reduction_P975_pct':float(np.quantile(100*(1-w/bw),.975)),
      'total_change_P025_pct':float(np.quantile(100*(totals/btotals-1),.025)),
      'total_change_P975_pct':float(np.quantile(100*(totals/btotals-1),.975))})

summary=pd.DataFrame(records);summary.to_csv(ROOT/'scenario_summary.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(daily).to_csv(ROOT/'scenario_daily.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(components).to_csv(ROOT/'scenario_components.csv',index=False,encoding='utf-8-sig')
pd.concat(input_records).to_csv(ROOT/'scenario_model_inputs.csv',index=False,encoding='utf-8-sig')
audit={'baseline_reproduces_saved_test':True,'outflow_concentrations_fixed':True,'operating_intensities_fixed':True,
 'pv_only_scales_electricity_component':True,'COD_TN_direction':'decrease due to upstream control','NH4_direction':'same proportional decrease as TN, authorized by user','flow_direction':'increase','levels_pct':[5,15,25],'test_dates':['2023-07-03','2023-12-31'],
 'bootstrap':'1000 paired resamples of 26 non-overlapping 7-day blocks, seed 42; conditional on saved model',
 'model_sha256':hashlib.sha256((R/'RE/model.joblib').read_bytes()).hexdigest(),
 'data_sha256':hashlib.sha256((R/'daily_inputs.csv').read_bytes()).hexdigest()}
(ROOT/'scenario_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'scenario_summary.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print(summary[['scenario','predicted_total_t','predicted_weighted_intensity','intensity_reduction_pct','total_change_pct','pv_avoided_t','COD_removal_pct','TN_removal_pct','grid_power_kWh_m3','outside_training_cells_pct']].to_string(index=False))
