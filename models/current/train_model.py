"""按四项排放重新训练PI-DPCH模型。

训练目标由电耗、N2O、CH4和乙酸氧化CO2相加得到。研究边界和旧文件
不同，因此重新计算所有标签，不能沿用旧文件中的总量或模型分数。
脚本只使用预测日前已经获得的运行记录。每个时间验证折单独填补缺失值。
运行示例：python train_model.py --plant RE --output results_RE
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, time, warnings
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore',message='X does not have valid feature names')
warnings.filterwarnings('ignore',message='Skipping features without any observed values')

ROOT = Path(__file__).resolve().parent
SEED = 42
TARGETS = {'total':'y_total', 'electricity':'y_electricity', 'n2o':'y_n2o',
           'ch4':'y_ch4', 'external_co2':'y_external_co2'}
PARTS = list(TARGETS)[1:]
LABELS = {'total':'总量', 'electricity':'电耗', 'n2o':'N₂O', 'ch4':'CH₄', 'external_co2':'外加碳源CO₂'}
BIO = ['days_since_prior_experiment','bio_nitrification_NH4_consumption_rate_mg_g_h',
       'bio_nitrification_NO3_production_rate_mg_g_h','bio_nitrification_PO4_uptake_rate_mg_g_h',
       'bio_denitrification_NO3_consumption_rate_mg_g_h','bio_anaerobic_PO4_release_rate_mg_g_h',
       'bio_nit_den_activity_ratio','bio_duplicate_signature_flag']


class CatBoostAdapter(RegressorMixin, BaseEstimator):
    """让CatBoost可被sklearn逐折复制，防止不同时间折共享已拟合的模型。"""
    def __init__(self, iterations=280, learning_rate=.04, depth=7, l2_leaf_reg=4, random_seed=42, thread_count=4):
        self.iterations=iterations; self.learning_rate=learning_rate; self.depth=depth
        self.l2_leaf_reg=l2_leaf_reg; self.random_seed=random_seed; self.thread_count=thread_count
    def fit(self,X,y):
        self.model_=CatBoostRegressor(iterations=self.iterations, learning_rate=self.learning_rate,
            depth=self.depth,l2_leaf_reg=self.l2_leaf_reg,random_seed=self.random_seed,
            thread_count=self.thread_count,loss_function='RMSE',verbose=False,allow_writing_files=False)
        self.model_.fit(X,y); self.n_features_in_=np.asarray(X).shape[1]; return self
    def predict(self,X): return self.model_.predict(X)


def score(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);den=np.abs(y)+np.abs(p)
    s=np.divide(2*np.abs(y-p),den,out=np.zeros_like(den),where=den>0)
    return {'R2':float(r2_score(y,p)),'RMSE':float(np.sqrt(mean_squared_error(y,p))),
            'MAE':float(mean_absolute_error(y,p)),'sMAPE_pct':float(np.mean(s)*100)}


def bank():
    """六种算法均采用预先固定的参数。测试期不用于调整参数。"""
    m={
      'LightGBM':LGBMRegressor(n_estimators=240,learning_rate=.04,num_leaves=31,
          min_child_samples=20,colsample_bytree=.8,reg_alpha=.05,reg_lambda=.25,
          random_state=SEED,n_jobs=4,verbosity=-1),
      'RandomForest':RandomForestRegressor(n_estimators=140,max_features=.65,
          min_samples_leaf=2,random_state=SEED,n_jobs=4),
      'ExtraTrees':ExtraTreesRegressor(n_estimators=140,max_features=.75,
          min_samples_leaf=1,random_state=SEED,n_jobs=4),
      'HistGradientBoosting':HistGradientBoostingRegressor(max_iter=160,learning_rate=.06,
          max_leaf_nodes=31,min_samples_leaf=20,l2_regularization=.15,random_state=SEED),
      'XGBoost':XGBRegressor(n_estimators=240,learning_rate=.04,max_depth=6,
          min_child_weight=3,subsample=.9,colsample_bytree=.8,reg_alpha=.05,
          reg_lambda=1.,objective='reg:squarederror',random_state=SEED,n_jobs=4),
      'CatBoost':CatBoostAdapter()}
    return {k:Pipeline([('imputer',SimpleImputer(strategy='median')),('model',v)]) for k,v in m.items()}


def fold_indices(d,n=5):
    # 先按日期划分，再取各厂记录；同一日期不会跨入训练和验证两侧。
    dates=np.array(sorted(d.date.unique()))
    for a,b in TimeSeriesSplit(n_splits=n).split(dates):
        yield np.flatnonzero(d.date.isin(dates[a])),np.flatnonzero(d.date.isin(dates[b]))


def oof(pipe,d,cols,target,n=5):
    """用较早日期训练、后续日期验证，返回未见过该条标签时的预测。"""
    p=np.full(len(d),np.nan); detail=[]
    for no,(a,b) in enumerate(fold_indices(d,n),1):
        fitted=clone(pipe).fit(d.iloc[a][cols],d.iloc[a][target])
        q=np.maximum(fitted.predict(d.iloc[b][cols]),0);p[b]=q
        detail.append({'fold':no,'train_start':str(d.iloc[a].date.min().date()),
          'train_end':str(d.iloc[a].date.max().date()),'valid_start':str(d.iloc[b].date.min().date()),
          'valid_end':str(d.iloc[b].date.max().date()),'train_rows':len(a),'valid_rows':len(b),
          **score(d.iloc[b][target],q)})
    return p,detail


def prepare(clean_path,flow_path,out):
    """从原清洗表生成新的四项清单和历史输入；原文件不作任何修改。"""
    d=pd.read_csv(clean_path,encoding='gb18030')
    d.columns=[re.search(r'（([^（）]+)）$',c)[1] if re.search(r'（([^（）]+)）$',c) else c for c in d]
    aliases={v:('CN' if '城北' in str(v) else 'RE' if '江边' in str(v) else str(v)) for v in d.plant_id.unique()}
    d['plant_id']=d.plant_id.map(aliases);d['date']=pd.to_datetime(d.date)
    d=d.sort_values(['plant_id','date']).reset_index(drop=True)
    d['original_flow_m3_d']=d.flow_m3_d
    f=pd.read_csv(flow_path);f['date']=pd.to_datetime(f.date);f=f.set_index('date').net_flow_m3_d
    sel=d.plant_id.eq('RE');d.loc[sel,'flow_m3_d']=d.loc[sel,'date'].map(f).to_numpy()
    assert d.loc[sel,'flow_m3_d'].notna().all() and (d.flow_m3_d>0).all()
    # 单位：m³/d × mg/L × 10^-3 = kg/d。
    for p in ['COD','BOD5','SS','NH4','TN','TP']:
        d[p+'_in_load_kg_d']=d.flow_m3_d*d[p+'_in_mg_L']/1000
        # 去除率保留原始浓度差的方向；异常值不改写为零。分母为零时记缺失。
        d[p+'_removal_rate']=(d[p+'_in_mg_L']-d[p+'_out_mg_L'])/d[p+'_in_mg_L'].replace(0,np.nan)
    d['electricity_kWh_m3']=d.electricity_kWh_d/d.flow_m3_d
    d['dry_sludge_kg_m3']=d.dry_sludge_kg_d/d.flow_m3_d
    d['acetate_kg_m3']=d.acetate_total_kg_d/d.flow_m3_d
    for c in ['sodium_chlorate','sodium_hypochlorite']:
        d[c+'_kg_m3']=d[c+'_kg_d']/d.flow_m3_d
    for k,n,de in [('COD_TN_ratio','COD','TN'),('BOD5_TN_ratio','BOD5','TN'),('COD_NH4_ratio','COD','NH4')]:
        d[k]=d[n+'_in_mg_L']/d[de+'_in_mg_L'].replace(0,np.nan)
    # 本次总量严格等于下面四个分项之和。全球变暖潜能值分别为273和27.9。
    d['E_electricity']=d.electricity_kWh_d*.6
    d['E_n2o']=d.flow_m3_d*(d.TN_in_mg_L-d.TN_out_mg_L).clip(lower=0)/1000*.0106*44/28*273
    d['E_ch4']=d.BOD5_in_load_kg_d*.019*27.9
    d['E_external_co2']=d.acetate_total_kg_d*88/82
    d['E_total']=d[['E_'+c for c in PARTS]].sum(axis=1,min_count=4)
    for b in TARGETS:d[TARGETS[b]]=d['E_'+b]/d.flow_m3_d
    assert np.allclose(d.y_total,d[[TARGETS[b] for b in PARTS]].sum(axis=1,min_count=4),equal_nan=True)
    d['year']=d.date.dt.year;d['month']=d.date.dt.month
    d['doy_sin']=np.sin(2*np.pi*d.date.dt.dayofyear/365.25)
    d['doy_cos']=np.cos(2*np.pi*d.date.dt.dayofyear/365.25)
    base=['flow_m3_d','water_temp_C','rain_flag','mlss_mean_mg_L','dry_sludge_kg_m3',
      *[p+'_in_mg_L' for p in ['COD','BOD5','NH4','TN','TP']],
      *[p+'_out_mg_L' for p in ['COD','NH4','TN','TP']],
      *[p+'_in_load_kg_d' for p in ['COD','BOD5','NH4','TN','TP','SS']],
      'COD_TN_ratio','BOD5_TN_ratio','COD_NH4_ratio','electricity_kWh_m3',
      'acetate_kg_m3','sodium_chlorate_kg_m3','sodium_hypochlorite_kg_m3',
      *[p+'_removal_rate' for p in ['COD','BOD5','NH4','TN','TP','SS']]]
    roots=['flow_m3_d','water_temp_C','mlss_mean_mg_L','electricity_kWh_m3',
      'acetate_kg_m3','sodium_chlorate_kg_m3','sodium_hypochlorite_kg_m3',
      'COD_in_mg_L','NH4_in_mg_L','TN_in_mg_L','TP_in_mg_L',
      'COD_in_load_kg_d','NH4_in_load_kg_d','TN_in_load_kg_d','TP_in_load_kg_d',
      'COD_TN_ratio','BOD5_TN_ratio']
    add={}
    for c in base:add[c+'_lag1']=d.groupby('plant_id')[c].shift(1)
    for c in roots:
        for w in [3,7]:
            add[c+f'_roll{w}_mean']=d.groupby('plant_id')[c].transform(lambda x:x.shift(1).rolling(w,min_periods=w).mean())
    d=pd.concat([d,pd.DataFrame(add,index=d.index)],axis=1)
    audit=[]
    for pl,g in d.groupby('plant_id'):
        for c in ['acetate_total_kg_d','sodium_chlorate_kg_d','sodium_hypochlorite_kg_d']:
            audit.append({'plant':pl,'variable':c,'days':len(g),'available':int(g[c].notna().sum()),
              'positive':int(g[c].gt(0).sum()),'zero':int(g[c].eq(0).sum()),'missing':int(g[c].isna().sum())})
    pd.DataFrame(audit).to_csv(out/'input_coverage.csv',index=False,encoding='utf-8-sig')
    # 不向新数据文件复制其他旧标签、合计输入或未使用的投加字段。
    keep=list(dict.fromkeys(['plant_id','date','year','month','doy_sin','doy_cos',
      'original_flow_m3_d',*base,*BIO,'electricity_kWh_d','acetate_total_kg_d',
      'sodium_chlorate_kg_d','sodium_hypochlorite_kg_d',
      *[p+'_'+a+'_mg_L' for p in ['COD','BOD5','SS','NH4','TN','TP'] for a in ['in','out']],
      *['E_'+b for b in TARGETS],*TARGETS.values(),*add]))
    d=d[keep];d.to_csv(out/'daily_inputs.csv',index=False,encoding='utf-8-sig')
    # 与原稿保持相同的首个建模日期及最后182天测试期。
    d=d[d.groupby('plant_id').cumcount()>=14].reset_index(drop=True)
    missing=d[list(TARGETS.values())].isna().any(axis=1)
    d.loc[missing,['plant_id','date',*TARGETS.values()]].to_csv(out/'excluded_target_rows.csv',index=False,encoding='utf-8-sig')
    excluded=int(missing.sum())
    d=d.loc[~missing].reset_index(drop=True)
    d['split']=np.where(d.date>='2023-07-03','test','train')
    d.to_csv(out/'model_inputs.csv',index=False,encoding='utf-8-sig')
    (out/'input_manifest.json').write_text(json.dumps({'source_sha256':hashlib.sha256(clean_path.read_bytes()).hexdigest(),
       'flow_check_sha256':hashlib.sha256(flow_path.read_bytes()).hexdigest(),
       'scope_components':PARTS,'daily_rows':len(d)+28+excluded,'model_rows':len(d),'missing_target_rows_excluded':excluded,
       'flow_corrected_days_RE':int((pd.read_csv(out/'daily_inputs.csv').query("plant_id == 'RE'").eval('flow_m3_d != original_flow_m3_d')).sum()),
       'test_start':'2023-07-03','seed':SEED},ensure_ascii=False,indent=2),encoding='utf-8')
    return d


def fit_scope(d,plant,out):
    out.mkdir(parents=True,exist_ok=True);start=time.time()
    if plant!='joint':d=d[d.plant_id.eq(plant)].copy()
    d=d.sort_values(['date','plant_id']).reset_index(drop=True)
    tr=d[d.split.eq('train')].reset_index(drop=True);te=d[d.split.eq('test')].reset_index(drop=True)
    if plant=='joint':
        for x in [tr,te]:x['plant_code']=x.plant_id.map({'CN':0,'RE':1})
    allcols=[c for c in d if (c.endswith('_lag1') or re.search(r'_roll[37]_mean$',c)) and 'removed_load' not in c]
    constants=[c for c in allcols if tr[c].nunique(dropna=True)<=1]
    allcols=[c for c in allcols if c not in constants]
    cal=['year','month','doy_sin','doy_cos']+(['plant_code'] if plant=='joint' else [])
    lag=[c for c in allcols if c.endswith('_lag1')]
    sets={'core_lag1':cal+[c for c in lag if 'removed_load' not in c],
          'multiscale':cal+allcols,
          'multiscale_bio':cal+allcols+BIO}
    models=bank();selection=[];feature_rows=[];foldrows=[];selected={};poof={};ptest={};ablation=[]
    predout=te[['plant_id','date','flow_m3_d']].copy();oofout=tr[['plant_id','date']].copy()
    # 第一步只在训练期比较输入组。固定LightGBM，避免将算法差异混入输入组比较。
    for b,y in TARGETS.items():
        choices=sets if b=='n2o' else {k:v for k,v in sets.items() if k!='multiscale_bio'}
        fs={}
        for name,cols in choices.items():
            p,_=oof(models['LightGBM'],tr,cols,y);ok=np.isfinite(p);s=score(tr.loc[ok,y],p[ok]);fs[name]=s['RMSE']
            feature_rows.append({'branch':b,'feature_set':name,'features':len(cols),**s})
            ft=clone(models['LightGBM']).fit(tr[cols],tr[y]);q=np.maximum(ft.predict(te[cols]),0)
            ablation.append({'branch':b,'feature_set':name,'features':len(cols),**{'OOF_'+k:v for k,v in s.items()},**{'TEST_'+k:v for k,v in score(te[y],q).items()}})
        best=min(fs,key=fs.get);cols=sets[best];candidates={}
        for name,pipe in models.items():
            p,details=oof(pipe,tr,cols,y);ok=np.isfinite(p);s=score(tr.loc[ok,y],p[ok])
            selection.append({'branch':b,'feature_set':best,'features':len(cols),'algorithm':name,**s})
            foldrows.extend({'branch':b,'algorithm':name,**a} for a in details);candidates[name]=(s['RMSE'],p)
        win=min(candidates,key=lambda k:candidates[k][0]);p=candidates[win][1]
        pipe=clone(models[win]).fit(tr[cols],tr[y]);q=np.maximum(pipe.predict(te[cols]),0)
        selected[b]={'algorithm':win,'feature_set':best,'features':cols,'pipeline':pipe}
        poof[b]=p;ptest[b]=q;predout['actual_'+b]=te[y];predout['pred_'+b]=q
        oofout['actual_'+b]=tr[y];oofout['pred_'+b]=p
        print(plant,b,win,best,len(cols),'OOF',round(candidates[win][0],5),flush=True)
    total=poof['total'];summ=np.sum([poof[b] for b in PARTS],axis=0);mask=np.isfinite(total)&np.isfinite(summ)
    truth=tr.loc[mask,'y_total'].to_numpy();weights=[]
    for w in np.linspace(0,1,101):weights.append({'weight_total':float(w),'weight_sum':float(1-w),**score(truth,w*total[mask]+(1-w)*summ[mask])})
    ws=pd.DataFrame(weights).sort_values('RMSE');w=float(ws.iloc[0].weight_total)
    sumtest=np.sum([ptest[b] for b in PARTS],axis=0);final=w*ptest['total']+(1-w)*sumtest
    predout['pred_sum']=sumtest;predout['pred_final']=final;oofout['pred_sum']=summ
    oofout['pred_final']=w*total+(1-w)*summ
    # 对照采用非负岭回归。正则强度只用训练期OOF记录内的时间验证选取。
    meta=pd.DataFrame({'date':tr.loc[mask,'date'].to_numpy(),'a':total[mask],'b':summ[mask],'y':truth})
    alpha_scores=[]
    for alpha in [1e-6,1e-4,.001,.01,.1,1,10,100]:
        m=Pipeline([('scale',StandardScaler()),('ridge',Ridge(alpha=alpha,positive=True,solver='lbfgs',max_iter=10000))])
        p,_=oof(m,meta,['a','b'],'y',4);v=np.isfinite(p);alpha_scores.append({'alpha':alpha,**score(meta.loc[v,'y'],p[v])})
    alpha=min(alpha_scores,key=lambda z:z['RMSE'])['alpha']
    ridge=Pipeline([('scale',StandardScaler()),('ridge',Ridge(alpha=alpha,positive=True,solver='lbfgs',max_iter=10000))]).fit(meta[['a','b']],truth)
    predout['pred_ridge']=np.maximum(ridge.predict(pd.DataFrame({'a':ptest['total'],'b':sumtest})),0)
    metrics=[]
    for b in TARGETS:metrics.append({'model':b,**score(te[TARGETS[b]],ptest[b])})
    for k in ['sum','final','ridge']:metrics.append({'model':k,**score(te.y_total,predout['pred_'+k])})
    names={'algorithm_selection':selection,'feature_selection':feature_rows,'fold_metrics':foldrows,'feature_ablation':ablation,'test_metrics':metrics,'ridge_selection':alpha_scores}
    for n,a in names.items():pd.DataFrame(a).to_csv(out/(n+'.csv'),index=False,encoding='utf-8-sig')
    ws.to_csv(out/'weights.csv',index=False,encoding='utf-8-sig');predout.to_csv(out/'test_predictions.csv',index=False,encoding='utf-8-sig');oofout.to_csv(out/'oof_predictions.csv',index=False,encoding='utf-8-sig')
    union=sorted(set(sum([v['features'] for v in selected.values()],[])))
    assert not any('removed_load' in c for c in union)
    summary={'plant':plant,'train_rows':len(tr),'test_rows':len(te),'train_start':str(tr.date.min().date()),
      'train_end':str(tr.date.max().date()),'test_start':str(te.date.min().date()),'test_end':str(te.date.max().date()),
      'weight_total':w,'weight_sum':1-w,'ridge_alpha':alpha,'seed':SEED,
      'feature_groups':{k:len(v) for k,v in sets.items()},'constant_columns_removed':constants,
      'selected':{b:{k:v for k,v in z.items() if k!='pipeline'} for b,z in selected.items()},
      'unique_inputs':len(union),'test_input_datapoints':len(union)*len(te),'metrics':metrics,
      'seconds':time.time()-start}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    joblib.dump({'branches':selected,'weight_total':w,'ridge':ridge,'summary':summary},out/'model.joblib',compress=3)
    print('DONE',plant,'seconds',round(time.time()-start),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plant',choices=['RE','CN','joint','all'],default='RE');p.add_argument('--output',type=Path,default=ROOT/'results')
    p.add_argument('--clean',type=Path,default=Path(r'D:\MJ\资环\碳排核算\模型2.0\清洗后数据\双厂_2019-2023_完整清洗日尺度数据_中文在前表头.csv'))
    p.add_argument('--flow',type=Path,default=ROOT.parent/'jiangbian_performance_load_20261012'/'原始月报水量口径核对.csv')
    p.add_argument('--prepared',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.prepared:
        d=pd.read_csv(a.prepared,float_precision='round_trip');d['date']=pd.to_datetime(d.date)
    else:d=prepare(a.clean,a.flow,a.output)
    for plant in (['RE','CN','joint'] if a.plant=='all' else [a.plant]):fit_scope(d,plant,a.output/plant)

if __name__=='__main__':main()
