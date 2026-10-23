"""Regenerate reviewer-requested displays from full-precision archived tables."""
from pathlib import Path
import os,json,shutil
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-r1-all')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
from paths import ROOT as A, ANALYSIS_ROOT as ROOT, BOUNDARY_ROOT as cache, table_path
F=ROOT/'figures';F.mkdir(parents=True,exist_ok=True)
C=['BeiJing','ShangHai','HongKong','Moscow','StPetersburg','Kazan','Yekaterinburg'];N=['Beijing','Shanghai','Hong Kong','Moscow','St Petersburg','Kazan','Yekaterinburg'];ALG=['Infomap','LeidenRBC','DCSBM','GeoLP','Spinglass','DirLou']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
master=pd.read_csv(table_path('results_summary/master_results.csv'));spatial=pd.read_csv(table_path('results_summary/spatial_compactness_summary.csv'))
def heat(ax,df,col,title,vmin=None,vmax=None,fmt='.2f',cmap='viridis'):
 vals=df.pivot(index='city',columns='algorithm',values=col).reindex(index=C,columns=ALG).to_numpy();im=ax.imshow(vals,aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax)
 ax.set_xticks(range(6),ALG,rotation=15,ha='right');ax.set_yticks(range(7),N);ax.set_title(title,loc='left',fontweight='bold')
 lo=np.nanmin(vals) if vmin is None else vmin; hi=np.nanmax(vals) if vmax is None else vmax
 for (i,j),v in np.ndenumerate(vals):ax.text(j,i,format(v,fmt),ha='center',va='center',fontsize=9,color='white' if (v-lo)/(hi-lo or 1)<.55 else 'black')
 plt.colorbar(im,ax=ax,pad=.02,fraction=.035)
fig,axes=plt.subplots(3,1,figsize=(8.2,9),layout='constrained')
for ax,k in zip(axes,['Q','ACT','ACD']):heat(ax,master,k,k+' (descriptive; no common quality ranking)')
fig.savefig(F/'diagnostics.pdf');plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(8.2,6.3),layout='constrained')
for ax,k in zip(axes,['avg_diameter_m','avg_intra_dist_m']):
 df=spatial.copy();df[k]/=1000;heat(ax,df,k,('Community diameter' if 'diameter' in k else 'Mean pairwise distance')+' (km)',fmt='.2f')
fig.savefig(F/'spatial.pdf');plt.close(fig)
poi=pd.read_csv(table_path('results_summary/poi_master.csv'));fig,ax=plt.subplots(figsize=(8.2,3.5),layout='constrained');heat(ax,poi,'poi_entropy_w','POI-weighted normalized entropy (full theoretical range)',0,1);fig.savefig(F/'entropy.pdf');plt.close(fig)
rep=pd.read_csv(ROOT/'representation_comparisons_flagged.csv');fig,axes=plt.subplots(3,1,figsize=(8.2,9),layout='constrained')
for ax,comp in zip(axes,['endpoint__l_space','endpoint__p_space','l_space__p_space']):
 df=rep[rep.comparison==comp];heat(ax,df,'nmi',comp.replace('__',' / ').replace('_space','').title(),0,1)
 for _,r in df[df.degenerate].iterrows():
  j=ALG.index(r.algorithm); i=C.index(r.city)
  ax.annotate('*',xy=(j,i),xytext=(8,-6),textcoords='offset points',ha='left',va='center',color='red',weight='bold',fontsize=15)
fig.savefig(F/'representation.pdf');plt.close(fig)
from plot_metric_correlations import main as plot_metric_correlations
plot_metric_correlations()
from plot_within_method_scatter import main as plot_within_method_scatter
plot_within_method_scatter()
# Map reference: cached administrative outlines and geographic graticules.
# Administrative outlines come from the configured offline boundary cache.
fnames=['Beijing_China','Shanghai_China','Hong_Kong_China','Moscow_Russia','Saint_Petersburg_Russia','Kazan_Russia','Yekaterinburg_Russia']
for city,name,base in zip(C,N,fnames):
 arr=np.load(A/'graphs'/f'{city}__Endpoint.npz');coords=arr['coordinates'];lat0=coords[:,0].mean();nodeidx={n:i for i,n in enumerate(arr['nodes'])}
 geo=json.loads((cache/(base+'_boundary.geojson')).read_text())['features'][0]['geometry'];polys=geo['coordinates'] if geo['type']=='MultiPolygon' else [geo['coordinates']]
 fig,axes=plt.subplots(3,2,figsize=(8.2,10.6),layout='constrained')
 for ax,alg in zip(axes.flat,ALG):
  part=np.load(A/'partitions/legacy'/f'{city}__{alg}.npz');mp=dict(zip(part['nodes'],part['labels']));ids=sorted(set(mp.values()));col={v:i for i,v in enumerate(ids)}
  # Permuted hues avoid immediate adjacent repetition; IDs are not cross-panel classes.
  colours=plt.colormaps['turbo'](np.mod(np.arange(len(ids))*.61803398875,1))
  for poly in polys:
   xy=np.asarray(poly[0]);ax.plot(xy[:,0],xy[:,1],color='.55',lw=.6,zorder=0)
  ax.scatter(coords[:,1],coords[:,0],c=[col[mp[n]] for n in arr['nodes']],s=1.5,cmap=ListedColormap(colours),rasterized=True,linewidths=0,vmin=0,vmax=max(1,len(ids)-1))
  # Geographic axes and labelled numeric communities supplement nominal colour.
  for label in ids:
   ind=np.array([nodeidx[n] for n in part['nodes'][part['labels']==label]])
   centre=np.median(coords[ind],axis=0)
   if len(ids)<=30 or len(ind)>=len(coords)*.02:ax.text(centre[1],centre[0],str(col[label]+1),fontsize=5.5,ha='center',va='center',bbox={'facecolor':'white','alpha':.75,'edgecolor':'none','pad':.2})
  ax.set_aspect(1/np.cos(np.radians(lat0)));ax.grid(color='.8',lw=.3,alpha=.5);ax.tick_params(labelsize=7)
  ax.set_title(f'{alg} | K={len(ids)}',fontsize=10);ax.set_xlabel('Longitude (°E)',fontsize=8);ax.set_ylabel('Latitude (°N)',fontsize=8)
  ax.annotate('N',xy=(.93,.95),xytext=(.93,.8),xycoords='axes fraction',ha='center',fontsize=8,arrowprops={'arrowstyle':'->','lw':1})
  xmin,xmax=ax.get_xlim();ymin,ymax=ax.get_ylim();width_km=(xmax-xmin)*111.195*np.cos(np.radians(lat0));km=next((k for k in [50,20,10,5,2,1] if k<width_km*.3),1);dx=km/(111.195*np.cos(np.radians(lat0)))
  x=xmin+(xmax-xmin)*.06;y=ymin+(ymax-ymin)*.05;ax.plot([x,x+dx],[y,y],color='black',lw=2);ax.text(x+dx/2,y+(ymax-ymin)*.02,f'{km} km',ha='center',fontsize=7)
 fig.suptitle(name+' | colours and numbers identify local partition labels',fontsize=12)
 fig.text(0.99,0.005,'Legend: each colour denotes a community label local to its panel; numbered labels identify communities.',ha='right',va='bottom',fontsize=8,bbox={'facecolor':'white','alpha':.9,'edgecolor':'.5','pad':3})
 fig.savefig(F/f'map_{city}.pdf',dpi=250);plt.close(fig);print('Map',city,flush=True)
print('Figures complete',flush=True)
