"""Fig 2: Switch-aware adaptive adversary vs. Moving Target Defense (final).
Clean vector (matplotlib) -> figures/threat_model_adaptive.pdf/png. Nothing cropped."""
import os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Ellipse, Rectangle
for f in ['Regular','Bold','Italic']:
    p=f'/usr/share/fonts/truetype/liberation2/LiberationSerif-{f}.ttf'
    if os.path.exists(p): fm.fontManager.addfont(p)
plt.rcParams.update({'font.family':'Liberation Serif','mathtext.fontset':'stix'})

NAVY='#3E5A78'; NAVYD='#2E4763'; STEEL='#5B7A9A'; LGRAY='#E9EDF2'; MGRAY='#9AA5B1'
GRAYF='#8f9aa6'; PEACH='#D9B48A'; BAND='#A9B6C8'; DARK='#222b36'; WHITE='#ffffff'
GBOX='#D3DAE2'; GBORD='#7a8694'; GREYT='#6b7684'

fig,ax=plt.subplots(figsize=(10.2,5.0)); ax.set_xlim(0,102); ax.set_ylim(0,50); ax.axis('off')

def rbox(x,y,w,h,fc=WHITE,ec=DARK,lw=1.4,rad=0.6,z=2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={rad}',fc=fc,ec=ec,lw=lw,zorder=z))
def txt(x,y,s,size=12,c=DARK,w='normal',ha='center',va='center',z=6,style='normal'):
    ax.text(x,y,s,fontsize=size,color=c,fontweight=w,ha=ha,va=va,zorder=z,style=style)
def arrow(x1,y1,x2,y2,c=NAVY,lw=3.2,ms=22,z=3,rad=0.0):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=ms,lw=lw,color=c,
                 zorder=z,connectionstyle=f'arc3,rad={rad}',shrinkA=0,shrinkB=0))

# ---- title band ----
ax.add_patch(Rectangle((0,45),102,5,fc=BAND,ec='none',zorder=1))
txt(51,47.5,'Switch-aware adaptive adversary vs. Moving Target Defense',size=17,w='bold',c='#1c2836')

# ================= LEFT: adversary (icon only, well-filled) =================
rbox(3,11,25,30,fc='#EDF1F6',ec=NAVY,lw=1.7,rad=1.0)
txt(15.5,43.0,'Adaptive Adversary',size=14.5,w='bold')
def hood(cx,cy,s):
    ax.add_patch(Polygon([[cx-6.2*s,cy-9.4*s],[cx-5.4*s,cy-3.0*s],[cx+5.4*s,cy-3.0*s],[cx+6.2*s,cy-9.4*s]],closed=True,fc=GRAYF,ec='none',zorder=4))
    ax.add_patch(Ellipse((cx,cy+2.0*s),6.4*s,8.4*s,fc=GRAYF,ec='none',zorder=4))
    ax.add_patch(Ellipse((cx,cy+2.6*s),4.3*s,5.7*s,fc='#e6e9ee',ec='none',zorder=5))
    ax.add_patch(Circle((cx,cy+2.9*s),2.2*s,fc='#b7bfc9',ec='none',zorder=6))
    ax.add_patch(Ellipse((cx,cy-1.1*s),5.0*s,2.4*s,fc=GRAYF,ec='none',zorder=6))
hood(15.5,25.5,1.5)

# ================= RIGHT: defended service =================
rbox(72,11,27,30,fc=LGRAY,ec=MGRAY,lw=1.5,rad=1.0)
txt(85.5,43.0,'Defended Service (MTD)',size=14,w='bold')
rbox(74,36.0,23,3.0,fc='#f4f6f9',ec=MGRAY,lw=1.0,rad=0.5)
ax.add_patch(Circle((76,37.5),0.7,fc='#c7ced7',ec=MGRAY,lw=0.8,zorder=4))
ax.add_patch(FancyBboxPatch((78,37.0),3,1.0,boxstyle='round,pad=0,rounding_size=0.3',fc='#c7ced7',ec=MGRAY,lw=0.8,zorder=4))
txt(90.0,34.0,'Model pool',size=12,w='bold'); txt(90.0,32.2,'$\\{f_1,\\ldots,f_K\\}$',size=12)
for i,off in enumerate([1.6,0.8,0.0]):
    rbox(88.8-off, 22.0-off, 7.0,7.0,fc='#7f93a9' if i<2 else '#6f849c',ec=NAVYD,lw=1.2,rad=0.5,z=3+i)
txt(89.5,25.5,'$f_k$',size=13,c=WHITE,w='bold',z=8)
rbox(74,16,12,11.5,fc=NAVYD,ec='#1b2a3b',lw=1.6,rad=0.6,z=6)
txt(80,23.0,'Deployed',size=12.5,c=WHITE,w='bold',z=9); txt(80,21.1,'model',size=12.5,c=WHITE,w='bold',z=9)
txt(80,18.8,'$f_c$',size=14,c=WHITE,w='bold',z=9)
# switch arrow: POOL -> DEPLOYED (replace f_c with f_k from pool)
arrow(88.2,19.8,86.3,20.5,c=NAVYD,lw=3.2,ms=20,rad=0.45,z=7)
txt(80.5,14.0,'switch',size=12,w='bold'); txt(80.5,12.2,'$f_c \\rightarrow f_k$',size=12.5)
# clock + "every P queries"
ax.add_patch(Circle((90.2,13.6),1.5,fc='none',ec=DARK,lw=1.4,zorder=6))
ax.add_patch(FancyArrowPatch((90.2,13.6),(90.2,14.7),arrowstyle='-',lw=1.2,color=DARK,zorder=7))
ax.add_patch(FancyArrowPatch((90.2,13.6),(91.0,13.6),arrowstyle='-',lw=1.2,color=DARK,zorder=7))
txt(95.5,13.9,'every $P$',size=11,w='bold'); txt(95.5,12.2,'queries',size=11,w='bold')

# ================= query / output arrows =================
arrow(28.2,33.5,71.5,33.5,c=NAVY,lw=3.6,ms=25)
txt(49.8,36.2,'iterative queries $\\mathbf{x}_t$',size=13.5,w='bold')
txt(49.8,34.5,'(repeat until misclassification)',size=11.5,w='bold',c='#33414f')
arrow(71.5,15.0,28.2,15.0,c=STEEL,lw=3.6,ms=25)
txt(49.8,13.1,'output',size=13.5,w='bold')

# ================= center: re-baseline cycle (grey boxes) =================
ox,oy,R=44.5,24.4,5.2
def arc_arrow(a0,a1):
    t=np.radians(np.linspace(a0,a1,40)); xs=ox+R*np.cos(t); ys=oy+R*np.sin(t)
    ax.plot(xs,ys,color=PEACH,lw=6.5,solid_capstyle='round',zorder=3)
    hx,hy=ox+R*np.cos(np.radians(a1)),oy+R*np.sin(np.radians(a1)); tang=np.radians(a1-90); L=1.8; W=1.25
    base=(hx-L*np.cos(tang),hy-L*np.sin(tang)); perp=np.radians(a1)
    ax.add_patch(Polygon([(hx,hy),(base[0]+W*np.cos(perp),base[1]+W*np.sin(perp)),
                          (base[0]-W*np.cos(perp),base[1]-W*np.sin(perp))],closed=True,fc=PEACH,ec='none',zorder=4))
for a0,a1 in [(80,-25),(200,95),(320,215)]: arc_arrow(a0,a1)
txt(ox-0.3,oy+0.9,'(2)',size=11.5,w='bold'); txt(ox-0.3,oy-1.0,'re-baseline',size=10.5,w='bold')
rbox(51.0,22.6,8.9,4.2,fc=GBOX,ec=GBORD,lw=1.2,rad=0.5,z=4)
txt(55.45,25.2,'(1) detect',size=10.5,w='bold'); txt(55.45,23.6,'switch',size=10.5,w='bold')
rbox(39.5,15.2,10.2,3.7,fc=GBOX,ec=GBORD,lw=1.2,rad=0.5,z=4)
txt(44.6,17.05,'(3) continue',size=10.5,w='bold')

# ================= bottom caption (large, colour-coded) =================
cy=6.2
txt(21.5,cy,'Switch timing:',size=13.5,w='bold',c=DARK,ha='left')
txt(38.0,cy,'grey-box = known',size=13.5,w='bold',c=GREYT,ha='left')
txt(58.6,cy,'|',size=13.5,w='bold',c=DARK,ha='left')
txt(60.8,cy,'black-box = must be detected',size=13.5,w='bold',c='#111417',ha='left')

out=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'figures','threat_model_adaptive')
for ext in ('pdf','png'): fig.savefig(out+'.'+ext,dpi=200,bbox_inches='tight')
print('saved',out+'.pdf')
