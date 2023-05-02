import matplotlib.pyplot as plt
import os
def get_value_from_log(log_path,model_name:str,mui:str):
    with open(log_path,"r") as f:
        datas=f.readlines()
    Rs,Ps,Fs=[],[],[]
    for line in datas:
        if line.find(model_name)!=-1 and line.find(mui)!=-1:
            R = float(line.split("R:")[1][:5])
            P = float(line.split("P:")[1][:5])
            F = float(line.split("F:")[1][:5])
            Rs.append(R)
            Ps.append(P)
            Fs.append(F)
    return Rs,Ps,Fs

fig=plt.figure(figsize=(8,4))
fontsize=13
ax=fig.add_subplot(111)
xz=[0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]

logsize=30


abspath=r"C:\Users\djc\Desktop\OCR背景攻击\OCR论文所有数据\eval_scale_log\{}.log".format(logsize)
R9,P9,F9=get_value_from_log(abspath,model_name='craft',mui="mui:0.09")
R12,P12,F12=get_value_from_log(abspath,model_name='craft',mui="mui:0.12")

# R9=[0]*len(R9)
# P9=[0]*len(R9)
# R12=[0]*len(R9)
# P12=[0]*len(R9)

ax.set_xticks([0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0])
ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
ax.set_xticklabels(["0.6", "0.7","0.8","0.9", "1.0","1.1","1.2",
                    "1.3","1.4","1.5","1.6","1.7","1.8","1.9","2.0"], fontsize=fontsize)
ax.set_yticklabels(["0","0.2","0.4","0.6","0.8","1.0"],fontsize=fontsize)
#plt.xlim(0,15)
plt.ylim(0,1)

R9[0]=R9[0]*0.9

p1,=plt.plot(xz,R9,c='lightcoral',linewidth=2,linestyle="-",marker="*",markersize=fontsize-5)
p2,=plt.plot(xz,P9,c='green',linewidth=2,marker="<",markersize=fontsize-5)
#p3,=plt.plot(xz,F9,c='lightcoral',linewidth=2,marker="^",markersize=fontsize-5)

p21,=plt.plot(xz,R12,c='lightcoral',linewidth=2,linestyle="-.",marker="*",markersize=fontsize-3)
#p21,=plt.plot(xz,R12,c='cornflowerblue',linewidth=2,linestyle="-.",marker="*",markersize=fontsize-3)
#p22,=plt.plot(xz,P12,c='limegreen',linewidth=2,linestyle="-.",marker="<",markersize=fontsize-3)
#p23,=plt.plot(xz,F12,c='lightcoral',linewidth=2,linestyle="-.",marker="^",markersize=fontsize-3)
p22,=plt.plot(xz,P12,c='green',linewidth=2,linestyle="-.",marker="<",markersize=fontsize-3)
#p23,=plt.plot(xz,F12,c='lightcoral',linewidth=2,linestyle="-.",marker="^",markersize=fontsize-3)

#plt.legend([p1, p2,p3], [r"R",r"P",r"F"], loc='upper center',fontsize=fontsize)
# plt.legend([p1, p2,p21,p22,], [r"$R^d/R^c$--MUI=0.09",r"$P^d/P^c$--MUI=0.09",r"$R^d/R^c$--MUI=0.12",
#                                r"$P^d/P^c$--MUI=0.12"],
#            loc='upper center',fontsize=fontsize)




plt.xlabel("Scaling factor",fontsize=fontsize+5)
plt.title(r"Patch size={}$\times${}".format(logsize,logsize),fontsize=fontsize+5)


plt.tight_layout()
plt.savefig(os.path.join(abspath,"fig",
                         '{}_show.pdf'.format(logsize)), bbox_inches='tight', dpi=800)
plt.show()