import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import mpl_toolkits.mplot3d.art3d as art3d
matplotlib.use("Qt5Agg")

# create the data for two groups of subjects. The first should have their values within a small circle of 0-5.
# The second group should have their values higher than 5
X=np.append(20*np.random.rand(120,2)-8,10*np.random.rand(20,2)-4,axis = 0 )
X_dist = X[:,0]**2+ X[:,1]**2
sub1 = X[X_dist<30]
sub2 = X[X_dist>45]
X = np.append(sub1,sub2,axis = 0)
y = np.array([1]*len(sub1)+[-1]*len(sub2))
color = ['steelblue']*len(sub1)+['crimson']*len(sub2)

fontbold2 = {'family': 'cambria','size': 18}

spec = matplotlib.gridspec.GridSpec(nrows=7, ncols=4, wspace=0.2, hspace=0.1)

fig = plt.figure()
fig.set_figwidth(14)
ax = fig.add_subplot(spec[1:6, 0])
plt.scatter(X[:,0],X[:,1], c=color, s = 60, edgecolors='black', linewidth=0.7)
ax.set_xlabel("$x_1$", font = fontbold2)
ax.set_ylabel("$x_2$", font = fontbold2)

ax = fig.add_subplot(spec[:, 1:3],projection = '3d')
ax.scatter(X[:,0],X[:,1],X[:,0]**2+X[:,1]**2, c=color, s = 60, edgecolors='black', linewidth = 1)
ax.set_xlabel("$x_1$", font = fontbold2)
ax.set_ylabel("$x_2$", font = fontbold2)
ax.set_zlabel("${x_1}^2+{x_2}^2$", fontfamily = 'Cambria', fontsize = 17)
xx, yy = np.meshgrid(range(-10,10), range(-10,10))
z = np.ones((20,20))*36
ax.plot_surface(xx, yy, z, alpha=0.3, color='black')
# plot the surface mesh
X3d=np.arange(-10,10,1)
Y3d=np.arange(-10,10,1)
X3d,Y3d=np.meshgrid(X3d,Y3d)
Z3d = X3d**2+Y3d**2
surf = ax.plot_surface(X3d,Y3d,Z3d, color='black',alpha = 0,linewidth=0.3,cstride=3,rstride=3,edgecolors='grey')
#plot the circle
circle = matplotlib.patches.Circle((0,0),6, color = 'black', alpha = 0.5,linewidth=2)
ax.add_patch(circle)
art3d.pathpatch_2d_to_3d(circle,z=36,zdir='z')
ax.azim = 60
ax.elev = 10
ax.dist = 7.75
ax.invert_yaxis()
ax.invert_xaxis()
ax.set_box_aspect(aspect=(1,1,1))

ax = fig.add_subplot(spec[1:6, 3])
plt.scatter(X[:,0],X[:,1], c=color, s = 60, edgecolors='black', linewidth=0.7)
ax.set_xlabel("$x_1$", font = fontbold2)
ax.set_ylabel("$x_2$", font = fontbold2)
circle = plt.Circle((0,0),6, color = 'grey', alpha = 0.3)
ax.add_patch(circle)
plt.subplots_adjust(left=0.07, right=0.975, top=0.99, bottom=0.11)

plt.savefig(r'\\klinik.uni-wuerzburg.de\homedir\userdata11\Sawalma_A\data\Documents\GitHub\analysis_examples\svm_intro.png',dpi = 150)

from ucimlrepo import fetch_ucirepo

# fetch dataset
heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets





