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





####################################
# PCA
####################################
import pandas as pd
import numpy as np
import matplotlib
from sklearn.preprocessing import StandardScaler
import scanpy as sc
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

x1 = np.random.random(40)
x2 = np.random.random(40)**2
data = pd.DataFrame({'x1':x1, 'x2':x1 +0.5*x2})

data = pd.DataFrame(StandardScaler().fit_transform(data),columns = data.columns)
data['x2']*=2

# compute the covariance matrix
cov_mat = np.cov(data,rowvar=0)

# find the eigenvalues and eigenvectors of the covariance matrix.
# from: https://builtin.com/data-science/step-step-explanation-principal-component-analysis
# eigenvectors are the directions of axies of most variance (i.e., axes of principle components).
# Eigenvalues are the coefficients attached to eigenvectors, which reflects the amount of variance carried in each PC
values, vectors = np.linalg.eig(cov_mat)
slopes = [vectors.T[np.abs(values)== np.max(np.abs(values))], vectors.T[np.abs(values)== np.min(np.abs(values))]]
slopes = [item[0,1]/item[0,0] for item in slopes]

# projections describe how would the points be projected on the eigenvector, they can be computed by finding where
# they intersect with the eigenvector if a line is projected from each poinjt perpendicular to the eigenvector
# Actually, this is my own understanding based on figures of PCA tutorials, but I will use this (possibly inaccurate)
# description for illustration purposes only (maybe it proves to be exactly what I said, let's see).
# You have to believe me on the formula
data['c1_pro_x'] = slopes[0]*(data['x2'] - data['x1']*slopes[0]) / (1+slopes[0]**2) + data['x1']
data['c1_pro_y'] = data['c1_pro_x'] * slopes[0]
data['c2_pro_x'] = slopes[1]*(data['x2'] - data['x1']*slopes[1]) / (1+slopes[1]**2) + data['x1']
data['c2_pro_y'] = data['c2_pro_x'] * slopes[1]

data['c1_x'] = np.sqrt(data['c1_pro_x']**2+data['c1_pro_y']**2)# distances of projections of the axis of Component #1
data['c1_x'] = np.where(data['c1_pro_x']<0,-1*np.sign(slopes[0])*data['c1_x'],np.sign(slopes[0])*data['c1_x'])

data['c2_x'] = np.sqrt(data['c2_pro_x']**2+data['c2_pro_y']**2)
data['c2_x'] =np.where(data['c2_pro_x']<0,-1*np.sign(slopes[1])*data['c2_x'],np.sign(slopes[1])*data['c2_x'])



image = Image.open(r'\\klinik.uni-wuerzburg.de\homedir\userdata11\Sawalma_A\data\Documents\GitHub\analysis_examples\eye.png')
im_rot45 = image.rotate(45)  # Specify the desired angle
im_rot_m45 = image.rotate(-45)  # Specify the desired angle

fig,(ax1,ax2,ax3) = plt.subplots(figsize = (11,4),ncols = 3)
ax1.scatter(data['x1'],data['x2'],edgecolor = 'black', c = 'steelblue',zorder = 2)
ax1.set_title('Original data')
ax1.set_xlabel('X1', size = 15)
ax1.set_ylabel('X2', size = 15)
ax1.set_xlim(-2.5,3.5)
ax1.set_ylim(-2.5,3.5)

ax2.scatter(data['x1'],data['x2'],edgecolor = 'black', c = 'steelblue',zorder = 2)
ax2.set_title('View #1')
ax2.set_xlabel('X1', size = 15)
ax2.set_xlim(-2.5,3.5)
ax2.set_ylim(-2.5,3.5)
ax2.imshow(im_rot45, extent=(2.5, 3.5, 2.5, 3.5))  # Adjust the extent and alpha as needed

ax3.scatter(data['x1'],data['x2'],edgecolor = 'black', c = 'steelblue',zorder = 2)
ax3.set_title('View #2')
ax3.set_xlabel('X1', size = 15)
ax3.set_xlim(-2.5,3.5)
ax3.set_ylim(-2.5,3.5)
ax3.imshow(im_rot_m45, extent=(1.5, 2.5, -2, -1))
plt.subplots_adjust(left=0.075, right=0.95, top=0.875, bottom=0.175,wspace = 0.25)

fig.savefig(r'\\klinik.uni-wuerzburg.de\homedir\userdata11\Sawalma_A\data\Documents\GitHub\analysis_examples\pca_intro_1.png',dpi = 200)

min_c1, max_c1 = 1.3*np.min(data[['c1_pro_x']]),np.max(data[['c1_pro_x']])*1.3
min_c2, max_c2 = 1.3*np.min(data[['c2_pro_x']]),np.max(data[['c2_pro_x']])*1.3

fig,(ax1,ax2,ax3,ax4) = plt.subplots(nrows = 1,ncols = 4, figsize = (13,4.5))
ax1.scatter(data['x1'],data['x2'],edgecolor = 'black', c = 'steelblue',zorder = 2)
ax1.set_title('Original data')
ax1.set_xlabel('X1', size = 15)
ax1.set_ylabel('X2', size = 15)

ax2.scatter(data['x1'],data['x2'],edgecolor = 'black', c = 'steelblue',zorder = 2)
ax2.plot([min_c1,max_c1],[min_c1*slopes[0],max_c1*slopes[0]], color = 'black',zorder = 1)
ax2.set_title('New axis that maximizes variance\n(AKA principal component #1)')
ax2.set_xlabel('X1', size = 15)

ax3.scatter(data['x1'],data['x2'],edgecolor = 'black', c = 'steelblue',zorder = 2)
ax3.plot([min_c1,max_c1],[min_c1*slopes[0],max_c1*slopes[0]], color = 'black',zorder = 1)
ax3.scatter(data['c1_pro_x'],data['c1_pro_y'], edgecolor = 'black',linewidth = 0.4,c = 'indianred',zorder = 2, s = 30)
for ind in data.index:
    ax3.plot(data.loc[ind,['x1','c1_pro_x']].values,data.loc[ind,['x2','c1_pro_y']].values, color = 'indianred',zorder = 1)
ax3.set_title('Component #1 projection')
ax3.set_xlabel('X1', size = 15)

ax4.scatter(data['x1'],data['x2'],edgecolor = 'black', c = 'steelblue',zorder = 2)
ax4.plot([min_c2,max_c2],[min_c2*slopes[1],max_c2*slopes[1]], color = 'black',zorder = 1)
ax4.scatter(data['c2_pro_x'],data['c2_pro_y'], edgecolor = 'black',linewidth = 0.4,c = 'indianred',zorder = 2, s = 30)
for ind in data.index:
    ax4.plot(data.loc[ind,['x1','c2_pro_x']].values,data.loc[ind,['x2','c2_pro_y']].values, color = 'indianred',zorder = 1)
ax4.set_title('Component #2 projection')
ax4.set_xlabel('X1', size = 15)

plt.subplots_adjust(left=0.07, right=0.975, top=0.875, bottom=0.15)
fig.savefig(r'\\klinik.uni-wuerzburg.de\homedir\userdata11\Sawalma_A\data\Documents\GitHub\analysis_examples\pca_intro_2.png',dpi = 200)


# compute pca
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data[['x1','x2']])


plt.plot(data_pca[:,0],data_pca[:,1],'o')
plt.plot(data['c1_x'],-data['c2_x'],'o')
