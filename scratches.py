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

#!mkdir data
#!wget http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz -O data/pbmc3k_filtered_gene_bc_matrices.tar.gz
#!cd data; tar -xzf pbmc3k_filtered_gene_bc_matrices.tar.gz
#!mkdir write

from utils import *
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
from sklearn.preprocessing import MaxAbsScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


x1 = np.random.random(40)
x2 = np.random.random(40)**2
data = pd.DataFrame({'x1':x1, 'x2':x1 +0.5*x2})

data = pd.DataFrame(StandardScaler().fit_transform(data),columns = data.columns)
data['x2']*=2

# compute the covariance matrix
cov_mat = np.cov(data,rowvar=0)

# from: https://builtin.com/data-science/step-step-explanation-principal-component-analysis
# eigenvectors are the directions of axies of most variance (i.e., axes of principle components).
# Eigenvalues are the coefficients attached to eigenvectors, which reflects the amount of variance carried in each PC
# Geometrically speaking, principal components represent the directions of the data that explain a maximal amount of
# variance, that is to say, the lines that capture most information of the data.

# find the eigenvalues and eigenvectors of the covariance matrix.
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


image = Image.open('eye.png')
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

#fig.savefig(r'\\klinik.uni-wuerzburg.de\homedir\userdata11\Sawalma_A\data\Documents\GitHub\analysis_examples\pca_intro_1.png',dpi = 200)

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
#fig.savefig(r'\\klinik.uni-wuerzburg.de\homedir\userdata11\Sawalma_A\data\Documents\GitHub\analysis_examples\pca_intro_2.png',dpi = 200)


# compute pca
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data[['x1','x2']])

fig,ax = plt.subplots(figsize = (5,4))
ax.scatter(data_pca[:,0],data_pca[:,1],edgecolor = 'black',s = 45)
ax.set_xlabel('PC1', size =15)
ax.set_ylabel('PC2', size =15)
ax.set_ylim(-4,4)
plt.subplots_adjust(left=0.15, right=0.95, top=0.875, bottom=0.175,wspace = 0.25)

fig.savefig(r'C:\Users\Sawalma_A\Documents\GitHub\analysis_examples\pcs_plot.png',dpi = 200)



# Data is optained from: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0151982
# Citation:Knaster, Peter et al. (2017). Data from: Diagnosing depression in chronic pain patients: DSM-IV Major Depressive Disorder vs. Beck Depression Inventory (BDI) [Dataset]. Dryad. https://doi.org/10.5061/dryad.14955

# PCA on depression data
data = pd.read_csv('dep_pain.csv')
data_scaled = StandardScaler().fit_transform(data.iloc[:,1::])
data = pd.DataFrame(data_scaled,columns = data.columns[1::])


# First, determine the best number of components
n_comp = data.shape[1]
pca_temp = PCA(n_components=n_comp)
data_pca_temp = pca_temp.fit_transform(data)
plot_pca(pca_temp)

n_comp = 2
pca = PCA(n_components=n_comp)
data_pca = pca.fit_transform(data)

comp = get_components(pca,or_cols = data.columns, plot_result= True, text_threshold=0.35)
comp



################################
# RNA Seq
################################
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')
results_file = 'write/pbmc3k.h5ad'  # the file that will store the analysis results

adata = sc.read_10x_mtx(
    'data/filtered_gene_bc_matrices/hg19/',  # the directory with the `.mtx` file
    var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
    cache=True)


df = adata.to_df()

adata.var_names_make_unique()
sc.pl.highest_expr_genes(adata, n_top=20, )
adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'

adata = adata[adata.obs.n_genes_by_counts < 2500, :]
adata = adata[adata.obs.pct_counts_mt < 5, :]

sc.tl.pca(adata, svd_solver='arpack')
sc.tl.pca()
sc.pl.pca(adata, color='CST3')



ran_d = pd.read_csv(r'C:\Users\Sawalma_A\Downloads\E-GEOD-19268-A-AFFY-2-normalized-expressions.tsv', delimiter = '\t')






# Data from:
# Zhang Y, Tong GH, Wei XX, Chen HY et al. Identification of Five Cytotoxicity-Related Genes Involved in the Progression of Triple-Negative Breast Cancer. Front Genet 2021;12:723477. PMID: 35046993
gendat = pd.read_csv('https://github.com/mmilano87/PCAPxDEG/raw/main/dataset/GSE183947_fpkm.csv')
gendat.index = gendat['Unnamed: 0'].values
gendat = gendat.drop('Unnamed: 0',axis = 1)
gendat = gendat.T
subgroup = np.array(['Control' if item.startswith('CAP') else 'Cancer' for item in list(gendat.index)])

# MaxAbsScaler scales every feature based on its maximum absolute value
trans_gendat = MaxAbsScaler().fit_transform(gendat)
trans_gendat = pd.DataFrame(trans_gendat, index = gendat.index, columns=gendat.columns)

# determine best number of components
pca = PCA(n_components=trans_gendat.shape[0])
pca_data = pca.fit_transform(trans_gendat)

plot_pca(pca)

n_comp = 10

pca = PCA(n_components=n_comp)
pca_data = pca.fit_transform(trans_gendat)
pca_data = pd.DataFrame(pca_data, columns = ['PCA'+str(i+1) for i in range(n_comp)])
pca_data.index =gendat.index

plot_pca(pca)
pca_comp = get_components(pca,or_cols=trans_gendat.columns, plot_result=True,text_threshold=0.005, max_plot_feature=10, max_plot_comp=5)

colors = np.where(subgroup=='Control','steelblue','indianred')

fig = plt.figure()
fig.set_figwidth(12)
ax = fig.add_subplot(121,projection = '3d')

ax.scatter(pca_data['PCA1'],pca_data['PCA3'],pca_data['PCA2'], c=colors, s = 60, edgecolors='white', linewidth = 1)
ax.set_xlabel('PCA1');ax.set_ylabel('PCA3');ax.set_zlabel('PCA2')
for group,color in zip(np.unique(subgroup),['indianred','steelblue']):
    ax.scatter(pca_data.loc[subgroup==group,'PCA1'].values[0],pca_data.loc[subgroup==group,'PCA2'].values[0],
               pca_data.loc[subgroup==group,'PCA3'].values[0],s=60,edgecolors='white',c=color, label = group)
plt.legend()

ax2 = fig.add_subplot(122)
ax2.scatter(pca_data['PCA1'],pca_data['PCA3'],c = colors, edgecolor = 'white',s = 60)
ax2.set_xlabel('PCA1');ax2.set_ylabel('PCA3')
for group,color in zip(np.unique(subgroup),['indianred','steelblue']):
    ax2.scatter(pca_data.loc[subgroup==group,'PCA1'].values[0],pca_data.loc[subgroup==group,'PCA2'].values[0],
                c=color, label = group)
plt.legend()


# determine the genese with highest and lowest expressions
th_90 = np.percentile(pca_data['PCA3'],90)
th_10 = np.percentile(pca_data['PCA3'],10)

pca_data['PCA3']>th_90








# Differentially expressed genes. I got those from this paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8762060/
inc_genes_1 = ['COL11A1','CXCL10','KPNA2','PLK1','KIFC1','E2F1','FANCA','RAD54L','FOXM1','MYBL2','HIST2H3C','HIST1H2A',
               'HIST1H2AB','TROAP','HIST1H1B','HIST1H2AH','HIST1H3B','HIST1H2BL','KIF2C','CCNB1','CBX2','TMEM132A','HN1','TK1','H2AFX']
inc_genes_2 = ['LPHN3','MAB21L1','FAT4','RUNX1T1','SEMA6A','TSHZ2','RAI2','CACNA1G','COL4A6','GFRA1','ARHGAP6',
               'PGM5', 'ABCA10','ABCA9','ABCA8','ALDH1A2','SPTBN4','FLG2','DES','SYNPO2','MYH11','PRDM16','MYOCD','PHYHIP']

grade_dat = pd.DataFrame({'id': ['CA.102548', 'CA.104338', 'CA.105094', 'CA.109745', 'CA.1906415', 'CA.1912627',
                                 'CA.1924346', 'CA.1926760', 'CA.1927842', 'CA.1933414', 'CA.1940640', 'CA.2004407',
                                 'CA.2005288', 'CA.2006047', 'CA.2008260', 'CA.2009329', 'CA.2009381', 'CA.2009850',
                                 'CA.2017611', 'CA.2039179', 'CA.2040686', 'CA.2045012', 'CA.2046297', 'CA.348981',
                                 'CA.354300', 'CA.359448', 'CA.94377', 'CA.98389', 'CA.98475', 'CA.99145'],
                          'age': [np.nan, np.nan, np.nan, np.nan, 49, 65, 46, 37, 36, 40, 66, 64, 46, 60, 59, 37, 47,
                                  49, 57, 42, 40, 40, 37, 56, 43, 30, np.nan, np.nan, np.nan, np.nan],
                          'grade': [2, 2, np.nan, 3, 2, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, np.nan, 3, 2, 3, 2, 3, 2, 3,
                                    np.nan, np.nan, 2, 2, np.nan, 2],
                          'Relapse':['no', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'no',
                                     'no', 'no', np.nan, 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no',
                                     'yes', 'no', 'no', 'no'],
                          'LN':['yes', 'yes', 'yes', 'no', 'no', 'yes', 'no', 'no', 'yes', 'no', 'yes', 'no', 'no',
                                'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no',
                                'yes', 'no', 'no', 'yes']})

grade_dat.index = grade_dat['id']
grade_dat['age_group'] = np.where(grade_dat['age']>grade_dat['age'].mean(),'old','young')
# Make sure the ids match
if np.all(grade_dat.id[0:30] == pca_data.index[0:30]):
    print("IDs match (y)")
grade_dat['PCA3'] = pca_data.loc[grade_dat.id,'PCA3'].values

calc_ttest(grade_dat.loc[grade_dat['age']<60,:], 'grade', [2,3], 'PCA3',tick_labels = None, colors = ['blue','red'], plot_result = False, return_ax = False, force_ttest = False, test_type = '2samp', verbose = False, y_label = None)
grade_dat.loc[grade_dat['grade'] ==2,'PCA3'].mean()
grade_dat.loc[grade_dat['grade'] ==3,'PCA3'].mean()


inc_genes_1.reverse()
inc_genes_2.reverse()
inc_genes = inc_genes_2+inc_genes_1


plt_comp = pca_comp.loc[inc_genes,'PCA3']
colors = ['indianred']*len(inc_genes_1)+['steelblue']*len(inc_genes_2)
fig, ax = plt.subplots(figsize=(4, 7.5))
ax.barh(plt_comp.index, plt_comp, color=colors, edgecolor='black', linewidth=0.75)
ax.barh(plt_comp.index[0], plt_comp.iloc[0], color=colors[0], edgecolor='black', linewidth=0.75, label = 'Gene group 1')
ax.barh(plt_comp.index[-1], plt_comp.iloc[-1], color=colors[-1], edgecolor='black', linewidth=0.75, label = 'Gene group 2')
ax.set_title('Gene groups (as per the paper)\nand their loadings on PC3')
ax.set_xlabel('PCA3 loading', size=13)
ax.set_ylabel('Chosen genes', size=13)
plt.legend()
plt.subplots_adjust(left=0.3, right=0.975, top=0.93, bottom=0.07)
ax.set_ylim(-0.75,len(inc_genes)+3)