import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Data
import scanpy as sc
#import metis
import squidpy as sq
from train import refine_label, dopca, mclust_R
from torch_geometric.utils import train_test_split_edges


def read_data(dataset, data_path='datasets/'):
    if dataset == "STARmap":
        file_fold = data_path + str(dataset)
        adata = sc.read(file_fold+"/STARmap_20180505_BY3_1k.h5ad")
        adata.var_names_make_unique()
        df_meta = pd.read_table(file_fold + '/Annotation_STARmap_20180505_BY3_1k.txt',
                                sep='\t', index_col=0)
        adata.obs['ground_truth'] = adata.obs["label"]
        adata.obs['Annotation'] = df_meta.loc[adata.obs_names, 'Annotation'].values

    if dataset == "BRCA1":
        file_fold = data_path + str(dataset) + '/BRCA' #please replace 'file_fold' with the download path
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5',
                               load_images=True)
        adata.var_names_make_unique()

        df_meta = pd.read_table(file_fold + '/BC_truth.txt',
                                sep='\t', header=None, index_col=0)
        df_meta.columns = ['groud_truth']
        adata.obs['ground_truth'] = df_meta.loc[adata.obs_names, 'groud_truth'].values
        # df_meta = pd.read_table(file_fold + '/metadata.tsv',sep='\t',index_col=0)
        #adata.obs['ground_truth'] = df_meta.loc[adata.obs_names, 'ground_truth'].values

    if dataset == "Mouse_hippocampus":
        adata = sq.datasets.slideseqv2()
        adata.var_names_make_unique()
    
    if dataset in ["Mouse_embryo_E9_E1S1", "Mouse_embryo_E9_E2S1",
                   "Mouse_embryo_E9_E2S2", "Mouse_embryo_E9_E2S3",
                   "Mouse_embryo_E9_E2S4"]: #ok
        file_fold = data_path + 'mouse_embryo/' + str(dataset)
        adata = sc.read(file_fold+".MOSTA.h5ad")
        adata.var_names_make_unique()
        adata.obs["ground_truth"] = adata.obs["annotation"]

    if dataset in ['E16_E1S3.MOSTA.h5ad',
                    'E16_E1S4.MOSTA.h5ad',
                    'E16_E1S1.MOSTA.h5ad',
                    'E16_E1S2.MOSTA.h5ad',
                    'E15_E2S1.MOSTA.h5ad',
                    'E14_E1S4.MOSTA.h5ad',
                    'E15_E1S1.MOSTA.h5ad',
                    'E15_E1S4.MOSTA.h5ad',
                    'E14_E1S5.MOSTA.h5ad',
                    'E15_E1S3.MOSTA.h5ad',
                    'E14_E1S2.MOSTA.h5ad',
                    'E15_E1S2.MOSTA.h5ad',
                    'E14_E2S2.MOSTA.h5ad',
                    'E14_E1S3.MOSTA.h5ad',
                    'E14_E2S1.MOSTA.h5ad',
                    'E14_E1S1.MOSTA.h5ad',
                    'E13_E1S4.MOSTA.h5ad',
                    'E13_E1S2.MOSTA.h5ad',
                    'E13_E1S3.MOSTA.h5ad',
                    'E13_E1S1.MOSTA.h5ad',
                    'E12_E1S4.MOSTA.h5ad',
                    'E12_E1S5.MOSTA.h5ad',
                    'E12_E1S1.MOSTA.h5ad',
                    'E12_E1S3.MOSTA.h5ad',
                    'E12_E1S2.MOSTA.h5ad',
                    'E11_E1S1.MOSTA.h5ad',
                    'E10_E1S3.MOSTA.h5ad',
                    'E11_E1S4.MOSTA.h5ad',
                    'E11_E1S3.MOSTA.h5ad',
                    'E10_E1S1.MOSTA.h5ad',
                    'E11_E1S2.MOSTA.h5ad',
                    'E10_E1S2.MOSTA.h5ad',
                    'E10_E2S1.MOSTA.h5ad']: #ok
        file_fold = data_path + 'mouse_embryo/' + str(dataset)
        adata = sc.read(file_fold)
        adata.var_names_make_unique()
        adata.obs["ground_truth"] = adata.obs["annotation"]

    if dataset == "Mouse_olfactory":

        _saved=True
        file_fold = data_path +  str(dataset) + '_slide_seqv2/'

        if _saved:
            adata = sc.read_h5ad(file_fold + 'Mouse_olfactory_slide_seqv2.h5ad')
            adata.var_names_make_unique()
        else:
            counts = pd.read_csv(file_fold+'Puck_200127_15.digital_expression.txt',
                                 sep='\t', index_col=0)
            coor_df = pd.read_csv(file_fold+'Puck_200127_15_bead_locations.csv',
                                  index_col=3)[['xcoord', 'ycoord']]
            used_barcode = pd.read_csv(file_fold + 'used_barcodes.txt', sep='\t', header=None)
            used_barcode = used_barcode[0]
            adata = sc.AnnData(counts.T)
            adata.var_names_make_unique()
            coor_df = coor_df.loc[adata.obs_names, ['xcoord', 'ycoord']]
            adata.obsm["spatial"] = coor_df.to_numpy()
            sc.pp.calculate_qc_metrics(adata, inplace=True)
            adata = adata[used_barcode,]
            sc.pp.filter_genes(adata, min_cells=50)

            normalization=1
            if normalization:
                # Normalization
                sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
            adata.write(file_fold + 'Mouse_olfactory_slide_seqv2.h5ad')

    if "DLPFC" in dataset:
        file_fold = data_path + str('DLPFC/') + dataset.split('_')[1]
        adata = sc.read_visium(path=file_fold,
                               count_file=dataset.split('_')[1] + '_filtered_feature_bc_matrix.h5',
                               #load_images=False
                               )
        adata.var_names_make_unique()
        # add ground_truth

        df_meta = pd.read_table(file_fold + '/' + dataset.split('_')[1] + '_truth.txt',
                                sep='\t', header=None, index_col=0)
        df_meta.columns = ['groud_truth']
        adata.obs['ground_truth'] = df_meta.loc[adata.obs_names, 'groud_truth'].values

    if dataset == "SSMOB":

        from scipy.sparse import csr_matrix
        _saved=True
        file_fold = data_path +  str(dataset) + '/'

        if _saved:
            adata = sc.read_h5ad(file_fold + 'SSMOB.h5ad')
            adata.var_names_make_unique()
        else:

            counts = pd.read_csv(file_fold + 'RNA_counts.tsv.gz', sep='\t', index_col=0).T
            counts.index = [f'Spot_{i}' for i in counts.index]
            adata = sc.AnnData(counts)
            adata.X = csr_matrix(adata.X, dtype=np.float32)

            df_pos = pd.read_csv(file_fold + 'position.tsv', sep='\t')
            adata.obsm['spatial'] = df_pos[['y', 'x']].values

            used_barcode = pd.read_csv(file_fold + 'used_barcodes.txt', sep='\t', header=None)
            used_barcode = used_barcode[0]
            adata = adata[used_barcode,]

            adata.write(file_fold + 'SSMOB.h5ad')

    return adata


def process_adata(adata):
    adata.var_names_make_unique()
    #Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4) ##normalized data
    sc.pp.log1p(adata)  #log-transformed data
    adata = adata[:, adata.var['highly_variable']]
    return adata


def Adata2Torch_data(adata): 
    G_df = adata.uns['Spatial_Net'].copy() 
    spots = np.array(adata.obs_names) 
    spots_id_tran = dict(zip(spots, range(spots.shape[0]))) 
    G_df['Spot1'] = G_df['Spot1'].map(spots_id_tran) 
    G_df['Spot2'] = G_df['Spot2'].map(spots_id_tran) 

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Spot1'], G_df['Spot2'])), 
        shape=(adata.n_obs, adata.n_obs))

    G = G + sp.eye(G.shape[0]) 
    edgeList = np.nonzero(G) 

    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense())) 
    data = train_test_split_edges(data)
    return data


def Spatial_Dis_Cal(adata, rad_dis=None, knn_dis=None, model='Radius', verbose=True):
    """\
    Calculate the spatial neighbor networks, as the distance between two spots.
    Parameters
    ----------
    adata:  AnnData object of scanpy package.
    rad_dis:  radius distance when model='Radius' 
    knn_dis:  The number of nearest neighbors when model='KNN' 
    model:
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_dis. 
        When model=='KNN', the spot is connected to its first knn_dis nearest neighbors.
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """
    assert(model in ['Radius', 'KNN', "BallTree"]) 
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index 
    # coor.columns = ['imagerow', 'imagecol']
    coor.columns = ['Spatial_X', 'Spatial_Y'] 

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_dis).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
      
        KNN_list = []
        for spot in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([spot]*indices[spot].shape[0], indices[spot], distances[spot])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=knn_dis+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for spot in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([spot]*indices.shape[1],indices[spot,:], distances[spot,:])))

    if model == "BallTree":
        from sklearn.neighbors import BallTree
        tree = BallTree(coor)
        distances, ind = tree.query(coor, k=knn_dis+1)
        indices = ind[:, 1:]
        KNN_list=[]

        for spot in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([spot]*indices.shape[1],indices[spot,:], distances[spot,:])))
        # for node_idx in range(coor.shape[0]):
        #     for j in np.arange(0, indices.shape[1]):
        #         KNN_list.append(node_idx, indices[node_idx][j])

    KNN_df = pd.concat(KNN_list) 
    KNN_df.columns = ['Spot1', 'Spot2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_spot_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Spot1'] = Spatial_Net['Spot1'].map(id_spot_trans) 
    Spatial_Net['Spot2'] = Spatial_Net['Spot2'].map(id_spot_trans) 
    if verbose:
        print('The graph contains %d edges, %d spots.' %(Spatial_Net.shape[0], adata.n_obs)) 
        print('%.4f neighbors per spot on average.' %(Spatial_Net.shape[0]/adata.n_obs)) 
    adata.uns['Spatial_Net'] = Spatial_Net


def get_initial_label(adata, n_clusters, refine=True, method="mclust"):
    features = adata.X
    if type(features) == np.ndarray:
        features = features
    else:
        features = features.todense()
    pca_input = dopca(features, dim = 20)
    if method == "mclust":
        pred = mclust_R(embedding=pca_input, num_cluster=n_clusters)
    if method == "louvain":
        adata.obsm["pca"] = pca_input
        sc.pp.neighbors(adata, n_neighbors=50, use_rep="pca")
        sc.tl.louvain(adata, resolution=n_clusters, random_state=0)
        pred=adata.obs['louvain'].astype(int).to_numpy()
    if refine:
        pred = refine_label(pred, adata.obsm["spatial"], radius=60)
    pred = list(map(int, pred))
    return np.array(pred)
