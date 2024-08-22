from train import stBERT
from data import Adata2Torch_data, Spatial_Dis_Cal, process_adata, read_data, get_initial_label
import warnings
from utils import *
from cfg import *

warnings.filterwarnings("ignore")
set_seed(seed=0)
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

def main(config, adata, n_clusters=10, graph_mode="knn", cluster_method="mclust",
             refine=True, data_save_path="./", true_labels=None, eval=True):
    adata = process_adata(adata)
    if graph_mode in ["knn", "KNN"]:
        Spatial_Dis_Cal(adata, knn_dis=5, model="KNN")
    else:
        Spatial_Dis_Cal(adata, rad_dis=graph_mode)

    data = Adata2Torch_data(adata)
    ss_labels = get_initial_label(adata, method=cluster_method,
                                  n_clusters=n_clusters)
    reso = n_clusters

    n_clusters = n_clusters if cluster_method == "mclust" else len(set(ss_labels))
    config.n_nodes, config.feat_dim, config.edge_in_features = data.x.shape[0], data.x.shape[1], data.train_pos_edge_index[1]
    model = stBERT(config, n_clusters=n_clusters).cuda()
    model.train(
        data, method=cluster_method,
        position=adata.obsm['spatial'], eval=eval, reso=reso,
        ss_labels=ss_labels, data_save_path=data_save_path,
        labels=true_labels, lr=config.learning_rate)

if __name__ == '__main__':

    config = ModelConfig("BRCA1", bert_name='BERT-mini'); print('data use:', config.dataset_name) #SSMOB
    adata = read_data(config.dataset_name)
    if config.groundtruth: config.n_clusters = adata.obs['ground_truth'].nunique()

    main(config, adata, n_clusters=config.n_clusters,
         cluster_method=config.cluster_method, refine=config.refine_labels,
         graph_mode=config.graph_mode, eval=True,
         data_save_path=config.data_save_path,
         true_labels=adata.obs['ground_truth'] if config.groundtruth else None
         )

