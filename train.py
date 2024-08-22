import torch
import torch.optim

import pandas as pd
import numpy as np
import torch.nn as nn

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import (
    v_measure_score, homogeneity_score, completeness_score)
from sklearn.decomposition import PCA
import scanpy as sc

from tqdm import tqdm
from stBERT import GraphBERT, mask_features

from torch.cuda.amp import autocast, GradScaler

class stBERT(nn.Module):
    def __init__(self, config, n_clusters=10,
                 reg_hidden_dim_1=64, reg_hidden_dim_2=32,
                 clamp=0.01) -> None:
        super(stBERT, self).__init__()
        self.n_clusters = n_clusters
        self.epochs = config.epochs
        self.input_dim = config.feat_dim
        self.embed_dim = config.embed_dim
        self.hidden_dim = config.hidden_dim
        self.clamp = clamp
        self.dataset_name = config.dataset_name
        self.evaluate_metric = 'ari' if config.groundtruth and ("Mouse_embryo" not in config.dataset_name) else 'sc'
        self.tmp_ari = -1
        self.tmp_sc = -1
        self.tmp_db = 100
        self.bert = GraphBERT(config)
        self.train_model = self._half_train_model_BERT
        self.bert_mask_rate = config.bert_mask_rate
        self.bert_name = config.bert_name
        self.load_model_encoder()
        for param in self.bert.encoder.parameters():
            param.requires_grad = False

    def train(self, data,  ss_labels=None, position=None, data_save_path=None,
                  method="mclust",labels=None, print_freq=100,
                  lr=1e-4, reso=0.5,
                  weight_decay=5e-05, eval=True):

        scaler = GradScaler()
        data = data.cuda()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.bert.parameters()),
                                     eps=1e-15, 
                                     lr=lr, weight_decay=weight_decay)

        criterion = nn.MSELoss()

        with tqdm(range(self.epochs), desc='Training') as pbar:
            for epoch in pbar:
                self.bert.train()
                optimizer.zero_grad()

                with autocast():
                    masked_x, mask = mask_features(data.x, self.bert_mask_rate,
                                                   rand_e=epoch
                                                   )
                    z = self.bert(masked_x, data.train_pos_edge_index)
                    loss = criterion(self.bert.mlm_recon(z, data.train_pos_edge_index)[mask], data.x[mask])

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if epoch > int(self.epochs-30):
                    completeness, hm, nmi, ari, z, pca_embedding, pred_label = self.eval_bert(
                        data, labels=labels, position=position,
                        save_name=data_save_path, method=method, reso=reso)

    def eval_bert(self, data, labels=None, reso=0.5,
                   position=None, save_name=None, method="mclust"):
        self.eval()
        with autocast():
            z = self.bert(data.x, data.train_pos_edge_index)
        pca_input = dopca(z.cpu().detach().numpy(), dim=20)
        if method == "mclust":
            pred_mclust = mclust_R(embedding=pca_input,
                                   num_cluster=self.n_clusters)
        if method == "louvain":
            adata_tmp = sc.AnnData(pca_input)
            sc.pp.neighbors(adata_tmp, n_neighbors=20)
            sc.tl.louvain(adata_tmp, resolution=reso, random_state=0)
            pred_mclust = adata_tmp.obs['louvain'].astype(int).to_numpy()

        if labels is not None:
            label_df = pd.DataFrame({"True": labels,
                                     "Pred": pred_mclust}).dropna()
            # label_df = pd.DataFrame({"True": labels, "Pred": pred}).dropna()
            completeness = completeness_score(
                label_df["True"], label_df["Pred"])
            hm = homogeneity_score(label_df["True"], label_df["Pred"])
            nmi = v_measure_score(label_df["True"], label_df["Pred"])
            ari = adjusted_rand_score(label_df["True"], label_df["Pred"])
        else:
            completeness, hm, nmi, ari = 0, 0, 0, 0
        return completeness, hm, nmi, ari, z.cpu().detach().numpy(), pca_input, pred_mclust
    
    def load_model_encoder(self):
        try:
            """Load BERT model encoder and convolutional layer parameters from a file."""
            # Path to the file containing the saved parameters
            file_path = f"./BERT_weight/{self.bert_name}_combined.pt"

            # Load the saved model parameters from the file
            model_params = torch.load(file_path)

            # Update the model's encoder and convolutional layer with the loaded parameters
            self.bert.encoder.load_state_dict(model_params['encoder'])
            self.bert.conv.load_state_dict(model_params['conv'])
        except:
            print(f"can not Load model parameters from {file_path}")

    def load_model_components(self,):
        """ Load specific model components from files based on an identifier. """
        components = {
            #'pos_embedding': self.bert.embedding.pos_embedding,
            'node_conv': self.bert.embedding.node_conv,
            'reconstruction_head': self.bert.reconstruction_head,
        }
        try:
            pos_embedding_path = f"./model_components/{self.bert_name}_{self.dataset_name}_pos_embedding.pt"
            self.bert.embedding.pos_embedding.data = torch.load(pos_embedding_path)
            #print(f"Loaded pos_embedding from {pos_embedding_path}")

            for component_name, component in components.items():
                file_path = f"./model_components/{self.bert_name}_{self.dataset_name}_{component_name}.pt"
                component.load_state_dict(torch.load(file_path))
                #print(f"Loaded {component_name} from {file_path}")
        except:
            print(f'can not load components')

def dopca(X, dim=10):
    pcaten = PCA(n_components=dim, random_state=0)
    X_10 = pcaten.fit_transform(X)
    return X_10


def mclust_R(embedding, num_cluster, modelNames='EEE', random_seed=0):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(
        embedding), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    mclust_res = mclust_res.astype('int')
    # mclust_res = mclust_res.astype('category')
    return mclust_res


