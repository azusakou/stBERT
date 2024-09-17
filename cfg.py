class ModelConfig:
    def __init__(self, dataset_name='DLPFC_151676', bert_name='BERT-base'):
        self.data = ["STARmap",
                     "BRCA1",
                     'Mouse_hippocampus', # no labels
                     "Mouse_olfactory", # no labels
                     "Mouse_embryo_E9_E1S1",
                     "Mouse_embryo_E9_E2S1",
                     "Mouse_embryo_E9_E2S2",
                     "Mouse_embryo_E9_E2S3",
                     "Mouse_embryo_E9_E2S4",
                     'DLPFC_151507',
                     'DLPFC_151508',
                     'DLPFC_151509',
                     'DLPFC_151510',
                     'DLPFC_151669',
                     'DLPFC_151670',
                     'DLPFC_151671',
                     'DLPFC_151672',
                     'DLPFC_151673',
                     'DLPFC_151674',
                     'DLPFC_151675',
                     'DLPFC_151676',
                     'SSMOB'
                     ]

        self.dataset_name = dataset_name # self.data[0]
        self.graph_mode = self.determine_graph_mode()
        self.n_clusters = self.determine_n_clusters()
        self.groundtruth = self.determin_groundtruth_exist()
        #self.cluster_method = "louvain" if "Mouse_olfactory" in self.dataset_name else "mclust"
        self.cluster_method = "louvain" if any(name in self.dataset_name for name in ["Mouse_olfactory", "SSMOB"]) else "mclust"

        self.learning_rate = 1e-5
        self.VAE_encoder_type = ['Mamba', 'SGCnov'][1]
        self.Discriminator_use = False
        self.Discriminator_mobius_use = False

        self.epochs = 1000
        self.hidden_dim = 256
        self.embed_dim = 128


        self.training_bert = True
        self.bert_name = bert_name # ['BERT-mini', 'BERT-small', 'BERT-base', 'BERT-large'][0]
        self.CustomBERT(self.bert_name)
        self.load_pretrained = True


        self.sc = 0
        self.db = 0
        self.fusion_layer_n = 6
        self.graph_fusion = True

        self.refine_labels = True if self.dataset_name == 'STARmap' else False
        self.device = 'cuda'
        self.data_save_path = self.determin_data_save_path()
        self.regularizer_use = False

        # for pretraining
        self.n_nodes = 10
        self.feat_dim = 10
        self.edge_in_features = 10

    def determine_graph_mode(self):
        if "DLPFC" in self.dataset_name: return 150
        elif "BRCA1" in self.dataset_name: return 300
        elif "Mouse_olfactory" in self.dataset_name:return 50
        elif "Mouse_hippocampus" in self.dataset_name: return 40
        elif "Mouse_embryo" in self.dataset_name: return "knn"
        elif "STARmap" in self.dataset_name: return 400
        elif "SSMOB" in self.dataset_name:return 50
        elif "MOSTA.h5ad" in self.dataset_name: return "knn"
        else: return None  # or some default value, if no matches

    def determine_n_clusters(self):
        if "DLPFC" in self.dataset_name:
            # Check for specific identifiers within DLPFC dataset names
            if any(specific in self.dataset_name for specific in ['151669', '151670', '151671', '151672']):
                return 5
            else:
                return 7
        elif "BRCA1" in self.dataset_name: return 20
        elif "Mouse_olfactory" in self.dataset_name: return 0.5
        elif "Mouse_hippocampus" in self.dataset_name: return 10
        elif "Mouse_embryo" in self.dataset_name: return 10
        elif "STARmap" in self.dataset_name: return 7
        elif "SSMOB" in self.dataset_name:return 0.8 #10
        elif "MOSTA.h5ad" in self.dataset_name: return 10
        else: return None  # or some default number of clusters

    def determin_groundtruth_exist(self):
        return not any(k in self.dataset_name for k in ['Mouse_h', 'Mouse_o', 'SSMOB'])

    def determin_data_save_path(self):
        if self.training_bert:
            return f'./results/DATA_{self.dataset_name}/EPOCH_{self.bert_name}_mask-{self.bert_mask_rate}_dim-{self.bert_dim}_nlays-{self.bert_layers}_nhead-{self.bert_heads}/'
        else:
            return f'./results/DATA_{self.dataset_name}/EPOCH{self.epochs}_{self.Discriminator_use}/'

    def CustomBERT(self, bert_name):
        # Initialize parameters based on the model name
        #print(bert_name)
        if bert_name == 'BERT-mini':
            self.bert_dim = 256
            self.bert_layers = 4
            self.bert_heads = 4
            self.learning_rate = 5e-4  # Higher learning rate for smaller model
            self.epochs = 400
            self.bert_mask_rate = 0.15

        elif bert_name == 'BERT-small':
            self.bert_dim = 512
            self.bert_layers = 8
            self.bert_heads = 8
            self.learning_rate = 5e-5
            self.epochs = 400
            self.bert_mask_rate = 0.15

        elif bert_name == 'BERT-base':
            self.bert_dim = 768
            self.bert_layers = 12
            self.bert_heads = 12
            self.learning_rate = 5e-5
            self.epochs = 400
            self.bert_mask_rate = 0.15

        elif bert_name == 'BERT-large':
            self.bert_dim = 1024
            self.bert_layers = 24
            self.bert_heads = 16
            self.learning_rate = 2e-4  # Lower learning rate for larger model
            self.epochs = 1500
            self.bert_mask_rate = 0.15

        else:
            raise ValueError("Unknown BERT model name")

