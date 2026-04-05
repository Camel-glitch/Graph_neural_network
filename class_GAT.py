import torch
import torch.nn as nn
import torch.nn.functional as F



class SparseGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super(SparseGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha 
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, edge_index):
        """
        x: Caractéristiques des nœuds (N, in_features)
        edge_index: Liste d'arêtes (2, E)
        """
        N = x.size(0)
        Wh = torch.mm(x, self.W) # (N, out_features)

        # 1. On récupère les caractéristiques des nœuds sources et cibles pour chaque arête
        # edge_index[0] sont les indices de départ, edge_index[1] les indices d'arrivée
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        # 2. Concaténation des caractéristiques pour chaque arête (Calcul de e_ij)
        # On ne calcule l'attention QUE pour les arêtes existantes (E), pas pour N*N
        edge_h = torch.cat((Wh[source_nodes], Wh[target_nodes]), dim=1) # (E, 2 * out_features)
        
        e = self.leakyrelu(torch.matmul(edge_h, self.a).squeeze()) # (E)

        # 3. Softmax normalisé par nœud de destination (attention score alpha_ij)
        # On utilise une astuce pour faire le softmax de manière éparse
        alpha = self.softmax_sparse(e, target_nodes, N)
        alpha = F.dropout(alpha, self.dropout, training=self.training)

        # 4. Agrégation des messages : Somme pondérée par l'attention
        # Pour chaque arête, on multiplie la valeur du voisin par son score d'attention
        v_weighted = Wh[source_nodes] * alpha.view(-1, 1) # (E, out_features)
        
        # On somme les contributions pour chaque nœud cible
        h_prime = torch.zeros_like(Wh)
        h_prime.index_add_(0, target_nodes, v_weighted)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def softmax_sparse(self, src, index, num_nodes):
        """ Calcule le softmax pour chaque groupe de nœuds voisins """
        # On soustrait le max pour la stabilité numérique
        e_max = torch.zeros(num_nodes, device=src.device)
        e_max.index_reduce_(0, index, src, reduce='amax', include_self=False)
        e_exp = torch.exp(src - e_max[index])
        
        # Somme des exponentielles par nœud cible
        e_sum = torch.zeros(num_nodes, device=src.device)
        e_sum.index_add_(0, index, e_exp)
        
        return e_exp / (e_sum[index] + 1e-16)

class InductiveGAT(nn.Module):
    def __init__(self, nfeat, alpha=0.2):
        """
        GAT pour tâche inductive (ex: PPI Dataset).
        Selon l'article : pas de dropout, 3 couches, skip connection sur la couche intermédiaire.
        """
        super(InductiveGAT, self).__init__()
        
        # Selon l'article : "we found no need to apply L2 regularization or dropout"
        self.dropout = 0.0 
        
        # --- Couche 1 ---
        # K = 4 têtes d'attention, F' = 256 features par tête. 
        # Sortie concaténée = 4 * 256 = 1024
        self.layer1_heads = nn.ModuleList([
            SparseGraphAttentionLayer(nfeat, 256, dropout=self.dropout, alpha=alpha, concat=True) 
            for _ in range(4)
        ])
        
        # --- Couche 2 (Couche intermédiaire) ---
        # K = 4 têtes, F' = 256 features. 
        # Entrée = 1024 (sortie de la couche 1). Sortie concaténée = 1024.
        self.layer2_heads = nn.ModuleList([
            SparseGraphAttentionLayer(1024, 256, dropout=self.dropout, alpha=alpha, concat=True) 
            for _ in range(4)
        ])
        
        # --- Couche 3 (Couche de sortie) ---
        # K = 6 têtes d'attention, 121 features chacune.
        # Entrée = 1024. Sortie = 121 (les têtes seront moyennées, pas concaténées).
        self.layer3_heads = nn.ModuleList([
            SparseGraphAttentionLayer(1024, 121, dropout=self.dropout, alpha=alpha, concat=False) 
            for _ in range(6)
        ])

    def forward(self, x, adj):
        # --- Passage Couche 1 ---
        x1 = torch.cat([att(x, adj) for att in self.layer1_heads], dim=1)
        x1 = F.elu(x1)
        
        # --- Passage Couche 2 avec Skip Connection ---
        x2 = torch.cat([att(x1, adj) for att in self.layer2_heads], dim=1)
        
        # "we have successfully employed skip connections across the intermediate attentional layer"
        # Les dimensions correspondent parfaitement (1024 -> 1024)
        x2 = x2 + x1 
        x2 = F.elu(x2)
        
        # --- Passage Couche 3 (Sortie) ---
        # Calcul des K=6 têtes
        out_projections = [att(x2, adj) for att in self.layer3_heads]
        
        # "that are averaged" (Moyenne des têtes)
        x3 = torch.mean(torch.stack(out_projections), dim=0)
        
        # "followed by a logistic sigmoid activation" (Pour classification multi-labels)
        return torch.sigmoid(x3)