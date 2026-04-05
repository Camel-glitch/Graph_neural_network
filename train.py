#####################################################
############## TRAIN FUNCTION #######################
#####################################################
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch_geometric.nn as graphnn
from sklearn.metrics import f1_score
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader




def evaluate(model, device, dataloader):
    score_list_batch = []

    # On passe le modèle en mode évaluation (désactive le dropout)
    model.eval()

    # Pas besoin de calculer les gradients pour l'évaluation, ça économise de la RAM
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index)

            # CORRECTION ICI : Le seuil de décision est à 0.5 pour des probabilités
            predict = np.where(output.cpu().numpy() >= 0.5, 1, 0)

            score = f1_score(batch.y.cpu().numpy(), predict, average="micro")
            score_list_batch.append(score)

    return np.array(score_list_batch).mean()

def train(model, loss_fcn, device, optimizer, max_epochs, train_dataloader, val_dataloader):
    epoch_list = []
    scores_list = []

    # loop over epochs
    for epoch in range(max_epochs):
        model.train()
        losses = []
        # loop over batches
        for i, train_batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            train_batch_device = train_batch.to(device)
            # logits is the output of the model
            logits = model(train_batch_device.x, train_batch_device.edge_index)
            # compute the loss
            loss = loss_fcn(logits, train_batch_device.y)
            # optimizer step
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        loss_data = np.array(losses).mean()
        print("Epoch {:05d} | Loss: {:.4f}".format(epoch + 1, loss_data))

        if epoch % 5 == 0:
            # evaluate the model on the validation set
            # computes the f1-score (see next function)
            score = evaluate(model, device, val_dataloader)
            print("F1-Score: {:.4f}".format(score))
            scores_list.append(score)
            epoch_list.append(epoch)

    return epoch_list, scores_list
