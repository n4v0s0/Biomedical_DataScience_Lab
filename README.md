# Biomedical_DataScience_Lab

This reposetory will be used to store and maintain the different implementations of GraphNeuralNetworks(GNNs) that will be produced by me throughout the Biomedical Datascience Lab in WS22/23

It currently stores: <br>
## LinkPrediction.py
-> Linkprediction on a subset of the PrimeKG Dataset using a SAGE encoder and a Linear decoder. We split the dataset by using torch's randomsplit() and subgraph(). The trainset consists of 20% of the nodes in the PrimeKG dataset, the valid set of 5%. We use LogLoss as criterion. Uses Dataloader on edges for batches.<br>


Several old scripts in /oldstuff.
