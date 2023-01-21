# Biomedical_DataScience_Lab

This reposetory will be used to store and maintain the different implementations of GraphNeuralNetworks(GNNs) that will be produced by me throughout the Biomedical Datascience Lab in WS22/23

It currently stores: <br>
## LinkPredictionLogLoss.py
-> Linkprediction on the PrimeKG Dataset using SAGE encoder. Dataset split with SKlearn's train_test_split. Using LogLoss. Uses Dataloader on edges for batches. Encoder not learning, Decoder learning. Problem with same metric values for train and valid set. <br>
## LinkPredictionTest.py 
-> Linkprediction on the PrimeKG Dataset using SAGE encoder. Dataset split PyTorch's RandLinkSplit. Using BCE Loss. Uses NeighbourhoodSampler for batches. Currently not learning.  Problem with same metric values for train and valid set.<br>

Several old scripts in /oldstuff.
