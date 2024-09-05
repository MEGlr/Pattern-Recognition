from convolution import CNNBackbone
from lstm import LSTMBackbone
from modules import Classifier, MultitaskRegressor, Regressor
#import pickle

import torch
from pathlib import Path

from tqdm import tqdm

import matplotlib 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


def save_checkpoint(model, model_name, filename):
    
    fpath = Path("../checkpoints/" + model_name + "/" + filename + ".pickle")

    # make the directories if they do not exist
    # its like mkdir -p
    fpath.parents[0].mkdir(parents=True, exist_ok=True)

    #pickle.dump(model, open(fpath, "wb"))
    torch.save(model.state_dict(), fpath)
    torch.save(model, fpath)


def training_loop(model, train_dataloader, optimizer, regression = False, device="cuda"):
    
    # set model state to training mode
    model.train()

    running_loss = 0

     # train on the whole train set
    for inputs, ground_truth, lengths in tqdm(train_dataloader, leave=False):

        # move data to device
        inputs = inputs.to(device)
        ground_truth = ground_truth.to(device)
        lengths = lengths.to(device)

        # reset the accumulated gradient
        optimizer.zero_grad()

        # forward pass
        loss, outputs = model(inputs, ground_truth, lengths)
        
        
        running_loss += loss.item()

        # compute the gradients
        loss.backward()

        # update the model parameters
        optimizer.step()

            
    return running_loss


def validation_loop(model, val_dataloader, criterion, regression = False, device="cuda"):

    # set model to evaluation mode
    model.eval()
    
    val_loss = 0


    with torch.no_grad():

        for inputs, ground_truth, lengths in val_dataloader:

            # move data to device
            inputs = inputs.to(device)
            ground_truth = ground_truth.to(device)
            lengths = lengths.to(device)
            loss, _ = model(inputs, ground_truth, lengths)
            val_loss += loss.item()


    return val_loss  # Return validation_loss and anything else you need


def overfit_with_a_couple_of_batches(model,train_dataloader, optimizer, device):


    # select a couple of batches from the dataloader
    # how many batches to select?
    batches_count = 3  # (select the first batches_count batches)
    train_subset_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_dataloader.dataset, [i for i in range(batches_count)]),
            batch_size = train_dataloader.batch_size
        )
    
    # Run the training loop for an absurd amount of epoch
    epochs = 1000  # An absurd number of epochs
    
    for epoch in range(epochs):
        
        loss = training_loop(model,  train_subset_dataloader, optimizer, device)
        
        #print loss every 100 epochs
        if(epoch % 100 == 0):
            print(f"Epoch {epoch: <3}: running loss = {loss:.3f}.")
            


def train_normal(model_name, model, train_dataloader, val_dataloader, optimizer, epochs, regression, hparams, patience=10, device="cuda", transfer = None):

    # initialize best score value
    # this is used for the early stopping
    best_loss = 1e10

    # counter counts how many epochs have not improved the val loss 
    counter = 0
    # Train the model
    for epoch in range(epochs):

        running_loss = training_loop(model, train_dataloader, optimizer, regression, device)
        val_loss = validation_loop(model, val_dataloader, optimizer, regression, device = device)

        if val_loss > best_loss:
            # if the loss is worst than the best
            counter +=1
            if counter >= patience:
                print("\nTraining terminated due to early stopping.")
                return
        else:
            # we reached a new minimum for the val_loss

            #update best loss
            best_loss = val_loss

            #reset counter
            counter = 0

            # save the best model
            save_checkpoint(model, model_name, "best")
                
                
        # save checkpoints every 5 epochs
        if(epoch % 5 == 0):
            save_checkpoint(model, model_name, f"epoch_{epoch}")
             
        
        print(f"Epoch {epoch: <3}: running loss = {running_loss:.3f}, validation loss = {val_loss:.3f}.")
            


def train(model_name, model, train_dataloader, val_dataloader, optimizer, epochs, hparams, regression = False, device="cuda", overfit_batch=False, transfer = None):

    # the device parameter
    # is used in the train loops
    # to move the data to the associated device

    if overfit_batch:
        overfit_with_a_couple_of_batches(model, train_dataloader, optimizer, device)
    else:
        # Train the model
        train_normal(model_name, model, train_dataloader, val_dataloader, optimizer, epochs, regression, hparams = hparams,  device=device, transfer=transfer)
        



from dataset import *


if __name__ == "__main__":

    """
    We never used this python file as a script!
    """

    print("This python file cannot be used as a script")

    quit()





    # the code as it was given

    # make the models float64
    torch.set_default_tensor_type(torch.DoubleTensor)

    beat_mel_specs = SpectrogramDataset(
            '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',
            train=True,
            class_mapping=CLASS_MAPPING,
            max_length=-1,
            read_spec_fn=read_mel_spectrogram)
        
    train_loader_beat_mel, val_loader_beat_mel = torch_train_val_split(beat_mel_specs, 32 ,32, val_size=.33)
        
    # test_loader_beat_mel = SpectrogramDataset(
    #         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',
    #         #input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat
    #         train=False,
    #         class_mapping=CLASS_MAPPING,
    #         max_length=-1,
    #         read_spec_fn=read_mel_spectrogram)


    input_size = beat_mel_specs.feat_dim

    backbone = LSTMBackbone(
        input_size,
        rnn_size=128,
        num_layers=1,
        bidirectional=False,
        dropout=0.1,
    )

    model = Classifier(
        backbone, num_classes=10, load_from_checkpoint=None
    )


            
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
    epochs = 100
    train_dataloader = train_loader_beat_mel
    val_dataloader = val_loader_beat_mel

    
    device = "cpu"
    overfit_batch = True

    
    train("simple_lstm_classifier", model, train_dataloader, val_dataloader, optimizer, epochs, device=device, overfit_batch=overfit_batch)

