from train import *
from evaluate import evaluate
from sklearn.metrics import f1_score

import sys
import json
import datetime
import scipy 

from pathlib import Path

# make the models float32
torch.set_default_tensor_type(torch.FloatTensor)



def train_helper(model_name, model_type, backbone, backbone_wrapper, dataset_type, regression = False, epochs=100, device="cpu", hparams={}, transfer = None, max_length = 1):
    """
    trains LSTM or CNN (specified by model_type) with input specified by dataset_type
    dataset_type can be one of the following string values:
        spectrogram (spectrograms), spectrogram_beat (beat-synced spectrograms),
        chroma (chromagrams), chroma_beat (beat-synced spectrograms),
        fused (concatenated spectrograms and chromagrams), fused_beat (concatenated beat-synced spectrograms and beat-synced chromagrams)
    model_type can be one of the following string values:
        CNN  (train a CNN)
        LSTM  (train a LSTM)
    """


    # ================ Add timestamp to the model name ================

    if(model_name != 'hyper_tuning'):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        #model_name += "_" + timestamp

    print("training ", model_name)
    print("epochs:", epochs)

    # ================ Just some lookup tables ================

    # train set path for the input 
    PATH = {
        'spectrogram':      '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms',
        'chroma':           '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms',
        'fused':            '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms',
        'spectrogram_beat': '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat',
        'chroma_beat':      '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat',
        'fused_beat':       '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat',        
        'multitask' :       '../input/patreco3-multitask-affective-music/data/multitask_dataset',
        'multitask_beat':   '../input/patreco3-multitask-affective-music/data/multitask_dataset_beat'
    }

    # specify the appropriate read function for the input 
    READ_FN = {
        'spectrogram':      read_mel_spectrogram,
        'chroma':           read_chromagram,
        'fused':            read_fused_spectrogram,
        'spectrogram_beat': read_mel_spectrogram,
        'chroma_beat':      read_chromagram,
        'fused_beat':       read_fused_spectrogram,
        'multitask' :       read_mel_spectrogram,
        'multitask_beat':   read_mel_spectrogram
    }


    # ================ Hyperparameters ================

    # set default values for the hyperparameters

    hparams.setdefault("lr", 1e-4)
    hparams.setdefault("batch_train", 64)
    hparams.setdefault("batch_eval", 64)
    
    if model_type == "CNN":
        # specify default hyperparameters for CNN training
        hparams.setdefault("cnn_in_channels", 1)
        hparams.setdefault("cnn_filters", [32, 64, 128, 256])
        hparams.setdefault("cnn_out_feature_size",1000)
    else:
        # specify default hyperparameters for LSTM training
        hparams.setdefault("rnn_size", 128)
        hparams.setdefault("rnn_layers", 1)
        hparams.setdefault("bidirectional", False)
        hparams.setdefault("dropout", 0.1)

    print("Using hparms: ")
    for key, value in hparams.items():
        print(key, value, sep='\t')
    print("Starting training ...")


    # save the hparams that will be used in a file
    fpath = Path("../checkpoints/" + model_name + "/hparams.json")

    # make the directories if they do not exist
    # its like mkdir -p
    fpath.parents[0].mkdir(parents=True, exist_ok=True)

    with open(fpath, 'w') as fp:
        json.dump(hparams, fp)

    
    # ================ Load Dataset ================

    # create train Dataset 
    beat_mel_specs = SpectrogramDataset(
            Path(PATH[dataset_type]),
            train=True,
            class_mapping=CLASS_MAPPING,
            regression = regression,
            max_length = max_length,
            read_spec_fn=READ_FN[dataset_type])
    

    # split train set into train and validation set 
    train_loader_beat_mel, val_loader_beat_mel = torch_train_val_split(beat_mel_specs, hparams["batch_train"], hparams["batch_eval"], val_size=.33)
        

    # ================ Define the Model ================

    if (model_type == "CNN"):

        # get the input_size for the given dtaset_type
        x_b1, _, _ = next(iter(train_loader_beat_mel))
        input_size = x_b1[0].shape

        # create the model
        backbone = CNNBackbone(
            input_size,
            in_channels = hparams["cnn_in_channels"],  
            filters  = hparams["cnn_filters"],
            feature_size = hparams["cnn_out_feature_size"]
        ).to(device)

    else:
        # the backbone is a LSTM

        input_size = beat_mel_specs.feat_dim

        backbone = backbone(
            input_size,
            rnn_size=hparams["rnn_size"],
            num_layers=hparams["rnn_layers"],
            bidirectional=hparams["bidirectional"],
            dropout=hparams["dropout"],
        ).to(device)

    
    # print a message if we are performing transfer learning
    load_from_checkpoint = transfer
    if transfer:
      print(" #### Transfer Learning ####")
      print(f"Loading model from {load_from_checkpoint} ...")


    if not regression:
        # it is a classification problem

        model = backbone_wrapper(
            backbone, num_classes=10, load_from_checkpoint = load_from_checkpoint
        ).to(device)

    else:
        # it is a regression problem

        model = backbone_wrapper(
            backbone, load_from_checkpoint=load_from_checkpoint
        ).to(device)

    # ================ Define the Optimizer ================

    learning_rate = hparams["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    

    # ================ Train ================

    train_dataloader = train_loader_beat_mel
    val_dataloader = val_loader_beat_mel

    overfit_batch=False

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train(
        model_name,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        epochs,
        hparams=hparams,
        device=device,
        overfit_batch=overfit_batch, 
        transfer = transfer
    )

    # ============ Calculate macro f1 score on the Best Model (in case of the classification task) ===========
    # ============ Calculate the Spearman Correlation otherwise ==============================================

    # load the best model
    saved_model_path = "../checkpoints/" + model_name + "/" + "best" + ".pickle"
    model = torch.load(saved_model_path, map_location=torch.device(device))

    # evaluate the model
    y_pred, y_true = evaluate(model, val_dataloader, regression, device=device)
    
    if not regression:
        
        # calculate the validation set f1-score and return it
        val_score = f1_score(y_true, y_pred, average='macro')
        print("Macro f1 score of the best model (on validation set): ", round(val_score*100, 3), " %.")

    else:

        # calculate spearman correlation between ground truth and predicted values (in validation set) and return it 
        metrics = ["Valence", "Energy", "Danceability"]

        # in case of the single task model, we need one metric for this task
        metrics_count = 1 

        if (type(y_pred[0]) == list):
            # in case of the multitask model
            # we need one metric for each task
            metrics_count = len(y_pred[0]) 

        spearman_list = []
        start = regression-1 if metrics_count == 1 else 0

        for i in range(start, start+metrics_count): # for each task

            if (type(y_pred[0]) == list): # if it is a multitask model

                y_true_curr = [row[i+1] for row in y_true]
                y_pred_curr = [row[i] for row in y_pred]

            else: # if it is a single task model

                y_true_curr = np.array(y_true).flatten()
                y_pred_curr = np.array(y_pred).flatten()
            
            # calculate the spearman correlation
            spearmanr = scipy.stats.spearmanr(
                    y_true_curr,
                    y_pred_curr
              ).correlation

            # save it in a list (the list has an element for each task)
            spearman_list.append(spearmanr)

            print(f"{metrics[i]}: Spearman correlation between ground truth and predicted values (in validation set) is equal to {spearmanr:.3f}")


            # make a plot --- real values vs predicted values
            val_score = spearmanr
            # Scatter Plot Predictions 
            plt.figure(figsize=(6,6))
            plt.scatter(y_true_curr, y_pred_curr)
            plt.title(f"{metrics[i]} : {model_type} Scatter Plot (validation set)")
            plt.xlabel('ground truths')
            plt.ylabel('predictions')
            reg_line = np.linspace(0,1,1000)
            plt.plot(reg_line,reg_line, '-', color='g')
            fpath = Path(f'../plots/scatter_plot_{metrics[i]}_{model_name}.png')
            plt.savefig(fpath, dpi=300)
            # plt.show()


        # if it is a mutlitask model, print the mean of the scores for each task
        if (type(y_pred[0]) == list):
            print("#########################################################################################################################")
            print("#########################################################################################################################")
            print(f"Mean Spearman correlation between ground truth and predicted values (in validation set) for all axis is equal to {np.mean(spearman_list):.4f}.")
            print("#########################################################################################################################")
            print("#########################################################################################################################")

            val_score = np.mean(spearman_list)
    
    # ================ Return the Best Model and the Macro f1 or Spearman correlation (when regression = True) in validation set ================
   
    return model, val_score
