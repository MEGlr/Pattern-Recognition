from main_train import *


# help function for printing in console
def step_seperator(step_num):
    """
    Just prints a message to the stdout
    to inform the user which step is currently running
    """

    print("\n\033[1;32m[*] Running Step {} ...\033[0m\n".format(step_num))

# --------------------------------------------------------
# define device to use
# use cuda if available else cpu

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# number of epochs for training
EPOCHS = 25

# hyper parameters
# In our case the dictionary is empty so we
# use the defaults, as they are defined in
# main_train.py in train_helper().
hparams = {}

# --------------------------------------------------------

#################
# STEP 7 ########
#################

train_ = True
if(len(sys.argv) == 2):
    if sys.argv[1] == "evaluate":
        print("loading best.pickle for models. no training.")
        train_ = False

step_seperator(7)

model_type = "CNN"
hparams = {}

#######################################
# train CNN on beat-synced spectrograms
#######################################

dataset_type = "spectrogram_beat"


model_name = "CNN2d_Classifier_" + dataset_type

if train_:
    train_helper(
                    model_name,
                    model_type,
                    CNNBackbone,
                    Classifier,
                    dataset_type,
                    regression = False,
                    epochs=EPOCHS,
                    device=device,
                    hparams=hparams
                )

######################
# evaluate on test set 
######################

fpath = Path("../checkpoints/" + model_name + "/"  + "best.pickle")
eval_path = Path("evaluate.py")
os.system(f"python {eval_path} {dataset_type} {fpath}")

###########################
# train CNN on spectrograms 
###########################

dataset_type = "spectrogram"
model_name = "CNN2d_Classifier_" + dataset_type

if train_:
    train_helper(
                    model_name,
                    model_type,
                    CNNBackbone,
                    Classifier,
                    dataset_type,
                    regression = False,
                    epochs=EPOCHS,
                    device=device,
                    hparams=hparams
                )

######################
# evaluate on test set 
######################

fpath = Path("../checkpoints/" + model_name + "/"  + "best.pickle")
eval_path = Path("evaluate.py")
os.system(f"python {eval_path} {dataset_type} {fpath}")


#################
# STEP 8 ########
#################

step_seperator(8)


# relation between numbers/ids and the task
markings = {1: "valence", 2:"energy", 3:"danceability"}

spearman_corr_list = []
dataset_type = "multitask_beat"
model_type = "CNN"
hparams = {}


for i in range(1, 4): # for each regression task

    print("###################################")
    print(f"Estimation for {markings[i]}:")
    print("###################################")


    # train the model
    _, spearman_corr = train_helper(
                                   "CNN_Regression_" + dataset_type + "_"+ markings[i],
                                    model_type,
                                    CNNBackbone,
                                    Regressor,
                                    dataset_type,
                                    regression = i,
                                    epochs=EPOCHS,
                                    device=device,
                                    hparams=hparams
                      )

    # save its performance
    spearman_corr_list.append(spearman_corr)
    

print("#########################################################################################################################")
print("#########################################################################################################################")

print("Using CNN 2D model:")
print(f"Mean Spearman correlation between ground truth and predicted values (in validation set) for all axis is equal to {np.mean(spearman_corr_list):.4f}.")

print("#########################################################################################################################")
print("#########################################################################################################################")



#################
# STEP 9 ########
#################

step_seperator(9)


# We perform transfer learning from the CNN trained on spectrogram_beat
# to the task of energy regression. More about our choices in the report.

model_type = "CNN"
dataset_type = "multitask_beat"

saved_model_path = "../checkpoints/CNN2d_Classifier_spectrogram_beat/best.pickle"

regression_task = 2 # energy

train_helper(
               "CNN2d_Regression_" + dataset_type + "_"+markings[regression_task],
                model_type,
                CNNBackbone,
                Regressor,
                dataset_type,
                regression = regression_task,
                epochs=EPOCHS,
                device=device,
                hparams=hparams, 
                transfer = saved_model_path
            )

# train_helper automatically evaluates the best model.
# So the Spearman Correlation will be printed in stdout.



#################
# STEP 10 #######
#################

step_seperator(10)

# just train a mutlitask model

model_type = "CNN"
dataset_type = "multitask_beat"

train_helper(
               "CNN2d_MultitaskRegression_" + dataset_type,
                model_type,
                CNNBackbone,
                MultitaskRegressor,
                dataset_type,
                regression = ":",
                epochs=EPOCHS,
                device=device,
                hparams=hparams, 
            )

# train_helper automatically evaluates the best model.
# So the Spearman Correlation will be printed in stdout.
