from lib import *

from helper_scripts.parser import *
from helper_scripts.lstm import *


BATCH_SIZE = 16

#torch.set_num_threads(4)
print(f"Using {torch.get_num_threads()} cores.")


# parse data
X_train, X_test, y_train, y_test, spk_train, spk_test = parser('dataset/free-spoken-digit-dataset/recordings')



# split data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

# normalize data
scale_fn = make_scale_fn(X_train)
X_train = scale_fn(X_train)
X_val = scale_fn(X_val)
X_test = scale_fn(X_test)


# =========== STEP 14 ===========


# Create the pytorch datasets
train_dataset = FrameLevelDataset(X_train, y_train)
val_dataset = FrameLevelDataset(X_val, y_val)
test_dataset = FrameLevelDataset(X_test, y_test)


train_loader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True
    )


val_loader = DataLoader(
        val_dataset,
        batch_size = BATCH_SIZE,
        shuffle = False
    )

test_loader = DataLoader(
        test_dataset,
        batch_size = BATCH_SIZE,
        shuffle = False
    )

# ===============================================
# LSTM without dropout and without early stopping
# ===============================================


#define LSTM model's parameters

input_dim = len(X_train[0][0])
rnn_size = 15
num_layers=2
epochs=50



 #  create the model
simple_lstm = BasicLSTM(input_dim=input_dim, 
                        rnn_size=rnn_size, 
                        output_dim=10, 
                        num_layers=num_layers, 
                        bidirectional=False,
                        dropout=0
                    )

# the optimized
optimizer = torch.optim.Adam(simple_lstm.parameters(), lr=1e-3)


# train the model
train_acc, val_acc, train_loss, val_loss = train_LSTM_model(
    simple_lstm,
    train_loader,
    val_loader,
    optimizer,
    epochs=epochs)

plot_leaning_curves_lstm(
    train_acc,
    val_acc,
    train_loss,
    val_loss,
    "Simple LSTM",
    "figures/step_14_lstm_simple.png")



create_cm_lstm(
    simple_lstm,
    val_loader,
    "Simple LSTM | Validation Set", 
    "figures/step_14_cm_val_lstm_simple.png")

create_cm_lstm(
    simple_lstm,
    test_loader,
    "Simple LSTM | Test Set", 
    "figures/step_14_cm_test_lstm_simple.png")


val_accuracy = accuracy_lstm_on_dataloader(simple_lstm, val_loader)
print(f"Simple LSTM val accuracy : {val_accuracy *100: .2f} %")

test_accuracy = accuracy_lstm_on_dataloader(simple_lstm, test_loader)
print(f"Simple LSTM test accuracy : {test_accuracy *100: .2f} %")

# ==================================================================
# LSTM with dropout and L2 regularization and without early stopping
# ==================================================================

dropout_lstm = BasicLSTM(input_dim=input_dim, 
                        rnn_size=rnn_size, 
                        output_dim=10, 
                        num_layers=num_layers, 
                        bidirectional=False,
                        dropout=0.65
                    )

# the optimized
optimizer = torch.optim.Adam(dropout_lstm.parameters(), lr=1e-3, weight_decay=1e-4)


# train the model
train_acc, val_acc, train_loss, val_loss = train_LSTM_model(
    dropout_lstm,
    train_loader,
    val_loader,
    optimizer,
    model_name = "dropout_l2_lstm",
    epochs=epochs)

plot_leaning_curves_lstm(
    train_acc,
    val_acc,
    train_loss,
    val_loss,
    "LSTM with dropout and L2",
    "figures/step_14_lstm_dropout_L2.png")

    

create_cm_lstm(
    dropout_lstm,
    val_loader,
    "LSTM with dropout and L2 | Validation Set", 
    "figures/step_14_cm_val_lstm_dropout_L2.png")

create_cm_lstm(
    dropout_lstm,
    test_loader,
    "LSTM with dropout and L2 | Test Set", 
    "figures/step_14_cm_test_lstm_dropout_L2.png")

val_accuracy = accuracy_lstm_on_dataloader(dropout_lstm, val_loader)
print(f"LSTM with dropout and L2 val accuracy : {val_accuracy * 100: .2f} %")

test_accuracy = accuracy_lstm_on_dataloader(dropout_lstm, test_loader)
print(f"LSTM with dropout and L2 test accuracy : {test_accuracy * 100: .2f} %")


# ===============================================================
# LSTM with dropout and L2 regularization and with early stopping
# ===============================================================

dropout_lstm = BasicLSTM(input_dim=input_dim, 
                        rnn_size=rnn_size, 
                        output_dim=10, 
                        num_layers=num_layers, 
                        bidirectional=False,
                        dropout=0.65
                    )

# the optimized
optimizer = torch.optim.Adam(dropout_lstm.parameters(), lr=1e-3, weight_decay=1e-4)


# train the model
train_acc, val_acc, train_loss, val_loss = train_LSTM_model(
    dropout_lstm,
    train_loader,
    val_loader,
    optimizer,
    epochs=epochs, 
    model_name = "dropout_l2_early_lstm",
    early_stopping=True)

plot_leaning_curves_lstm(
    train_acc,
    val_acc,
    train_loss,
    val_loss,
    "LSTM with dropout and L2 (early stopping)",
    "figures/step_14_lstm_dropout_L2_early_stopping.png")

create_cm_lstm(
    dropout_lstm,
    val_loader,
    "LSTM with dropout and L2 (early stopping) | Validation Set", 
    "figures/step_14_cm_val_lstm_dropout_L2_early_stopping.png")

create_cm_lstm(
    dropout_lstm,
    test_loader,
    "LSTM with dropout and L2 (early stopping) | Test Set", 
    "figures/step_14_cm_test_lstm_dropout_L2_early_stopping.png")

val_accuracy = accuracy_lstm_on_dataloader(dropout_lstm, val_loader)
print(f"LSTM with dropout and L2 and early stopping val accuracy : {test_accuracy*100 : .2f} %")

test_accuracy = accuracy_lstm_on_dataloader(dropout_lstm, test_loader)
print(f"LSTM with dropout and L2 and early stopping test accuracy : {test_accuracy*100 : .2f} %")


# =============================================================================
# Bidirectional LSTM with dropout and L2 regularization and with early stopping
# =============================================================================


bidirectional_lstm = BasicLSTM(input_dim=input_dim, 
                        rnn_size=rnn_size, 
                        output_dim=10,
                        num_layers=num_layers, 
                        bidirectional=True,
                        dropout=0.65
                    )

# the optimized
optimizer = torch.optim.Adam(bidirectional_lstm.parameters(), lr=1e-3, weight_decay=1e-4)


# train the model
train_acc, val_acc, train_loss, val_loss = train_LSTM_model(
    bidirectional_lstm,
    train_loader,
    val_loader,
    optimizer,
    epochs=epochs, 
    model_name = "dropout_l2_early_bidirectional_lstm",
    early_stopping=True)

plot_leaning_curves_lstm(
    train_acc,
    val_acc,
    train_loss,
    val_loss,
    "Bidirectional LSTM with dropout and L2 (early stopping) ",
    "figures/step_14_bidirectional_lstm_dropout_L2_early_stopping.png")

create_cm_lstm(
    bidirectional_lstm,
    val_loader,
    "Bidirectional LSTM with dropout and L2 (early stopping) | Validation Set", 
    "figures/step_14_cm_val_bidirectional_lstm_dropout_L2_early_stopping.png")

create_cm_lstm(
    bidirectional_lstm,
    test_loader,
    "Bidirectional LSTM with dropout and L2 (early stopping) | Test Set", 
    "figures/step_14_cm_test_bidirectional_lstm_dropout_L2_early_stopping.png")

val_accuracy = accuracy_lstm_on_dataloader(bidirectional_lstm, val_loader)
print(f"Bidirectional LSTM with dropout, L2 and early stopping val accuracy : {test_accuracy * 100 : .2f} %")

test_accuracy = accuracy_lstm_on_dataloader(bidirectional_lstm, test_loader)
print(f"Bidirectional LSTM with dropout, L2 and early stopping test accuracy : {test_accuracy * 100 : .2f} %")

