import numpy as np
import math
import torch
from torch import nn
from torch.optim import AdamW, lr_scheduler
from lion_pytorch import Lion

from .utils import my_spearmanr, my_pearsonr


def getNumParams(model, print=False):
    """
    Get the number of (trainable) parameters in a model
    """
    if print:
        print("Model structure:")
        print(model)
        print('\n')
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def inputShape(DataLoader):
    X, y = next(iter(DataLoader))
    print(f"\nShape of X [N, C, H, W]\t\t: {X.shape}")
    input_channels = X.shape[1] # Get the number of channels
    feature_height = X.shape[2] # Get the height of the input
    feature_width = X.shape[3] # Get the width of the input
    print(f"Shape of y (type)\t\t: {y.shape} ({y.dtype})")
    print(f"Total training instances\t: {len(DataLoader.dataset)}\n")
    return input_channels, feature_height, feature_width


def createOptimizer(model, config):
    """
    Create an optimiser based on some choice
    """
    mapping = {
        "Lion" : Lion(model.parameters(), lr=config['lr'], betas=(0.9, 0.99), weight_decay=config['weight_decay']),
        "AdamW": AdamW(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), weight_decay=config['weight_decay'])
    }
    return mapping[config["optimizer"]]

def createLoss(config):
    """
    Create a Loss function object
    """
    mapping = {
        "MSE"   : nn.MSELoss().to(config["device"]), #Smooth loss
        "L1"    : nn.L1Loss().to(config["device"]), #Absolute Error
        "Huber" : nn.HuberLoss(delta=0.9).to(config["device"]) # Huber loss, but close to SmoothL1
    }
    return mapping[config["loss"]]


def train(TrainLoader, model, config, ValLoader):
    """
    Training a model with provided structure and parameters.
    Loss function, Optimiser and LR scheduler are created here and used.

    Retuns a trained model
    """
    epochs = config["epochs"]
    patience = config["patience"]
    device = config["device"]
    best_model_path = config["best_model_path"]

    #create loss and optimizer
    loss_fn = createLoss(config=config)
    optimizer = createOptimizer(model=model, config=config)

    #Scheduler
    def createLRScheduler(config):
        """
        Create a LR scheduler object
        """
        if config["scheduler"] == "OneCycleLR":
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(TrainLoader), epochs=config["epochs"])
        elif config["scheduler"] == "ReduceLROnPlateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        else:
            scheduler = None
        return scheduler

    scheduler = createLRScheduler(config=config)

    # Store some key variables
    #train_losses = [] # store history
    #val_losses = [] # store history
    best_score = -1 # value for early stopping
    run_val_rho = 0
    epochs_no_improve = 0
    best_epoch = 0
    
    # print output header
    print("\t\tT loss\tV loss\tT r\tV r\tT rho\tV rho")
    
    for epoch in range(epochs): # Train for the stipulated number of epochs (or untill early stopping)
            
        model.train() # Set model to training mode
        
        train_loss = 0
        y_pred = []
        y_true = []
        for (X, y) in TrainLoader: # Returns X and y in batch-sized chunks. 
            X, y = X.to(device), y.to(device)
    
            # Compute loss
            pred = model(X)
            loss = loss_fn(pred, y)
    
            # Backpropagation
            optimizer.zero_grad() # clears old gradients from the last step 
            loss.backward() # computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
            optimizer.step() # causes the optimizer to take a step based on the gradients of the parameters.
            #xm.mark_step() # xm-code

            if config["scheduler"] == "OneCycleLR":
                scheduler.step()
        
            # Save results
            train_loss += loss.item()
            y_pred.append(pred.cpu().detach())
            y_true.append(y.cpu().detach())
        
        train_loss /= len(TrainLoader)*1.0 # Divide the loss by the number of batches
        #train_losses.append(train_loss)
        
        # After completing training, evaluate model on the validation set.
        
        # Store some key variables for validation.
        valid_loss = 0
        y_pred_val = []
        y_true_val = []

        model.eval()
        with torch.no_grad(): # Turn off gradient calculation for inference.
            for (X, y) in ValLoader:
                X, y = X.to(device), y.to(device)
    
                # Compute prediction error
                pred = model(X)
                loss = loss_fn(pred, y)
                #xm.mark_step() # xm-code
                
                # Save results
                valid_loss += loss.item()
                y_pred_val.append(pred.cpu().detach())
                y_true_val.append(y.cpu().detach())
        
        valid_loss /= len(ValLoader)*1.0
        #val_losses.append(valid_loss)

        if config["scheduler"] == "ReduceLROnPlateau":
            scheduler.step(valid_loss)
        
        # Check for improvement in validation scores.
        #if min_val_loss - valid_loss >= threshold:
        run_val_rho = my_spearmanr(np.concatenate(y_pred_val), np.concatenate(y_true_val))
        run_val_r = my_pearsonr(np.concatenate(y_pred_val), np.concatenate(y_true_val))
        val_score = run_val_rho + run_val_r
        if val_score > best_score:
            #torch.save(model.state_dict(), best_model_path) # Save the model
            torch.save(model, best_model_path)
            epochs_no_improve = 0
            #min_val_loss = valid_loss
            best_epoch = epoch
            #best_r = stats.pearsonr(np.concatenate(y_pred_val), np.concatenate(y_hat_val))[0]
            best_score = val_score
        else:
            epochs_no_improve += 1
                
        # Output the results of the epoch
        output = [] # List of output
        output.append(f"Epoch {epoch+1}:")
        # Losses
        output.append(round(train_loss, 4))
        output.append(round(valid_loss, 4))
        # Pearson r
        output.append(round(my_pearsonr(np.concatenate(y_pred), np.concatenate(y_true)), 4))
        output.append(round(my_pearsonr(np.concatenate(y_pred_val), np.concatenate(y_true_val)), 4) if len(y_pred_val) > 0 else "-")
        # Spearman rho
        output.append(round(my_spearmanr(np.concatenate(y_pred), np.concatenate(y_true)), 4))
        output.append(round(run_val_rho, 4) if len(y_pred_val) > 0 else "-")
        if ValLoader and epochs_no_improve == 0: # mark the best epochs
            output.append("*")
        print("\t".join(str(x) for x in output))
        
        # Check for early stopping
        if epochs_no_improve == patience:
            print(f"Early stop, best epoch {best_epoch+1}")
            break

    return model


def train_W(TrainLoader, model, config, ValLoader):
    """
    Training a model with provided structure and parameters.
    Loss function, Optimiser and LR scheduler are created here and used.

    Retuns a trained model
    """
    epochs = config["epochs"]
    patience = config["patience"]
    device = config["device"]
    best_model_path = config["best_model_path"]

    #create loss and optimizer
    loss_fn = createLoss(config=config)
    optimizer = createOptimizer(model=model, config=config)

    #Scheduler
    def createLRScheduler(config):
        """
        Create a LR scheduler object
        """
        if config["scheduler"] == "OneCycleLR":
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(TrainLoader), epochs=config["epochs"])
        elif config["scheduler"] == "ReduceLROnPlateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        else:
            scheduler = None
        return scheduler

    scheduler = createLRScheduler(config=config)

    # Store some key variables
    #train_losses = [] # store history
    #val_losses = [] # store history
    best_score = -1 # value for early stopping
    run_val_rho = 0
    epochs_no_improve = 0
    best_epoch = 0
    
    # print output header
    print("\t\tT loss\tV loss\tT r\tV r\tT rho\tV rho")
    
    for epoch in range(epochs): # Train for the stipulated number of epochs (or untill early stopping)
            
        model.train() # Set model to training mode
        
        train_loss = 0
        y_pred = []
        y_true = []
        for (X, y, w) in TrainLoader: # Returns X and y in batch-sized chunks. 
            X, y, w = X.to(device), y.to(device), w.to(device)
    
            # Compute loss
            pred = model(X)
            loss = (loss_fn(pred, y) * w).mean()
    
            # Backpropagation
            optimizer.zero_grad() # clears old gradients from the last step 
            loss.backward() # computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
            optimizer.step() # causes the optimizer to take a step based on the gradients of the parameters.
            #xm.mark_step() # xm-code

            if config["scheduler"] == "OneCycleLR":
                scheduler.step()
        
            # Save results
            train_loss += loss.item()
            y_pred.append(pred.cpu().detach())
            y_true.append(y.cpu().detach())
        
        train_loss /= len(TrainLoader)*1.0 # Divide the loss by the number of batches
        #train_losses.append(train_loss)
        
        # After completing training, evaluate model on the validation set.
        
        # Store some key variables for validation.
        valid_loss = 0
        y_pred_val = []
        y_true_val = []

        model.eval()
        with torch.no_grad(): # Turn off gradient calculation for inference.
            for (X, y, w) in ValLoader:
                X, y, w = X.to(device), y.to(device), w.to(device)
    
                # Compute prediction error
                pred = model(X)
                loss = (loss_fn(pred, y) * w).mean()
                #xm.mark_step() # xm-code
                
                # Save results
                valid_loss += loss.item()
                y_pred_val.append(pred.cpu().detach())
                y_true_val.append(y.cpu().detach())
        
        valid_loss /= len(ValLoader)*1.0
        #val_losses.append(valid_loss)

        if config["scheduler"] == "ReduceLROnPlateau":
            scheduler.step(valid_loss)
        
        # Check for improvement in validation scores.
        #if min_val_loss - valid_loss >= threshold:
        run_val_rho = my_spearmanr(np.concatenate(y_pred_val), np.concatenate(y_true_val))
        run_val_r = my_pearsonr(np.concatenate(y_pred_val), np.concatenate(y_true_val))
        val_score = run_val_rho + run_val_r
        if val_score > best_score:
            #torch.save(model.state_dict(), best_model_path) # Save the model
            torch.save(model, best_model_path)
            epochs_no_improve = 0
            #min_val_loss = valid_loss
            best_epoch = epoch
            #best_r = stats.pearsonr(np.concatenate(y_pred_val), np.concatenate(y_hat_val))[0]
            best_score = val_score
        else:
            epochs_no_improve += 1
                
        # Output the results of the epoch
        output = [] # List of output
        output.append(f"Epoch {epoch+1}:")
        # Losses
        output.append(round(train_loss, 4))
        output.append(round(valid_loss, 4))
        # Pearson r
        output.append(round(my_pearsonr(np.concatenate(y_pred), np.concatenate(y_true)), 4))
        output.append(round(my_pearsonr(np.concatenate(y_pred_val), np.concatenate(y_true_val)), 4) if len(y_pred_val) > 0 else "-")
        # Spearman rho
        output.append(round(my_spearmanr(np.concatenate(y_pred), np.concatenate(y_true)), 4))
        output.append(round(run_val_rho, 4) if len(y_pred_val) > 0 else "-")
        if ValLoader and epochs_no_improve == 0: # mark the best epochs
            output.append("*")
        print("\t".join(str(x) for x in output))
        
        # Check for early stopping
        if epochs_no_improve == patience:
            print(f"Early stop, best epoch {best_epoch+1}")
            break

    return model


def predict(model, DataLoader, config):
    """
    Function to make predictions using the supplied model.
    
    Returns two np arrays, predicted values from the model, and true 
    values from the data loader.
    """
    
    model.eval()
    PredList=[] # Save predictions
    label_list = [] # Save true labels
    
    with torch.no_grad():
        for (data, label) in DataLoader:
            data = data.to(device=config["device"])
            pred = model(data)
            true_label = label.numpy()
            
            PredList.append(pred.cpu().detach().numpy())
            label_list.append(true_label)

    return np.concatenate(PredList), np.concatenate(label_list)


def predict_W(model, DataLoader, config):
    """
    Function to make predictions using the supplied model.
    
    Returns two np arrays, predicted values from the model, and true 
    values from the data loader.
    """
    
    model.eval()
    PredList=[] # Save predictions
    label_list = [] # Save true labels
    
    with torch.no_grad():
        for (data, label, weight) in DataLoader:
            data = data.to(device=config["device"])
            pred = model(data)
            true_label = label.numpy()
            
            PredList.append(pred.cpu().detach().numpy())
            label_list.append(true_label)

    return np.concatenate(PredList), np.concatenate(label_list)


def justPredict(model, DataLoader, config):
    """
    Function to make predictions using the supplied model. It doesn't return true values.
    
    Returns Predicted expressions for the sequences in the data loader
    """
    
    model.eval()
    PredList=[] 
    
    with torch.no_grad():
        for (data, label) in DataLoader:
            data = data.to(device=config["device"])
            pred = model(data)
            
            PredList.append(pred.cpu().detach().numpy())

    return np.concatenate(PredList)