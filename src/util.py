import matplotlib.pyplot as plt

def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    # Get rid of the . in learning rate
    lr_arr = str(learning_rate).split('.')
    learning_rate = lr_arr[0] + '_' + lr_arr[1]
    
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name, batch_size, learning_rate, epoch)
    return path

def plot_training_curve(model_name, iters, epochs, train_losses, val_losses, train_acc, val_acc):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10,3))
    
    # Training Loss
    ax1.set_title("Training Loss")
    ax1.plot(iters, train_losses, label="Training")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")
    
    # Training Accuracy
    ax2.set_title("Training Accuracy")
    ax2.plot(iters, train_acc, label="Training")   
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Training Accuracy")
    
    # Validation Accuracy
    ax3.set_title("Validation Accuracy")
    ax3.plot(epochs, val_acc, label="Validation")    
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Validation Accuracy")

    # Validation losss
    ax4.set_title("Validation Loss")
    ax4.plot(epochs, val_losses, label="Validation")    
    ax4.set_xlabel("Epochs")
    ax4.set_ylabel("Validation Loss")
    
    fig.tight_layout()
    plt.savefig('../training_curves/' + model_name + '.png')