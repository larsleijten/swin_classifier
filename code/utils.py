import torch

# Declare validation function 
def validation(val_loader, model, loss_fn, device):
    size = len(val_loader.dataset)
    num_batches = len(val_loader)
    model.eval()
    test_loss, correct = 0, 0
    # Do not keep track of gradients
    with torch.no_grad():
        # Loop over the batches in the dataloader
        for X, y in val_loader:
            X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.float)
            
            # Model predictions
            X = X[None, :].to(device=device, dtype = torch.float)
            with torch.cuda.amp.autocast():
                pred = model(X).to(device, dtype = torch.float)
            
            

            # Make a binary prediction at the threshold of 0.5
            bin_pred = torch.round(pred.unsqueeze(1))#.transpose(0,1)
            # Keep track of loss and accuracy
            test_loss += loss_fn(pred.unsqueeze(1), y.unsqueeze(1).float()).item()
            correct += (bin_pred == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return test_loss, (100*correct)