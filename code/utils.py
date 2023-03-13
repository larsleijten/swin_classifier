import torch

# Declare validation function 
def validation(val_loader, model, loss_fn, device):
    size = len(val_loader.dataset)
    num_batches = len(val_loader)
    model.eval()
    test_loss, correct, TP, TN, FP, FN = 0, 0, 0, 0, 0, 0
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
            TP += (bin_pred == y and bin_pred == torch.tensor(1.0)).type(torch.float).sum().item()
            TN += (bin_pred == y and bin_pred == torch.tensor(0.0)).type(torch.float).sum().item()
            FP += (bin_pred != y and bin_pred == torch.tensor(1.0)).type(torch.float).sum().item()
            FN += (bin_pred != y and bin_pred == torch.tensor(0.0)).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    sensitivity = TP / (TP + FN) * 100
    specificity = TN / (TN + FP) * 100
    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n Sensitivity: {sensitivity:>0.1f}%, \n Specificity: {specificity:>0.1f}%" )
    
    return test_loss, (100*correct)