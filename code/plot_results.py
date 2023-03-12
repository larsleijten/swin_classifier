import csv
import matplotlib.pyplot as plt



# Train Loss

# Set the path to the CSV files
train_csv = '/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/results/combined_train_10.csv'
validation_csv = '/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/results/combined_validation_10.csv'

# Create empty lists to store the data
train_losses = []

# Load the data from the CSV file
with open(train_csv, "r") as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        train_losses.append(float(row[0]))

# Create a line plot of the training loss over time
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks([])
plt.show()


# Validation 

val_losses = []
val_accuracies = []

# Load the data from the CSV file
with open(validation_csv, "r") as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        val_losses.append(float(row[0]))
        val_accuracies.append(float(row[1]))

# Create a  plot of the validation loss over time
plt.plot(range(1, 11), val_losses)
plt.title("Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()



# Validation Accuracy
plt.plot(range(1, 11), val_accuracies)
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()









