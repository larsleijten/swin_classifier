import csv
import os
from tiger.io import read_image, write_image
import numpy as np

# Create a file where the labels are stored

with open('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/data/labels.csv', mode='w', newline='') as output_file:
    writer = csv.writer(output_file)
    writer.writerow(["PatchID", "label"])

# Open the CSV file with the candidates
with open('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/data/candidates_V2.csv', newline='') as input_file:
    
    # Create a CSV reader object
    reader = csv.reader(input_file)
    patch_id = 0
    prev_scanID = ""
    image = None
    header = None

    # Loop through each row in the file
    for candidate in reader:
        if(candidate[0]!= "seriesuid" and os.path.isfile(os.path.join('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/data/ct_images/combined', candidate[0] + ".nii.gz"))):
            # Extract the information from the row
            scanID = candidate[0]
            coordX = float(candidate[1])
            coordY = float(candidate[2])
            coordZ = float(candidate[3])
            label = int(candidate[4])
 
            # Load a new CT file if we move on to patches in a new file
            if (prev_scanID != scanID):
                print("Working on CT scan with ID: " + scanID)
                ct_file_path = os.path.join('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/data/ct_images/combined', scanID + ".nii.gz")
                temp_image, header = read_image(ct_file_path)
            
                # Manually implement zero padding, which is less computationally expensive
                # Padding is added because the patches can partly lay outside the CT frame
                image = np.zeros(
                    (temp_image.shape[0] + 96, temp_image.shape[1] + 96, temp_image.shape[2] + 96)
                )
                image[49:49+temp_image.shape[0], 49:49+temp_image.shape[1], 49:49+temp_image.shape[2]] = temp_image
                prev_scanID = scanID
        

            # Save a patch as training/test data if it has a suspect tumour, or one in 250 cases when negative (to avoid creating 700,000 patch_files)
            if (label == 1 or np.random.randint(0, 250) == 99):

                coordX = int((coordX - header.origin[0]) / header.spacing[0])
                coordY = int((coordY - header.origin[1]) / header.spacing[1])
                coordZ = int((coordZ - header.origin[2]) / header.spacing[2])
                
                # This takes the padding into account
                startX = coordX
                stopX = coordX + 96
                startY = coordY
                stopY = coordY + 96
                startZ = coordZ 
                stopZ = coordZ + 96
                
                # Save the patch
                new_image = image[startX:stopX, startY:stopY, startZ:stopZ]
                patch_path = os.path.join('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/data/ct_images/patches', str(patch_id) + ".nii.gz")
                write_image(patch_path, new_image, header)
                with open('/mnt/netcache/bodyct/experiments/scoliosis_simulation/luna/swin_classifier/data/labels.csv', mode='a', newline='') as output_file:
                    writer = csv.writer(output_file)
                    writer.writerow([patch_id, label])
                
                patch_id+=1




            

                