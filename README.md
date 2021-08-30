# Locus Coeruleus Pre-processing
 
## Software to automate scaling of images of brain stains for further analysis.
A pretrained convultional neural network (ResNet50v2) is used as a network base. Additional layers and training were added. The Network was trained to recognize the borders of the area containing the locus coeruleus in each image, and then resize the shrunken nissl-stained images to align with images of stains that were not shrunken.

An anaconda environment for use with the code is included in <code>environment.yml</code>. Nvidia CuDnn libraries will need to be added to the enviornment's directories. Models are stored in the Output subdirectory and should be unzipped with 7zip before use.

Current network performs well on images but will need to be retrained to perform well on differently formatted images that Idon't currently have access to
When access to the rest of the dataset is granted:
Select flourescent and nissl stained images from different animals to include in training.
Run Hyperparameter search, evaluate new network.

### Model Building
<code>Model Building.py</code> is used to construct and validate the CNN. Performs cross validation on the image set available to evaluate performance
Data.csv contains the annotated data used for training and validation. Has image name and coordinates of lower left / upper right points bounding area contianing the LC to detect. Also contains flag, 0 if flourescent and 1 if nissil. Images referenced in this table are in the Images subdirectory.

### Model Testing
<code>Model Testing.py</code> can be used to test the CNN on other images. A more representative training set should be built when access to the full dataset is granted to see how the model generalizes. Images for testing are kept in the New Images subdirectory and annotations are made in the Data.csv table.

### Model Utilization
<code>Scaling.py</code> should be used when a satisfactory model has been trained. Run this in the directory of a brain's images to scale the nissil images to the appropriate size. This creates a new directory of images that can be loaded into Amira for analysis.
