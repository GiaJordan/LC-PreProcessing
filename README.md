# Locus Coeruleus Pre-processing
 
## Software to automate scaling of images of brain scans/stains for further analysis.
A pretrained convultional neural network (ResNet50v2) is used as a network base. Additional layers and training were added. The Network was trained to recognize the borders of the locus coeruleus in each image, and then resize the shrunken nissl-stained images to align with images of stains that were not shrunken.


Current network performs well on images but will need to be retrained to perform well on differently formatted, collected images/stains.
When access to the rest of the dataset is granted:
Select flourescent and nissl stained images from different animals to include in training.
Run Hyperparameter search, evaluate new network.