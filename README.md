# LC PreProcessing
 
 Software to automate scaling of images of brain scans/stains for further analysis.
A pretrained convultional neural network (ResNet50v2) is used as a network base. Additional layers and training were added. The Network was trained to recognize the borders of the locus coeruleus in each image, and then resize the shrunken nissl-stained images to align with images of stains that were not shrunken.
