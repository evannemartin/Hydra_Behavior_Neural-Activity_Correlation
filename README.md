# Hydra_Behavior_Neural-Activity_Correlation

Hydra vulgaris is a highly deformable aquatic animal that is characterized by simple behaviors such as contraction, elongation and bending. 
This animal is an interesting model in Neurosciences. Indeed, this is one of the first animal of the evolution that has neurons.  The organization of animal neurons in a nerve net, and the transparency of the animal allow the imaging of almost all neurons’ activity using fluorescent calcium probes (GCaMP) and imaging with time-lapse confocal microscopy.
Previous studies have shown that the activity of specific neuronal ensembles is associated with stereotypical behaviors. The use of machine learning could help to automatize the classification of animal behaviors, and to elucidate the neural code of the animal. 

The aim of this project is to implement a method using tracking, image quantification and classification to correlate specific behaviors with region-dependent neural activity. For this, we extracted the different behaviors during a video (such as elongation, contraction and bending) and neural activity data by tracking GCaMP fluorescence. 
To automatically extract animal behaviors, we used DeepLabCut to track points of interest within the freely-behaving animal. With the data generated, the characterization of the behavior was possible by computing the variation of animal length and head angle. 
To extract the neural activity, a segmentation into the 3 main regions (“head”, “feet” and “mid-body”) of the animal has been made in order to compute the average intensity of the fluorescence. 
Finally, we used a LSTM neural network to predict behavior from the extracted neural activity in the different regions of the animal.
