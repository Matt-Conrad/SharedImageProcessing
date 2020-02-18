import numpy as np
from scipy.ndimage.measurements import label

def getBiggestComp(image):
    """ Uses connected components to get the breast """
    structure = np.ones([3,3], dtype=np.int) # Relational matrix (8-connected)
    # Run connected components to label the various connected components
    labeled_image, n_components = label(image, structure=structure) 

    counts = np.bincount(labeled_image.flatten())
    ind = np.argmax(counts[1:]) + 1
    biggestComp = (labeled_image == ind).astype(np.uint8)

    return biggestComp