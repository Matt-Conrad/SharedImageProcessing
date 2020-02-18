import numpy as np
from scipy.ndimage.measurements import label

def getBiggestComp(image):
    """ Uses connected components to get the breast """
    structure = np.ones([3,3], dtype=np.int) # Relational matrix (8-connected)
    # Run connected components to label the various connected components
    labeled_image, n_components = label(image, structure=structure) 

    # Loop through the components and get the biggest component
    nPixelsInBiggestComp = 0
    biggestComp = 0
    for i in range(1,n_components+1): # Start at 1 to avoid considering background 
        component = (labeled_image == i)
        pixelsInComp = np.sum(component)
        if pixelsInComp > nPixelsInBiggestComp:
            nPixelsInBiggestComp = pixelsInComp
            biggestComp = component

    # Create binary mask in the shape of the biggest component
    img = np.zeros(image.shape)
    img[biggestComp] = 1
    return img

def getBiggestComp2(image):
    """ Uses connected components to get the breast """
    structure = np.ones([3,3], dtype=np.int) # Relational matrix (8-connected)
    # Run connected components to label the various connected components
    labeled_image, n_components = label(image, structure=structure) 

    counts = np.bincount(labeled_image.flatten())
    ind = np.argmax(counts[1:]) + 1
    biggestComp = (labeled_image == ind).astype(np.uint8)

    return biggestComp