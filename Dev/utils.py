from os import walk
import numpy as np
import nibabel as nib




def load_nii(img_path):
    """
    Parameters
    ----------

    img_path: string
    String with the path of the 'nii' or 'nii.gz' image file name.

    Returns
    -------
    Three element, the first is a numpy array of the image values,
    the second is the affine transformation of the image, and the
    last one is the header of the image.
    """
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header

def standardize(image):
    
    # initialize to array of zeros, with same shape as the image
    standardized_image = np.zeros(image.shape)

    # iterate over channels
    for c in range(image.shape[2]):
        # get a slice of the image 
        # at channel c and z-th dimension `z`
        image_slice = image[:,:,c]
        # subtract the mean from image_slice
        centered = image_slice - np.mean(image)
        # divide by the standard deviation (only if it is different from zero)
        if np.std(centered) != 0:
            centered_scaled = centered/np.std(centered)

            # update  the slice of standardized image
            # with the scaled centered and scaled image
            standardized_image[ :, :,c] = centered_scaled
        else:
            standardized_image[ :, :,c] = centered

    return standardized_image
# Load test images
def load_test_data(data_path,nb_samples=10):
  X = []
  files = []
  affines = []
  headers = []

  count = nb_samples

  for (_,_,filenames) in walk(data_path):
    for filename in filenames :

      img_data,affine,header = load_nii(data_path+filename)
      img_data = standardize(img_data)
      img_data = img_data.transpose((2,0,1))
      img_data = np.expand_dims(img_data,axis=-1)
      
      files.append(filename)
      affines.append(affine)
      headers.append(header)
      #image = tf.convert_to_tensor(img_data,tf.float32)
      X.append(img_data)

      count -= 1

      if count == 0:
        return X, files,affines,headers  
          
  return X, files,affines,headers  