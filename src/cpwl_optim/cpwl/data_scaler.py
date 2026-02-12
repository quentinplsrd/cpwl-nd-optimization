import numpy as np


def rescale_data(data: np.ndarray,
                 between_one_and_two: bool = True):

    """Rescale the data set.

    Args:
      data: The data set to rescale. 
      
      between_one_and_two: If the data should be rescaled to the range [1,2] instead of [0,1].
      
    Returns:
      The rescaled data, the transformation coefficients (slopes and intercepts).
      
      `data` can be recovered by applying: slopes*rescaled_data + intercepts
      
    """    
    
    # --- Type checks ---
    if not isinstance(data, np.ndarray):
        raise TypeError(f"`data` must be a numpy.ndarray, got {type(data).__name__}.")
    if data.ndim != 2:
        raise ValueError(f"`data` must be a 2D array, got shape {data.shape}.")
    if not isinstance(between_one_and_two, bool):
        raise TypeError("`between_one_and_two` must be a boolean.")
        
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    
    rescaled_data = (data - data_min) / (data_max - data_min)
    rescaled_data += 1*between_one_and_two
        
    slopes = data_max - data_min
    intercepts = data_min - between_one_and_two*slopes
    
    return rescaled_data, slopes, intercepts


def invert_transform(slopes, intercepts):
    return 1/slopes, -intercepts/slopes
