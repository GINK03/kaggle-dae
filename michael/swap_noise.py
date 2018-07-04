

import numpy as np
import random
def noise(array):
  print('now noising') 
  height = len(array)
  width = len(array[0])

  copy = np.copy(array)

  for h in range(height):
    for w in range(width):
      if random.random() <= 0.15:
        swap_target_h = random.randint(0,h)
        copy[h, w] = array[swap_target_h, w]

  print('finish noising') 
  return copy
  
