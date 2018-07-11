

import numpy as np
import random
from numba.decorators import jit

@jit
def noise(array):
  print('now noising') 
  height = len(array)
  width = len(array[0])
  print('start rand')  
  rands = np.random.uniform(0, 1, (height, width) )
  print('finish rand')  
  copy  = np.copy(array)

  for h in range(height):
    for w in range(width):
      if rands[h, w] <= 0.10:
        swap_target_h = random.randint(0,h)
        copy[h, w] = array[swap_target_h, w]
      #print(rands[h,w])
  print('finish noising') 
  return copy
  
