"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
def numerical_gradient(f, x, eps=1e-5):

    grads = np.zeros(x.shape , dtype = float)
    x_copy = np.array(x, dtype = float, copy = True)
    # np.nd_index help iterate through every single element of numpy array 
    # as we need differentiation w.r.t.every element in weight or bias array 

    # we change one element at a time and calculate the value of f() for intire input.
    # that helps to measure change w.r.t. single element

    '''
    for ith element 
    d(f)/d(x) = (f(x+eps) -f(x-eps))/2*eps
    at a time only one element of X is perturbated
    '''
    for idx in np.ndindex(x.shape):
        # save the original value 
        orig = x_copy[idx]
        # perturbate this element by adding eps
        x_copy[idx] = orig + eps
        # calculate f(x+eps)
        f_plus = f(x_copy)
        # restore the perturbation 
        x_copy[idx] = orig
        # now do same to calculate f_minus

        x_copy[idx] = orig - eps
        f_minus = f(x_copy)
        x_copy[idx] = orig

        grads[idx] = (f_plus - f_minus) / (2.0 * eps)
    return grads

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    '''
    mathematics to be implemented 
    rel_error = |a-n|\max(|a|,|n|,tol)
    '''
    a = analytic_grad
    n = numeric_grad
    # np.maximum = calculates maximum value of two numpy arrays and returns a new one with maximum element values 
    # np.max() is a reduction operation that converts nparray to single value 
    rel_error = np.abs(a-n)/np.maximum(np.maximum(np.abs(a), np.abs(n)), tol)
    return float(np.max(rel_error))

# Step 3 - make_dense
def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer.

    Inputs:
      in_dim: int, input feature size
      out_dim: int, output feature size
      weight_init_fn: callable(in_dim, out_dim) -> (W, b)

    Returns layer dict with keys:
      params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
      forward(x) -> (y, cache) with y shape (batch, out_dim)
      backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
        Analytic dx/dW/db must match numerical_gradient via gradient_check.
    """
    # TODO: your approach here
    W, b = weight_init_fn(in_dim,out_dim)
    
    # incase the return type of parameters is not numpy array
    params = {
      'W' : np.asarray(W, dtype = float),
      'b' : np.asarray(b , dtype = float)
    }

    def forward(x):
      x = np.asarray(x, dtype = float)
      y = x @ params['W'] + params['b']
      cache = x
      return (y,cache)

    def backward(dout,cache):
      dout = np.asarray(dout, dtype = float)
      x = cache

      # analytical gradients derived by gradient chain rule 

      dx = dout @ params['W'].T # this dx is used to back propagate gradient i.e works as 'dout' parameter for the previous layer.
      dW = x.T @ dout
      db = np.sum(dout,axis = 0 )
      grads = {
        'W' : dW,
        'b' : db
      }
      return dx , grads

    return {
      'params' : params,
       'forward' : forward,
       'backward' : backward
    }

# Step 4 - make_activation
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer.

    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
              (zero negatives, pass non-negatives). Other kinds optional.

    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)

    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    # TODO: your approach here

    if kind != 'relu':
        raise NotImplementedError(f"Unsupported activation kind or not implemented yet: {kind}")

    def forward(x):
      # making cache as a copy of original array 
      cache = np.array(x, dtype = float , copy = True)
      y = np.where(x > 0 , x , 0)
      return (y, cache) 

    def backward(dout , cache):
      dx = np.where(cache > 0 , 1.0 , 0.0)
      # now apply the chain rule
      dx = dx * dout
      return (dx , {})

    return {
      'forward' : forward,
      'backward' : backward,
      'params' : {} # as activation layer has no learnable parameters.
    }

# Step 5 - initialize_weights
def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer.

    Inputs:
      in_dim: int fan-in
      out_dim: int fan-out
      scheme: str initialization family (default 'he')

    Returns:
      W: np.ndarray shape (in_dim, out_dim), finite, symmetry-breaking,
         scale stable with depth (fan-in dependent)
      b: np.ndarray shape (out_dim,), near zero
    """
    # TODO: your approach here
    scheme = scheme.lower() 

    if scheme == 'he' or scheme == 'xavier': # implement he-normal
      mean = 0.0

      if scheme == 'he':
        std = np.sqrt(2.0/in_dim)
      else:
        std = np.sqrt(2.0/(in_dim + out_dim))

      W = np.random.normal(loc = mean , scale = std , size = (in_dim , out_dim))

    elif scheme == 'he_uniform' or scheme == 'xavier_uniform': # implement he-uniform
      
      if scheme == 'he_uniform':
        limit = np.sqrt(6.0/in_dim)

      if scheme == 'xavier_uniform':
        limit = np.sqrt(6.0/(in_dim + out_dim))

      W = np.random.uniform(-limit , limit , size = (in_dim , out_dim))
    else :
      raise NotImplementedError(f'unfamiliar weight initialisation technique {scheme}')

    b = np.zeros(out_dim , dtype = float)
    return W,b

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

