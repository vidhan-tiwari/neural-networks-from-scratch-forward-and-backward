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

# Step 4 - make_activation (not yet solved)
# TODO: implement

# Step 5 - initialize_weights (not yet solved)
# TODO: implement

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

