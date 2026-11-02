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

# Step 2 - gradient_check (not yet solved)
# TODO: implement

# Step 3 - make_dense (not yet solved)
# TODO: implement

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

