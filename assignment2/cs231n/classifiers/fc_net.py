import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    self.c = num_classes
    self.d = input_dim
    self.m = hidden_dim
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    self.params['W1'] = np.random.normal(0, weight_scale, (self.d, self.m))
    self.params['W2'] = np.random.normal(0, weight_scale, (self.m, self.c))
    self.params['b1'] = np.zeros(self.m)
    self.params['b2'] = np.zeros(self.c)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    W1, b1, W2, b2 = self.params['W1'], \
                        self.params['b1'], \
                        self.params['W2'], \
                        self.params['b2']
    hidden_layer, cache_hidden_layer = affine_relu_forward(X, W1, b1) # hidden layer
    scores, cache_scores = affine_forward(hidden_layer, W2, b2) # output layer

    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores

    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = self.reg * 0.5 * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss = data_loss + reg_loss
    # layer 2
    dx1, dW2, db2 = affine_backward(dscores, cache_scores)
    dW2 += self.reg * W2
    # layer 1
    dx, dW1, db1 = affine_relu_backward(dx1, cache_hidden_layer)
    dW1 += self.reg * W1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    grads.update({'W1': dW1,'b1': db1,'W2': dW2,'b2': db2})
    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}
    self.c = num_classes
    self.d = input_dim
    self.dims = [input_dim] + hidden_dims + [num_classes]

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    # print self.dims
    
    self.params.update({'W%d'%i : \
                  weight_scale * np.random.randn(self.dims[i-1], self.dims[i]) \
                  for i in xrange(1, self.num_layers+1)})
    self.params.update({'b%d'%i : np.zeros(self.dims[i])
        for i in xrange(1, self.num_layers+1)})
    # print {k: v.shape for k,v in self.params.items()}
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = {'bn_param%d'%i : {'mode': 'train', \
                          'running_mean': np.zeros(self.dims[i]), \
                          'running_var': np.zeros(self.dims[i])\
                          } for i in xrange(1, self.num_layers)}
      self.params.update({'beta%d'%i : \
                            np.ones(self.dims[i]) for i in xrange(1, self.num_layers)})
      self.params.update({'gamma%d'%i : \
                            np.zeros(self.dims[i]) for i in xrange(1, self.num_layers)})
    # print self.bn_params
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.

    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode
      
    if self.use_batchnorm:
      for bn_param in self.bn_params.keys():
        self.bn_params[bn_param]['mode'] = mode

    scores = None
    cache_hidden_layer = dict()
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    
    input_x = X
    reg_loss = 0.0
    scores = None

    for l in xrange(1, self.num_layers +1):
      # extract the params
      Wl, bl = self.params['W%d'%l], self.params['b%d'%l]
      affine_cache, bn_cache, dropout_cache, relu_cache = None, None, None, None
      
      # First Affine Transformation
      affine_out, affine_cache = affine_forward(input_x, Wl, bl)
      input_x = affine_out

      # Only affine trnsformation for Output layer 
      if(l == self.num_layers):
        cache_scores = affine_cache
        scores = affine_out
        continue

      # Second batch Normalization
      if(self.use_batchnorm):
        gamma_l = self.params['gamma%d'%l]
        beta_l = self.params['beta%d'%l]
        bn_param_l = self.bn_params['bn_param%d'%l]
        bn_out, bn_cache = batchnorm_forward(input_x, gamma_l, beta_l, bn_param_l)
        input_x = bn_out

      # Third, Dropout
      if(self.use_dropout):
        dropout_out, dropout_cache = dropout_forward(input_x, self.dropout_param)
        input_x = dropout_out

      # and finally Relu
      relu_out, relu_cache = relu_forward(input_x)
      input_x = relu_out

      reg_loss += self.reg * 0.5 * np.sum(Wl ** 2)
      cache_hidden_layer[l] = (affine_cache, bn_cache, dropout_cache, relu_cache)


        # zl, cache_hidden_layer[l] = affine_norm_relu_forward(\
        #   input_x, Wl, bl, gamma_l, beta_l, bn_param_l)
      # hidden layers forward propagation

      # else:
      #   zl, cache_hidden_layer[l] = affine_relu_forward(input_x, Wl, bl)



    # Output Layer
    # WL, bL = self.params['W%d'%self.num_layers], self.params['b%d'%self.num_layers]
    # scores, cache_scores = affine_forward(zl, WL, bL) # output layer

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    data_loss, dscores = softmax_loss(scores, y)
    # reg_loss += self.reg * 0.5 * np.sum(WL ** 2)
    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss = data_loss + reg_loss
    # print loss
    # Lth layer
    WL = self.params['W%d'%self.num_layers]
    dXL_minus_1, dWL, dbL = affine_backward(dscores, cache_scores)
    grads.update({ 'W%d'%self.num_layers : self.reg * WL + dWL })
    grads.update({ 'b%d'%self.num_layers : dbL })

    # for hidden (RELU) layers
    dout_z = dXL_minus_1
    for l in range(1, self.num_layers)[::-1]:
      # print l, cache_hidden_layer[l][1].shape
      (affine_cache, bn_cache, dropout_cache, relu_cache) = cache_hidden_layer[l]

      drelu = relu_backward(dout_z, relu_cache)
      dout_z = drelu

      if(self.use_dropout):
        ddropout = dropout_backward(dout_z, dropout_cache)
        dout_z = ddropout

      if(self.use_batchnorm):
        dbn, dgamma_l, dbeta_l = batchnorm_backward(dout_z, bn_cache)
        grads.update({ 'gamma%d'%l : dgamma_l, 'beta%d'%l : dbeta_l })
        dout_z = dbn
      
      dx, dWl, dbl = affine_backward(dout_z, affine_cache)

      # if(self.use_batchnorm):
      #   dx, dWl, dbl, dgamma_l, dbeta_l = affine_norm_relu_backward(\
      #     dXl_minus_1, cache_hidden_layer[l])
      #   grads.update({ 'gamma%d'%l : dgamma_l, 'beta%d'%l : dbeta_l })
      # else:
      #   dx, dWl, dbl = affine_relu_backward(dXl_minus_1, cache_hidden_layer[l])

      dout_z = dx
      grads.update({ 'W%d'%l : self.reg * self.params['W%d'%l] + dWl })
      grads.update({ 'b%d'%l : dbl })

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
