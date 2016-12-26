import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    f_i = X[i, :].dot(W)
    f_i -= np.max(f_i)
    exp_score_i = np.exp(f_i)
    p = exp_score_i / np.sum(exp_score_i)
    log_probs = -np.log(p[y[i]])
    # print exp_score_i,np.sum(exp_score_i), p, p[y[i]], log_probs
    loss += log_probs # correct log prob

    dscores = p
    dscores[y[i]] -= 1
    dW += np.outer(X[i, :], dscores)

  loss = loss / num_train + 0.5 * reg * np.sum(W ** 2)
  dW = dW / num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X, W)
  scores -= np.max(scores, axis = 1, keepdims=True)
  exp_scores = np.exp(scores)
  p = exp_scores / np.sum(exp_scores, axis = 1, keepdims=True)
  log_probs = -np.log(p[np.arange(num_train), y])
  loss = np.sum(log_probs)/num_train + 0.5 * reg * np.sum(W ** 2)

  dscores = p
  dscores[np.arange(num_train), y] -= 1
  dscores /= num_train
  dW = X.T.dot(dscores) + reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

