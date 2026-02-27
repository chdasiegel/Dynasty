"""
Compatibility shim for unpickling a scikit-learn model that
was trained in an environment where the compiled loss extension
module was available as a top-level `_loss` package.

The pickle stored references like `_loss.CyHalfSquaredError`.
In standard scikit-learn wheels, those classes actually live in
`sklearn._loss._loss`. We simply re-export them here so that
unpickling can succeed inside the Hugging Face Space.
"""

from sklearn._loss import _loss as _sk_loss


# Re-export only the class referenced in the pickle.
# In this environment, `_loss.HalfSquaredError` does not exist,
# but the pickle only needs `_loss.CyHalfSquaredError`.
CyHalfSquaredError = _sk_loss.CyHalfSquaredError

