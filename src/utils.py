import torch
from gpytorch.utils.broadcasting import _mul_broadcast_shape


def _match_batch_dims(x1, *args):
    batch_shape = x1.shape[:-2]
    # Make sure the batch shapes agree for training/test data
    output = [x1]
    for x2 in args:
        if batch_shape != x1.shape[:-2]:
            batch_shape = _mul_broadcast_shape(batch_shape, x1.shape[:-2])
            x1 = x1.expand(*batch_shape, *x1.shape[-2:])
        if batch_shape != x2.shape[:-2]:
            batch_shape = _mul_broadcast_shape(batch_shape, x2.shape[:-2])
            x1 = x1.expand(*batch_shape, *x1.shape[-2:])
            x2 = x2.expand(*batch_shape, *x2.shape[-2:])
        output.append(x2)
    return output


def _match_dtype(x_in, *args):
    in_dtype = x_in.dtype
    output = []
    for arg in args:
        out = arg.to(in_dtype)
        output.append(out)
    return output


def unit_vector(v):
    """ Returns the unit vector of the vector.  """
    return v / torch.norm(v)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return torch.acos(torch.clamp(torch.dot(v1_u.float(), v2_u.float()), -1.0, 1.0))


def _outer_product(v1, v2):
    return torch.einsum('p,q->pq', v1, v2)