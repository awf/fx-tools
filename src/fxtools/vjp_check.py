import torch
from awfutils.pytree_utils import pt_rand_like, PyTree


def vjp_check(f, f_vjp, x, verbose=False):
    """
    Check that manually-defined VJP f_vjp matches torch's AD
    """
    ret = f(x)
    dret = torch.rand_like(ret)
    vjp_given = f_vjp(x, dret)

    vjp_torch = torch.autograd.functional.vjp(f, x, dret)

    PyTree.assert_close(vjp_given, vjp_torch, verbose=verbose)
    print("VJP OK:", f)


def vjp_check_fwdbwd(f, f_fwd, f_bwd, args):
    """
    Check that manually-defined VJP pair f_fwd, f_bwd matches torch's AD
       ret_given,aux = f_fwd(x)
       vjp_given = f_bwd(aux, dret)

    Should match
       ret_torch,vjp_torch = torch.autograd.functional.vjp(f, x, dret)

    """
    ret_given, aux = f_fwd(*args)
    dret = pt_rand_like(ret_given)
    vjp_given = f_bwd(aux, dret)

    assert isinstance(args, tuple)
    args = tuple(torch.tensor(a) if isinstance(a, float) else a for a in args)

    if isinstance(args, tuple) and len(args) == 1:
        x_for_torch = args[0]
    else:
        x_for_torch = args
    ret_torch, vjp_torch = torch.autograd.functional.vjp(f, x_for_torch, dret)

    PyTree.assert_close(ret_given, ret_torch)
    PyTree.assert_close(vjp_given, vjp_torch)


def vjp_check_fd(f, f_vjp, x, delta=1e-4, tol=1e-3):
    """
    Check that manually-defined VJP f_vjp obeys a finite-difference test
    We note that
      f(x + dx) ~ f(x) + J @ dx
      f(x + dx) - f(x) ~ J @ dx
    So, left-multiplying by dret.T:
      dret.T @ (f(x + dx) - f(x - dx))/2 ~ dret.T @ J @ dx
                                         = <f_vjp(x,dret), dx>
    So we assert closeness of:
      <dret, f(x + dx) - f(x - dx)> ~ 2*<f_vjp(x,dret), dx>
    """
    ret = f(x)
    dret = torch.randn_like(ret)
    dx = torch.randn_like(x)

    def dot(a, b):
        assert a.shape == b.shape
        return torch.dot(torch.flatten(a), torch.flatten(b))

    lhs = dot(dret, f(x + delta * dx) - f(x - delta * dx)) / (2 * delta)

    fval, fvjp = f_vjp(x, dret)
    rhs = dot(fvjp, dx)
    print(f"vjp_check_fd: FD={lhs:.4f}, Code={rhs:.4f}, diff={lhs-rhs:.4f}")

    torch.testing.assert_close(lhs, rhs, atol=tol, rtol=tol)
