# `fastfields.auto`

`fastfields.auto` adds dispatch, not new operations. Each function in the
tables below forwards to the function of the same name in the backend that
matches the type of its first array argument, and has that function's
signature, so an operation is documented on the page of its backend. The
[overview](../auto.md) explains how the dispatch works.

A backend is listed for an operation when it implements it. Calling an
operation on an array whose backend is not listed raises `NotImplementedError`
naming the backends that are.

## Distance transforms

| Operation | Documented on |
|---|---|
| `dt_euclidean` | [numpy][fastfields.numpy.dt_euclidean], [torch][fastfields.torch.dt_euclidean], [cupy][fastfields.cupy.dt_euclidean] |
| `dt_euclidean_` | [numpy][fastfields.numpy.dt_euclidean_], [torch][fastfields.torch.dt_euclidean_], [cupy][fastfields.cupy.dt_euclidean_] |
| `dt_l1` | [numpy][fastfields.numpy.dt_l1], [torch][fastfields.torch.dt_l1], [cupy][fastfields.cupy.dt_l1] |
| `dt_l1_` | [numpy][fastfields.numpy.dt_l1_], [torch][fastfields.torch.dt_l1_], [cupy][fastfields.cupy.dt_l1_] |
| `dt_spline_table` | [numpy][fastfields.numpy.dt_spline_table], [torch][fastfields.torch.dt_spline_table], [cupy][fastfields.cupy.dt_spline_table] |
| `dt_spline_brent` | [numpy][fastfields.numpy.dt_spline_brent], [torch][fastfields.torch.dt_spline_brent], [cupy][fastfields.cupy.dt_spline_brent] |
| `dt_spline_gaussnewton` | [numpy][fastfields.numpy.dt_spline_gaussnewton], [torch][fastfields.torch.dt_spline_gaussnewton], [cupy][fastfields.cupy.dt_spline_gaussnewton] |
| `dt_mesh` | [numpy][fastfields.numpy.dt_mesh], [torch][fastfields.torch.dt_mesh], [cupy][fastfields.cupy.dt_mesh] |

## Positive-definite linear algebra

| Operation | Documented on |
|---|---|
| `sym_matvec` | [numpy][fastfields.numpy.sym_matvec], [torch][fastfields.torch.sym_matvec], [cupy][fastfields.cupy.sym_matvec] |
| `sym_matvec_backward` | [numpy][fastfields.numpy.sym_matvec_backward], [cupy][fastfields.cupy.sym_matvec_backward] |
| `sym_addmatvec_` | [numpy][fastfields.numpy.sym_addmatvec_], [torch][fastfields.torch.sym_addmatvec_], [cupy][fastfields.cupy.sym_addmatvec_] |
| `sym_submatvec_` | [numpy][fastfields.numpy.sym_submatvec_], [torch][fastfields.torch.sym_submatvec_], [cupy][fastfields.cupy.sym_submatvec_] |
| `sym_solve` | [numpy][fastfields.numpy.sym_solve], [torch][fastfields.torch.sym_solve], [cupy][fastfields.cupy.sym_solve] |
| `sym_solve_` | [numpy][fastfields.numpy.sym_solve_], [torch][fastfields.torch.sym_solve_], [cupy][fastfields.cupy.sym_solve_] |
| `sym_invert` | [numpy][fastfields.numpy.sym_invert], [torch][fastfields.torch.sym_invert], [cupy][fastfields.cupy.sym_invert] |
| `sym_invert_` | [numpy][fastfields.numpy.sym_invert_], [torch][fastfields.torch.sym_invert_], [cupy][fastfields.cupy.sym_invert_] |

## Resampling

| Operation | Documented on |
|---|---|
| `resample` | [numpy][fastfields.numpy.resample], [torch][fastfields.torch.resample], [cupy][fastfields.cupy.resample] |
| `restriction` | [numpy][fastfields.numpy.restriction], [torch][fastfields.torch.restriction], [cupy][fastfields.cupy.restriction] |
| `spline_coeff` | [numpy][fastfields.numpy.spline_coeff], [torch][fastfields.torch.spline_coeff], [cupy][fastfields.cupy.spline_coeff] |
| `spline_coeff_` | [numpy][fastfields.numpy.spline_coeff_], [torch][fastfields.torch.spline_coeff_], [cupy][fastfields.cupy.spline_coeff_] |

## Spline sampling (pushpull)

| Operation | Documented on |
|---|---|
| `pull` | [numpy][fastfields.numpy.pull], [torch][fastfields.torch.pull], [cupy][fastfields.cupy.pull] |
| `push` | [numpy][fastfields.numpy.push], [torch][fastfields.torch.push], [cupy][fastfields.cupy.push] |
| `count` | [numpy][fastfields.numpy.count], [torch][fastfields.torch.count], [cupy][fastfields.cupy.count] |
| `grad` | [numpy][fastfields.numpy.grad], [torch][fastfields.torch.grad], [cupy][fastfields.cupy.grad] |

## Field regularisers

| Operation | Documented on |
|---|---|
| `field_matvec` | [numpy][fastfields.numpy.field_matvec], [torch][fastfields.torch.field_matvec], [cupy][fastfields.cupy.field_matvec] |
| `field_addmatvec` | [numpy][fastfields.numpy.field_addmatvec], [torch][fastfields.torch.field_addmatvec], [cupy][fastfields.cupy.field_addmatvec] |
| `field_addmatvec_` | [numpy][fastfields.numpy.field_addmatvec_], [torch][fastfields.torch.field_addmatvec_], [cupy][fastfields.cupy.field_addmatvec_] |
| `field_submatvec` | [numpy][fastfields.numpy.field_submatvec], [torch][fastfields.torch.field_submatvec], [cupy][fastfields.cupy.field_submatvec] |
| `field_submatvec_` | [numpy][fastfields.numpy.field_submatvec_], [torch][fastfields.torch.field_submatvec_], [cupy][fastfields.cupy.field_submatvec_] |
| `field_adddiag` | [numpy][fastfields.numpy.field_adddiag], [torch][fastfields.torch.field_adddiag], [cupy][fastfields.cupy.field_adddiag] |
| `field_adddiag_` | [numpy][fastfields.numpy.field_adddiag_], [torch][fastfields.torch.field_adddiag_], [cupy][fastfields.cupy.field_adddiag_] |
| `field_subdiag` | [numpy][fastfields.numpy.field_subdiag], [torch][fastfields.torch.field_subdiag], [cupy][fastfields.cupy.field_subdiag] |
| `field_subdiag_` | [numpy][fastfields.numpy.field_subdiag_], [torch][fastfields.torch.field_subdiag_], [cupy][fastfields.cupy.field_subdiag_] |
| `field_addkernel` | [numpy][fastfields.numpy.field_addkernel], [torch][fastfields.torch.field_addkernel], [cupy][fastfields.cupy.field_addkernel] |
| `field_addkernel_` | [numpy][fastfields.numpy.field_addkernel_], [torch][fastfields.torch.field_addkernel_], [cupy][fastfields.cupy.field_addkernel_] |
| `field_subkernel` | [numpy][fastfields.numpy.field_subkernel], [torch][fastfields.torch.field_subkernel], [cupy][fastfields.cupy.field_subkernel] |
| `field_subkernel_` | [numpy][fastfields.numpy.field_subkernel_], [torch][fastfields.torch.field_subkernel_], [cupy][fastfields.cupy.field_subkernel_] |
| `field_relax` | [numpy][fastfields.numpy.field_relax], [torch][fastfields.torch.field_relax], [cupy][fastfields.cupy.field_relax] |
| `field_cg` | [numpy][fastfields.numpy.field_cg] |
| `field_matvec_rls` | [numpy][fastfields.numpy.field_matvec_rls], [torch][fastfields.torch.field_matvec_rls], [cupy][fastfields.cupy.field_matvec_rls] |
| `field_diag_rls` | [numpy][fastfields.numpy.field_diag_rls], [torch][fastfields.torch.field_diag_rls], [cupy][fastfields.cupy.field_diag_rls] |
| `field_relax_rls` | [numpy][fastfields.numpy.field_relax_rls], [torch][fastfields.torch.field_relax_rls], [cupy][fastfields.cupy.field_relax_rls] |
| `field_precond` | [numpy][fastfields.numpy.field_precond], [torch][fastfields.torch.field_precond], [cupy][fastfields.cupy.field_precond] |
| `field_precond_` | [numpy][fastfields.numpy.field_precond_], [cupy][fastfields.cupy.field_precond_] |
| `field_forward` | [numpy][fastfields.numpy.field_forward], [torch][fastfields.torch.field_forward], [cupy][fastfields.cupy.field_forward] |

## Flow regularisers

| Operation | Documented on |
|---|---|
| `flow_matvec` | [numpy][fastfields.numpy.flow_matvec], [torch][fastfields.torch.flow_matvec], [cupy][fastfields.cupy.flow_matvec] |
| `flow_addmatvec` | [numpy][fastfields.numpy.flow_addmatvec], [torch][fastfields.torch.flow_addmatvec], [cupy][fastfields.cupy.flow_addmatvec] |
| `flow_addmatvec_` | [numpy][fastfields.numpy.flow_addmatvec_], [torch][fastfields.torch.flow_addmatvec_], [cupy][fastfields.cupy.flow_addmatvec_] |
| `flow_submatvec` | [numpy][fastfields.numpy.flow_submatvec], [torch][fastfields.torch.flow_submatvec], [cupy][fastfields.cupy.flow_submatvec] |
| `flow_submatvec_` | [numpy][fastfields.numpy.flow_submatvec_], [torch][fastfields.torch.flow_submatvec_], [cupy][fastfields.cupy.flow_submatvec_] |
| `flow_adddiag` | [numpy][fastfields.numpy.flow_adddiag], [torch][fastfields.torch.flow_adddiag], [cupy][fastfields.cupy.flow_adddiag] |
| `flow_adddiag_` | [numpy][fastfields.numpy.flow_adddiag_], [torch][fastfields.torch.flow_adddiag_], [cupy][fastfields.cupy.flow_adddiag_] |
| `flow_subdiag` | [numpy][fastfields.numpy.flow_subdiag], [torch][fastfields.torch.flow_subdiag], [cupy][fastfields.cupy.flow_subdiag] |
| `flow_subdiag_` | [numpy][fastfields.numpy.flow_subdiag_], [torch][fastfields.torch.flow_subdiag_], [cupy][fastfields.cupy.flow_subdiag_] |
| `flow_addkernel` | [numpy][fastfields.numpy.flow_addkernel], [torch][fastfields.torch.flow_addkernel], [cupy][fastfields.cupy.flow_addkernel] |
| `flow_addkernel_` | [numpy][fastfields.numpy.flow_addkernel_], [torch][fastfields.torch.flow_addkernel_], [cupy][fastfields.cupy.flow_addkernel_] |
| `flow_subkernel` | [numpy][fastfields.numpy.flow_subkernel], [torch][fastfields.torch.flow_subkernel], [cupy][fastfields.cupy.flow_subkernel] |
| `flow_subkernel_` | [numpy][fastfields.numpy.flow_subkernel_], [torch][fastfields.torch.flow_subkernel_], [cupy][fastfields.cupy.flow_subkernel_] |
| `flow_relax` | [numpy][fastfields.numpy.flow_relax], [torch][fastfields.torch.flow_relax], [cupy][fastfields.cupy.flow_relax] |
| `flow_matvec_rls` | [numpy][fastfields.numpy.flow_matvec_rls], [torch][fastfields.torch.flow_matvec_rls], [cupy][fastfields.cupy.flow_matvec_rls] |
| `flow_diag_rls` | [numpy][fastfields.numpy.flow_diag_rls], [torch][fastfields.torch.flow_diag_rls], [cupy][fastfields.cupy.flow_diag_rls] |
| `flow_relax_rls` | [numpy][fastfields.numpy.flow_relax_rls], [torch][fastfields.torch.flow_relax_rls], [cupy][fastfields.cupy.flow_relax_rls] |
| `flow_precond` | [numpy][fastfields.numpy.flow_precond], [torch][fastfields.torch.flow_precond], [cupy][fastfields.cupy.flow_precond] |
| `flow_precond_` | [numpy][fastfields.numpy.flow_precond_], [cupy][fastfields.cupy.flow_precond_] |
| `flow_forward` | [numpy][fastfields.numpy.flow_forward], [torch][fastfields.torch.flow_forward], [cupy][fastfields.cupy.flow_forward] |

## The module

The module itself, with `dlpack_backend` and the re-exported enums:

::: fastfields.auto
