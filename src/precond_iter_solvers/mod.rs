use ndarray::prelude::*;
use libnum::{Zero, One, Float};

use super::options::*;


pub fn pcg_solver<F>(a_mat: ArrayView2<F>, p_mat: ArrayView2<F>, x_vec: ArrayView1<F>, b_vec: ArrayViewMut1<F>, opt: &IterOptions<F>) -> F 
    where F: Float + Zero + One
{   
    //Here we need to assert that all of the dimensions are the correct length or else we need to kill the function
    let ndim_x = x_vec.len_of(Axis(0));
    let ndim_b = b_vec.len_of(Axis(0));

    let nrows_a = a_mat.len_of(Axis(0));
    let nrows_p = p_mat.len_of(Axis(0));

    let ncols_a = a_mat.len_of(Axis(1));
    let ncols_p = p_mat.len_of(Axis(1));

    assert!(ndim_b == ndim_x, 
    "The dimensions of the x vector and b vector are not equal to one another.
    The dimension of x is {} and dimension of b is {}",
    ndim_x, ndim_b);

    assert!(nrows_a == ncols_a, 
    "The A matrix must have the same number of rows and columns.
    The number of columns is {} and number of rows is {}",
    ncols_a, nrows_a);

    assert!(nrows_p == ncols_p,
    "The preconditioned matrix must have the same number of rows and columns.
    The number of columns is {} and number of rows is {}",
    ncols_p, nrows_p);

    assert!((ncols_p == ncols_a) & (nrows_p == nrows_a),
    "The preconditioned and A matrix must have the same number of rows and columns.
    The number of columns is {} and number of rows is {} in the preconditioned matrix.
    The number of columns is {} and number of rows is {} in the A matrix.",
    ncols_p, nrows_p, ncols_a, nrows_a);

    assert!(ndim_b == nrows_a, 
    "The number of columns of A must be equal to the number of rows of x vector.
    The number of rows of x is {} and number of cols of b is {}",
    ndim_x, ncols_a);

    let err: F = F::one();
    let _tol: F = opt.sol_tol;

    err

    // for iter in 0 .. opt.iter_limit{

    // }
}

