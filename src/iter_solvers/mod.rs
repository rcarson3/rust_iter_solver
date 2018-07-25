use ndarray::prelude::*;
use ndarray::Zip;
use libnum::{Zero, One, Float};//NumCast

use super::options::IterOptions;

///A conjugate gradient solver used to iteratively solve a symmetric A x = b type problem.
///Input: a_mat - a 2D matrix with nxn dimensions. This matrix must also be symmetric in order to use the conjugate gradient method.
///x_vec - a 1D vector that has n dimensions. The initial values inputted into this vector are the initial guesses to the solution.
///b_vec - a 1D vector that has n dimensions. The vector RHS of the Ax=b problem.
///opt - the iterative option structure. It tells us a number of things that we need to worry about for our iterative problems.
///Output - err - a Float that tells us what the error of our solution was found to be. You should check to make sure this meets
///         your set error tolerances.
pub fn cg_solver<F: 'static>(a_mat: ArrayView2<F>,mut x_vec: ArrayViewMut1<F>, b_vec: ArrayView1<F>, opt: &IterOptions<F>) -> F 
    where F: Float + Zero + One
{
    //Here we need to assert that all of the dimensions are the correct length or else we need to kill the function
    let ndim_x = x_vec.len_of(Axis(0));

    let ndim_b = b_vec.len_of(Axis(0));

    let nrows_a = a_mat.len_of(Axis(0));
    let ncols_a = a_mat.len_of(Axis(1));

    assert!(ndim_b == ndim_x, 
    "The dimensions of the x vector and b vector are not equal to one another.
    The dimension of x is {} and dimension of b is {}",
    ndim_x, ndim_b);

    assert!(nrows_a == ncols_a, 
    "The A matrix must have the same number of rows and columns.
    The number of columns is {} and number of rows is {}",
    ncols_a, nrows_a);

    assert!(ndim_b == nrows_a, 
    "The number of columns of A must be equal to the number of rows of x vector.
    The number of rows of x is {} and number of cols of b is {}",
    ndim_x, ncols_a);

    let tol: F = opt.sol_tol;

    let mut ri = Array1::<F>::zeros(ndim_x);
    let mut pki = Array1::<F>::zeros(ndim_x);
    let mut a_pk = Array1::<F>::zeros(ndim_x);

    let mut tau: F = F::zero();
    let mut pkt_a_pk: F = F::zero();
    let mut mu: F = F::zero();

    ri.assign(&(&a_mat.dot(&x_vec) - &b_vec));

    pki.assign(&ri);

    let mut rtr: F = ri.dot(&ri);
    let mut err: F = rtr.sqrt();
    let mut rt1r1: F = rtr.clone();

    for _istep in 0..opt.iter_limit{

        a_pk.assign(&a_mat.dot(&pki));

        pkt_a_pk = pki.dot(&a_pk);
        
        mu = rtr/pkt_a_pk;

        x_vec.scaled_add(mu, &pki);

        ri.scaled_add(-mu, &a_pk);

        rtr = ri.dot(&ri);

        err = rtr.sqrt();

        if err.abs() < tol{
            break;
        }

        tau = rtr/rt1r1;

        rt1r1 = rtr.clone();

        Zip::from(&mut pki).and(&ri).apply(|pki, &ri|{
            *pki = ri + tau * *pki;  
        });

    }

    err
}

//We can set the below in the future to allowing rectangular A matrices. Therefore, it could be applied to solve full-rank least squares
//type problems. 

///A conjugate gradient normal equation residual (CGNR) solver used to iteratively solve a nonsymmetric A x = b type problem.
///This method creates a symmetric problem to solve by doing Ax = b == {A^T A x = A^T b}
///Input: a_mat - a 2D matrix with nxn dimensions. This matrix must also be nonsymmetric in order to use the CGNR method.
///x_vec - a 1D vector that has n dimensions. The initial values inputted into this vector are the initial guesses to the solution.
///b_vec - a 1D vector that has n dimensions. The vector RHS of the Ax=b problem.
///opt - the iterative option structure. It tells us a number of things that we need to worry about for our iterative problems.
///Output - err - a Float that tells us what the error of our solution was found to be. You should check to make sure this meets
///         your set error tolerances.
pub fn cgnr_solver<F: 'static>(a_mat: ArrayView2<F>,mut x_vec: ArrayViewMut1<F>, b_vec: ArrayView1<F>, opt: &IterOptions<F>) -> F 
    where F: Float + Zero + One
{
    //Here we need to assert that all of the dimensions are the correct length or else we need to kill the function
    let ndim_x = x_vec.len_of(Axis(0));

    let ndim_b = b_vec.len_of(Axis(0));

    let nrows_a = a_mat.len_of(Axis(0));
    let ncols_a = a_mat.len_of(Axis(1));

    assert!(ndim_b == ndim_x, 
    "The dimensions of the x vector and b vector are not equal to one another.
    The dimension of x is {} and dimension of b is {}",
    ndim_x, ndim_b);

    assert!(nrows_a == ncols_a, 
    "The A matrix must have the same number of rows and columns.
    The number of columns is {} and number of rows is {}",
    ncols_a, nrows_a);

    assert!(ndim_b == nrows_a, 
    "The number of columns of A must be equal to the number of rows of x vector.
    The number of rows of x is {} and number of cols of b is {}",
    ndim_x, ncols_a);

    let tol: F = opt.sol_tol;

    let mut ri = Array1::<F>::zeros(ndim_x);
    let mut r_t = Array2::<F>::zeros((1, ndim_x));
    let mut pki = Array1::<F>::zeros(ndim_x);
    let mut zi = Array1::<F>::zeros(ndim_x);
    let mut a_pk = Array1::<F>::zeros(ndim_x);

    let mut tau: F = F::zero();
    let mut a_pk_t_a_pk: F = F::zero();
    let mut mu: F = F::zero();

    ri.assign(&(&a_mat.dot(&x_vec) - &b_vec));

    r_t.assign(&ri);

    zi.assign(&r_t.dot(&a_mat));

    pki.assign(&zi);

    let mut ztz: F = zi.dot(&zi);
    let mut rtr: F = ri.dot(&ri);
    let mut err: F = rtr.sqrt();
    let mut zt1z1: F = ztz.clone();

    for _istep in 0..opt.iter_limit{

        a_pk.assign(&a_mat.dot(&pki));

        a_pk_t_a_pk = a_pk.dot(&a_pk);
        
        mu = ztz/a_pk_t_a_pk;

        x_vec.scaled_add(mu, &pki);

        ri.scaled_add(-mu, &a_pk);

        rtr = ri.dot(&ri);

        err = rtr.sqrt();

        if err.abs() < tol{
            break;
        }

        r_t.assign(&ri);

        zi.assign(&r_t.dot(&a_mat));

        ztz = zi.dot(&zi);

        tau = ztz/zt1z1;

        zt1z1 = ztz.clone();

        Zip::from(&mut pki).and(&zi).apply(|pki, &zi|{
            *pki = zi + tau * *pki;  
        });

    }

    err
}

//We can set the below in the future to allowing rectangular A matrices. When applied to this solver the underlying system must be
//consistent

///A conjugate gradient normal equation error (CGNE) solver used to iteratively solve a nonsymmetric A x = b type problem.
///This method creates a symmetric problem to solve by doing Ax = b == {AA^T y = b, x = A^T y}
///Input: a_mat - a 2D matrix with nxn dimensions. This matrix must also be nonsymmetric in order to use the CGNR method.
///x_vec - a 1D vector that has n dimensions. The initial values inputted into this vector are the initial guesses to the solution.
///b_vec - a 1D vector that has n dimensions. The vector RHS of the Ax=b problem.
///opt - the iterative option structure. It tells us a number of things that we need to worry about for our iterative problems.
///Output - err - a Float that tells us what the error of our solution was found to be. You should check to make sure this meets
///         your set error tolerances.
pub fn cgne_solver<F: 'static>(a_mat: ArrayView2<F>,mut x_vec: ArrayViewMut1<F>, b_vec: ArrayView1<F>, opt: &IterOptions<F>) -> F 
    where F: Float + Zero + One
{
    //Here we need to assert that all of the dimensions are the correct length or else we need to kill the function
    let ndim_x = x_vec.len_of(Axis(0));

    let ndim_b = b_vec.len_of(Axis(0));

    let nrows_a = a_mat.len_of(Axis(0));
    let ncols_a = a_mat.len_of(Axis(1));

    assert!(ndim_b == ndim_x, 
    "The dimensions of the x vector and b vector are not equal to one another.
    The dimension of x is {} and dimension of b is {}",
    ndim_x, ndim_b);

    assert!(nrows_a == ncols_a, 
    "The A matrix must have the same number of rows and columns.
    The number of columns is {} and number of rows is {}",
    ncols_a, nrows_a);

    assert!(ndim_b == nrows_a, 
    "The number of columns of A must be equal to the number of rows of x vector.
    The number of rows of x is {} and number of cols of b is {}",
    ndim_x, ncols_a);

    let tol: F = opt.sol_tol;

    let mut ri = Array1::<F>::zeros(ndim_x);
    let mut r_t = Array2::<F>::zeros((1, ndim_x));
    let mut pki = Array1::<F>::zeros(ndim_x);
    let mut zi = Array1::<F>::zeros(ndim_x);
    let mut a_pk = Array1::<F>::zeros(ndim_x);

    let mut tau: F = F::zero();
    let mut pki_t_pki: F = F::zero();
    let mut mu: F = F::zero();

    ri.assign(&(&a_mat.dot(&x_vec) - &b_vec));

    r_t.assign(&ri);

    zi.assign(&r_t.dot(&a_mat));

    pki.assign(&zi);

    let mut rtr: F = ri.dot(&ri);
    let mut err: F = rtr.sqrt();
    let mut rt1r1: F = rtr.clone();

    for _istep in 0..opt.iter_limit{

        a_pk.assign(&a_mat.dot(&pki));

        pki_t_pki = pki.dot(&pki);
        
        mu = rtr/pki_t_pki;

        x_vec.scaled_add(mu, &pki);

        ri.scaled_add(-mu, &a_pk);

        rtr = ri.dot(&ri);

        err = rtr.sqrt();

        if err.abs() < tol{
            break;
        }

        r_t.assign(&ri);

        zi.assign(&r_t.dot(&a_mat));

        tau = rtr/rt1r1;

        rt1r1 = rtr.clone();

        Zip::from(&mut pki).and(&zi).apply(|pki, &zi|{
            *pki = zi + tau * *pki;  
        });

    }

    err
}