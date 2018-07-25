use libnum::{Float};

pub struct IterOptions<F: Float>{
    ///The tolerance that the solution is enforced to be within.
    ///This value for conjugate gradient type problems is suggested to be
    ///set to the following: tol * ||b_vec||_2
    pub sol_tol: F,
    ///The limit that we want to keep our solvers from going over.
    ///Most of the iterative solvers have an upper bound of solving the system
    ///within N steps where N is the number of rows and cols in the A matrix.
    ///In practice, the upper limit is hardly ever reached.
    pub iter_limit: u32,
    ///This parameter is used for methods such as GMRES methods where we restart our
    ///search for the correct solution.
    pub restart_iter: u32,

}

///Default options that can use for our iterative linear solvers for float32
impl Default for IterOptions<f32>{
    fn default() -> IterOptions<f32>{
        IterOptions{
            sol_tol: 1.0e-7,
            iter_limit: 10000,
            restart_iter: 25,
        }
    }
}

///Default options that can use for our iterative linear solvers for float64
impl Default for IterOptions<f64>{
    fn default() -> IterOptions<f64>{
        IterOptions{
            sol_tol: 1.0e-16,
            iter_limit: 10000,
            restart_iter: 25,
        }
    }
}