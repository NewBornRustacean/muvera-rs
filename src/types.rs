use ndarray::ScalarOperand;
use num_traits::Float;

/// Aggregation type used during encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aggregation {
    Sum,
    Avg,
}

/// Trait bound for numeric types used in FDE
///
/// Supports f32, f64 (possibly f16 in the future)
pub trait FDEFloat: Float + ScalarOperand + 'static {}

impl FDEFloat for f32 {}
impl FDEFloat for f64 {}
