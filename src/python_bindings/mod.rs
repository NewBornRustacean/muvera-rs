#[cfg(feature = "python-bindings")]
use ndarray::ArrayView2;
#[cfg(feature = "python-bindings")]
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyModule;
#[cfg(feature = "python-bindings")]
use pyo3::wrap_pyfunction;

#[cfg(feature = "python-bindings")]
use crate::encoder::fde_encoder::{FDEEncoder, FDEEncoding};
#[cfg(feature = "python-bindings")]
use crate::types::Aggregation;

#[cfg(feature = "python-bindings")]
#[pyfunction]
fn encode_fde<'py>(
    py: Python<'py>,
    token_embeddings: PyReadonlyArray2<'py, f32>,
    agg: &str,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let tokens: ArrayView2<f32> = token_embeddings.as_array();
    let embedding_dim = tokens.ncols();
    
    // Create encoder with matching dimension and reasonable defaults
    let encoder = FDEEncoder::new(128, embedding_dim, 42);
    
    let mode = match agg {
        "mean" => Aggregation::Avg,
        "max" => Aggregation::Sum,
        _ => Aggregation::Avg,
    };
    let result = encoder.encode(tokens, mode);
    Ok(result.into_pyarray(py))
}

#[cfg(feature = "python-bindings")]
#[pymodule]
fn muvera(m: Bound<'_, PyModule>) -> PyResult<()> {
    m.clone().add_function(wrap_pyfunction!(encode_fde, m)?)?;
    Ok(())
}
