#[cfg(feature = "python-bindings")]
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

#[cfg(feature = "python-bindings")]
use crate::encoder::fde_encoder::{FDEEncoder, FDEEncoding};
#[cfg(feature = "python-bindings")]
use crate::types::Aggregation;

#[cfg(feature = "python-bindings")]
#[pyclass]
struct PyFDEEncoder {
    inner: FDEEncoder<f32>,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyFDEEncoder {
    #[new]
    #[pyo3(signature = (buckets, dim, seed=42))]
    fn new(buckets: usize, dim: usize, seed: u64) -> Self {
        Self {
            inner: FDEEncoder::new(buckets, dim, seed),
        }
    }

    /// Encodes a single multi-vector.
    /// agg: "mean" for documents, "max" for queries.
    fn encode<'py>(
        &self,
        py: Python<'py>,
        token_embeddings: PyReadonlyArray2<'py, f32>,
        agg: &str,
    ) -> Bound<'py, PyArray1<f32>> {
        let tokens = token_embeddings.as_array();
        let mode = match agg {
            "max" | "sum" => Aggregation::Sum,
            _ => Aggregation::Avg,
        };
        let result = self.inner.encode(tokens, mode);
        result.into_pyarray(py)
    }
}

#[cfg(feature = "python-bindings")]
#[pymodule]
fn muvera(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFDEEncoder>()?;
    Ok(())
}
