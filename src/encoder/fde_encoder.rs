use ndarray::{parallel::prelude::*, Array1, Array2, ArrayView2, ArrayView3, Axis};
use rand::{self, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use crate::types::{Aggregation, FDEFloat};

/// Trait for fixed-dimensional encoding from token embeddings.
///
/// This trait defines methods for encoding token embeddings into
/// fixed-dimensional vectors using the FDE (Fixed Dimensional Encoding) algorithm.
///
/// # Type Parameters
/// - `T`: The numeric float type, e.g., `f32` or `f64`.
pub trait FDEEncoding<T: FDEFloat + Send + Sync> {
    /// Encode a single multi-vector (2D tokens) into a fixed-dimensional vector.
    ///
    /// # Arguments
    /// - `tokens`: 2D array of shape `(num_tokens, embedding_dim)`.
    /// - `mode`: Aggregation mode, either sum or average across buckets.
    ///
    /// # Returns
    /// - 1D array of length `buckets * embedding_dim` representing the encoded vector.
    fn encode(&self, tokens: ArrayView2<T>, mode: Aggregation) -> Array1<T>;

    /// Encode a batch of multi-vectors (3D tokens) into fixed-dimensional vectors.
    ///
    /// # Arguments
    /// - `batch_tokens`: 3D array of shape `(batch_size, num_tokens, embedding_dim)`.
    /// - `mode`: Aggregation mode, either sum or average across buckets.
    ///
    /// # Returns
    /// - 2D array of shape `(batch_size, buckets * embedding_dim)` where each row
    ///   is the encoded vector for the corresponding batch element.
    fn batch_encode(&self, batch_tokens: ArrayView3<T>, mode: Aggregation) -> Array2<T>;

    /// Encode a query token embedding using sum aggregation.
    ///
    /// # Arguments
    /// - `tokens`: 2D array of shape `(num_tokens, embedding_dim)`.
    ///
    /// # Returns
    /// - Fixed-dimensional encoded query vector.
    fn encode_query(&self, tokens: ArrayView2<T>) -> Array1<T> {
        self.encode(tokens, Aggregation::Sum)
    }

    /// Encode a document token embedding using average aggregation.
    ///
    /// # Arguments
    /// - `tokens`: 2D array of shape `(num_tokens, embedding_dim)`.
    ///
    /// # Returns
    /// - Fixed-dimensional encoded document vector.
    fn encode_doc(&self, tokens: ArrayView2<T>) -> Array1<T> {
        self.encode(tokens, Aggregation::Avg)
    }

    /// Batch encode queries using sum aggregation.
    ///
    /// # Arguments
    /// - `batch_tokens`: 3D array of shape `(batch_size, num_tokens, embedding_dim)`.
    ///
    /// # Returns
    /// - 2D array of encoded query vectors.
    fn encode_query_batch(&self, batch_tokens: ArrayView3<T>) -> Array2<T> {
        self.batch_encode(batch_tokens, Aggregation::Sum)
    }

    /// Batch encode documents using average aggregation.
    ///
    /// # Arguments
    /// - `batch_tokens`: 3D array of shape `(batch_size, num_tokens, embedding_dim)`.
    ///
    /// # Returns
    /// - 2D array of encoded document vectors.
    fn encode_doc_batch(&self, batch_tokens: ArrayView3<T>) -> Array2<T> {
        self.batch_encode(batch_tokens, Aggregation::Avg)
    }
}
/// Fixed Dimensional Encoder (FDE) implementation.
///
/// Encodes variable-length token embeddings into fixed-length vectors using
/// randomized hyperplanes and aggregation.
///
/// # Fields
/// - `buckets`: Number of hyperplanes / buckets to hash tokens into.
/// - `dim`: Embedding dimensionality of input tokens.
/// - `hyperplanes`: Random hyperplanes matrix used for projection and hashing.
pub struct FDEEncoder<T: FDEFloat> {
    pub buckets: usize,
    pub dim: usize,
    pub hyperplanes: Array2<T>,
}

impl<T: FDEFloat> FDEEncoder<T> {
    /// Creates a new FDE encoder with the specified number of buckets and embedding dimension.
    ///
    /// # Arguments
    /// - `buckets`: Number of hash buckets (hyperplanes).
    /// - `dim`: Dimensionality of token embeddings.
    /// - `seed`: RNG seed for reproducible hyperplane initialization.
    ///
    /// # Returns
    /// A new `FDEEncoder` instance.
    pub fn new(buckets: usize, dim: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Create hyperplanes directly with the right shape
        let hyperplanes = Array2::from_shape_fn((dim, buckets), |_| {
            let sample: f64 = StandardNormal.sample(&mut rng);
            T::from(sample).unwrap() // safe unwrap since f32/f64 supported
        });

        Self {
            buckets,
            dim,
            hyperplanes,
        }
    }
}

impl Default for FDEEncoder<f32> {
    fn default() -> Self {
        Self::new(128, 768, 42)
    }
}

impl<T: FDEFloat + Send + Sync + num_traits::FromPrimitive> FDEEncoding<T> for FDEEncoder<T> {
    /// Encode a single multi-vector of tokens into a fixed-dimensional vector.
    ///
    /// Projects tokens onto hyperplanes, hashes them into buckets,
    /// aggregates by sum or average per bucket, and concatenates the results.
    ///
    /// # Arguments
    /// - `multi_vector_tokens`: 2D array of token embeddings `(num_tokens, dim)`.
    /// - `mode`: Aggregation mode (`Sum` or `Avg`).
    ///
    /// # Returns
    /// A 1D array of length `buckets * dim` representing the encoded vector.

    fn encode(&self, multi_vector_tokens: ArrayView2<T>, mode: Aggregation) -> Array1<T> {
        let buckets = self.buckets;
        assert_eq!(multi_vector_tokens.ncols(), self.dim);

        // 1. Projection (num_tokens, buckets)
        let projections = multi_vector_tokens.dot(&self.hyperplanes);

        // 2. ReLU Activation
        let activated = projections.mapv(|x| if x > T::zero() { x } else { T::zero() });

        // 3. Aggregation
        match mode {
            // Query: Max-pooling
            Aggregation::Sum => {
                activated.fold_axis(Axis(0), T::zero(), |&acc, &x| if x > acc { x } else { acc })
            }
            // Document: Mean-pooling
            Aggregation::Avg => {
                activated.mean_axis(Axis(0)).unwrap_or_else(|| Array1::zeros(buckets))
            }
        }
    }

    /// Encode a batch of multi-vectors using parallel processing.
    ///
    /// Divides the batch across threads for concurrent encoding.
    ///
    /// # Arguments
    /// - `batch_tokens`: 3D array `(batch_size, num_tokens, dim)`.
    /// - `mode`: Aggregation mode (`Sum` or `Avg`).
    ///
    /// # Returns
    /// 2D array of encoded vectors `(batch_size, buckets * dim)`.

    fn batch_encode(&self, batch_tokens: ArrayView3<T>, mode: Aggregation) -> Array2<T>
    where
        T: FDEFloat + Sync + Send,
        Self: Sync,
    {
        let (batch_size, _n, _) = batch_tokens.dim();
        let buckets = self.buckets;

        // Pre-allocate output array
        let mut result = Array2::<T>::zeros((batch_size, buckets));

        // Process in parallel chunks for better cache locality
        let chunk_size =
            (batch_size + rayon::current_num_threads() - 1) / rayon::current_num_threads();

        result
            .axis_chunks_iter_mut(Axis(0), chunk_size)
            .into_par_iter()
            .zip(batch_tokens.axis_chunks_iter(Axis(0), chunk_size))
            .for_each(|(mut result_chunk, tokens_chunk)| {
                for (i, tokens_2d) in tokens_chunk.axis_iter(Axis(0)).enumerate() {
                    let encoded = self.encode(tokens_2d, mode);
                    result_chunk.row_mut(i).assign(&encoded);
                }
            });

        result
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    const DIM: usize = 4;
    const BUCKETS: usize = 3;

    fn create_encoder() -> FDEEncoder<f32> {
        FDEEncoder::new(BUCKETS, DIM, 42)
    }

    #[test]
    fn test_output_dimension_reduction() {
        let enc = create_encoder();
        let tokens = array![[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]];
        let vec = enc.encode(tokens.view(), Aggregation::Sum);

        // Output length must be equal to buckets, not buckets * dim
        assert_eq!(vec.len(), BUCKETS);
    }

    #[test]
    fn test_relu_behavior_correctness() {
        let enc = create_encoder();
        let tokens = array![[1.0, 1.0, 1.0, 1.0]];
        let projections = tokens.dot(&enc.hyperplanes);
        let encoded = enc.encode(tokens.view(), Aggregation::Sum);

        for i in 0..BUCKETS {
            // if the projection is negative or zero, the encoded value should be zero due to ReLU
            if projections[[0, i]] <= 0.0 {
                assert_eq!(encoded[i], 0.0);
            } else {
                assert_eq!(encoded[i], projections[[0, i]]);
            }
        }
    }

    #[test]
    fn test_asymmetric_property() {
        let enc = create_encoder();
        let tokens = array![[1.0, 1.0, 1.0, 1.0], [0.1, 0.1, 0.1, 0.1]];

        let query_vec = enc.encode_query(tokens.view()); // Max
        let doc_vec = enc.encode_doc(tokens.view()); // Mean

        // Max aggregation should produce values >= Mean aggregation
        for i in 0..BUCKETS {
            assert!(query_vec[i] >= doc_vec[i]);
        }
    }

    #[test]
    fn test_batch_encode_concurrency() {
        let enc = create_encoder();
        let batch_tokens = array![
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            [[0.9, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]]
        ];
        let result = enc.batch_encode(batch_tokens.view(), Aggregation::Avg);
        assert_eq!(result.shape(), &[2, BUCKETS]);
    }

    #[test]
    fn test_numerical_stability() {
        let enc = create_encoder();
        let tokens = array![[f32::MAX, f32::MIN, 0.0, 1.0]];
        let vec = enc.encode(tokens.view(), Aggregation::Avg);
        assert!(vec.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_reproducibility() {
        let enc1: FDEEncoder<f64> = FDEEncoder::new(BUCKETS, DIM, 42);
        let enc2 = FDEEncoder::new(BUCKETS, DIM, 42);
        let tokens = array![[0.1, 0.2, 0.3, 0.4]];
        assert_eq!(
            enc1.encode_query(tokens.view()),
            enc2.encode_query(tokens.view())
        );
    }

    #[test]
    fn test_padding_impact() {
        let enc = create_encoder();
        let tokens_real = array![[0.5, 0.5, 0.5, 0.5]];
        let tokens_with_padding = array![[0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]];

        // Sum/Max should ignore zero padding, Mean will be affected.
        assert_eq!(
            enc.encode_query(tokens_real.view()),
            enc.encode_query(tokens_with_padding.view())
        );
    }
}
