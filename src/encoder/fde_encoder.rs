use crate::types::{Aggregation, FDEFloat};
use ndarray::parallel::prelude::*;
use ndarray::{s, Array1, Array2, Array3, ArrayView2, ArrayView3, Axis};
use rand;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

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

impl<T: FDEFloat + Send + Sync> FDEEncoding<T> for FDEEncoder<T> {
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
        let embedding_dim = multi_vector_tokens.ncols();
        let buckets = self.buckets;

        assert_eq!(embedding_dim, self.dim);

        // Project tokens onto hyperplanes (vectorized)
        // projections shape: (num_tokens, buckets)
        let projections: Array2<T> = multi_vector_tokens.dot(&self.hyperplanes);

        // Convert projections > 0 to binary mask (usize)
        let bin_mask_usize = projections.mapv(|x| if x > T::zero() { 1usize } else { 0usize });

        // Weighted sum of binary mask to get bucket indices
        let weights = Array1::from_iter((1..=buckets).map(|i| i));
        let hashes: Array1<usize> = bin_mask_usize.dot(&weights);
        let bucket_indices: Vec<usize> = hashes.iter().map(|h| h % buckets).collect();

        // Prepare accumulation buffers per bucket
        let mut bucket_sums = vec![ndarray::Array1::<T>::zeros(embedding_dim); buckets];
        let mut bucket_counts = vec![T::zero(); buckets];

        // Accumulate token embeddings into buckets
        for (i, &bucket_index) in bucket_indices.iter().enumerate() {
            let token = multi_vector_tokens.row(i);
            bucket_sums[bucket_index] = &bucket_sums[bucket_index] + &token;
            bucket_counts[bucket_index] = bucket_counts[bucket_index] + T::one();
        }

        // Final aggregation: sum or average per bucket
        let mut result = Array1::<T>::zeros(buckets * embedding_dim);

        for (i, (vec, &count)) in bucket_sums.iter().zip(bucket_counts.iter()).enumerate() {
            let mut chunk = result.slice_mut(s![i * embedding_dim..(i + 1) * embedding_dim]);

            if count == T::zero() {
                chunk.fill(T::zero());
            } else if mode == Aggregation::Avg {
                // divide each value by count
                chunk.assign(&vec.mapv(|x| x / count));
            } else {
                // just copy the vector directly
                chunk.assign(vec);
            }
        }
        result
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
        let (batch_size, _, embedding_dim) = batch_tokens.dim();
        let buckets = self.buckets;

        // Pre-allocate output array
        let mut result = Array2::<T>::zeros((batch_size, buckets * embedding_dim));

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
    use super::*;
    use ndarray::array;

    const DIM: usize = 4;
    const BUCKETS: usize = 3;
    const SEED: u64 = 42;

    // Helper: create encoder with fixed seed
    fn create_encoder() -> FDEEncoder<f32> {
        FDEEncoder::new(BUCKETS, DIM, SEED)
    }

    #[test]
    fn test_new_hyperplanes_shape() {
        let enc = create_encoder();
        assert_eq!(enc.hyperplanes.shape(), &[DIM, BUCKETS]);
    }

    #[test]
    fn test_encode_output_shape() {
        let enc = create_encoder();
        // 2 tokens, each dim=4
        let tokens = array![[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]];
        let vec = enc.encode(tokens.view(), Aggregation::Sum);
        // output length = buckets * dim
        assert_eq!(vec.len(), BUCKETS * DIM);
    }

    #[test]
    fn test_encode_empty_tokens() {
        let enc = create_encoder();
        let tokens = Array2::<f32>::zeros((0, DIM));
        let vec = enc.encode(tokens.view(), Aggregation::Sum);
        assert_eq!(vec.len(), BUCKETS * DIM);
        assert!(vec.iter().all(|&x| x == 0.0));
    }

    #[test]
    #[should_panic]
    fn test_encode_dim_mismatch() {
        let enc = create_encoder();
        // tokens have dim=3, encoder expects 4
        let tokens = array![[0.1, 0.2, 0.3]];
        enc.encode(tokens.view(), Aggregation::Sum);
    }

    #[test]
    fn test_encode_query_vs_doc_aggregation() {
        let enc = create_encoder();
        let tokens = array![[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]];

        let sum_vec = enc.encode_query(tokens.view());
        let avg_vec = enc.encode_doc(tokens.view());

        // They should differ if buckets > 0
        assert_eq!(sum_vec.len(), avg_vec.len());
        assert!(sum_vec != avg_vec);
    }

    #[test]
    fn test_encode_single_token() {
        let enc = create_encoder();
        let tokens = array![[1.0, 2.0, 3.0, 4.0]];
        let vec = enc.encode(tokens.view(), Aggregation::Sum);
        assert_eq!(vec.len(), BUCKETS * DIM);
        assert!(vec.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_encode_all_zero_tokens() {
        let enc = create_encoder();
        let tokens = array![[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]];
        let vec = enc.encode(tokens.view(), Aggregation::Sum);
        assert_eq!(vec.len(), BUCKETS * DIM);
        assert!(vec.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_bucket_indices_in_range() {
        let enc = create_encoder();
        let tokens = array![
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2]
        ];
        let projections = tokens.dot(&enc.hyperplanes);
        let bin_mask = projections.mapv(|x| if x > 0.0 { 1u8 } else { 0u8 });
        let bin_mask_usize = bin_mask.mapv(|x| x as usize);
        let weights = Array1::from_iter((1..=BUCKETS).map(|i| i));
        let hashes: Array1<usize> = bin_mask_usize.dot(&weights);
        for h in hashes.iter() {
            let idx = h % BUCKETS;
            assert!(idx < BUCKETS);
        }
    }

    // batched encode test
    #[test]
    fn test_deterministic_output() {
        let enc = create_encoder();
        let tokens = array![[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]];
        let out1 = enc.encode(tokens.view(), Aggregation::Sum);
        let out2 = enc.encode(tokens.view(), Aggregation::Sum);
        assert_eq!(out1, out2);
    }
    #[test]
    fn test_batch_encode_output_shape() {
        let enc = create_encoder();
        // 2 batches, 3 tokens each, each token dim=4
        let batch_tokens = array![
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2]
            ],
            [
                [1.1, 1.2, 1.3, 1.4],
                [1.5, 1.6, 1.7, 1.8],
                [1.9, 2.0, 2.1, 2.2]
            ]
        ];
        let result = enc.batch_encode(batch_tokens.view(), Aggregation::Sum);
        // output shape: (batch_size, buckets * dim)
        assert_eq!(result.shape(), &[2, BUCKETS * DIM]);
    }

    #[test]
    fn test_batch_encode_empty_tokens() {
        let enc = create_encoder();
        // 2 batches, 0 tokens each, each token dim=4
        let batch_tokens = Array3::<f32>::zeros((2, 0, DIM));
        let result = enc.batch_encode(batch_tokens.view(), Aggregation::Sum);
        assert_eq!(result.shape(), &[2, BUCKETS * DIM]);
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    #[should_panic]
    fn test_batch_encode_dim_mismatch() {
        let enc = create_encoder();
        // tokens have dim=3, encoder expects 4
        let batch_tokens = array![
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]
        ];
        enc.batch_encode(batch_tokens.view(), Aggregation::Sum);
    }

    #[test]
    fn test_batch_encode_query_vs_doc_aggregation() {
        let enc = create_encoder();
        let batch_tokens = array![
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            [[3.0, 4.0, 5.0, 6.0], [4.0, 5.0, 6.0, 7.0]]
        ];

        let sum_result = enc.batch_encode(batch_tokens.view(), Aggregation::Sum);
        let avg_result = enc.batch_encode(batch_tokens.view(), Aggregation::Avg);

        // They should differ if buckets > 0
        assert_eq!(sum_result.shape(), avg_result.shape());
        assert!(sum_result != avg_result);
    }

    #[test]
    fn test_batch_encode_single_token_per_batch() {
        let enc = create_encoder();
        let batch_tokens = array![[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]];
        let result = enc.batch_encode(batch_tokens.view(), Aggregation::Sum);
        assert_eq!(result.shape(), &[2, BUCKETS * DIM]);
        assert!(result.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_batch_encode_all_zero_tokens() {
        let enc = create_encoder();
        let batch_tokens = array![
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        ];
        let result = enc.batch_encode(batch_tokens.view(), Aggregation::Sum);
        assert_eq!(result.shape(), &[2, BUCKETS * DIM]);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_batch_encode_deterministic_output() {
        let enc = create_encoder();
        let batch_tokens = array![
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            [[0.9, 1.0, 1.1, 1.2], [1.3, 1.4, 1.5, 1.6]]
        ];
        let out1 = enc.batch_encode(batch_tokens.view(), Aggregation::Sum);
        let out2 = enc.batch_encode(batch_tokens.view(), Aggregation::Sum);
        assert_eq!(out1, out2);
    }

    #[test]
    fn test_batch_encode_consistency_with_single_encode() {
        let enc = create_encoder();
        let single_tokens = array![[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]];
        let batch_tokens = array![[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]];

        let single_result = enc.encode(single_tokens.view(), Aggregation::Sum);
        let batch_result = enc.batch_encode(batch_tokens.view(), Aggregation::Sum);

        // First row of batch result should match single encode result
        assert_eq!(single_result, batch_result.row(0));
    }

    #[test]
    fn test_batch_encode_multiple_batches_consistency() {
        let enc = create_encoder();
        let tokens1 = array![[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]];
        let tokens2 = array![[0.9, 1.0, 1.1, 1.2], [1.3, 1.4, 1.5, 1.6]];

        let single1 = enc.encode(tokens1.view(), Aggregation::Sum);
        let single2 = enc.encode(tokens2.view(), Aggregation::Sum);

        let batch_tokens = array![
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            [[0.9, 1.0, 1.1, 1.2], [1.3, 1.4, 1.5, 1.6]]
        ];
        let batch_result = enc.batch_encode(batch_tokens.view(), Aggregation::Sum);

        // Each row should match the corresponding single encode result
        assert_eq!(single1, batch_result.row(0));
        assert_eq!(single2, batch_result.row(1));
    }

    #[test]
    fn test_batch_encode_different_token_counts() {
        let enc = create_encoder();
        // Create arrays with different token counts manually
        let mut batch_tokens = Array3::<f32>::zeros((2, 3, DIM));

        // First batch: 2 tokens
        batch_tokens.slice_mut(s![0, 0, ..]).assign(&array![0.1, 0.2, 0.3, 0.4]);
        batch_tokens.slice_mut(s![0, 1, ..]).assign(&array![0.5, 0.6, 0.7, 0.8]);
        // Third token in first batch remains zero

        // Second batch: 3 tokens
        batch_tokens.slice_mut(s![1, 0, ..]).assign(&array![0.9, 1.0, 1.1, 1.2]);
        batch_tokens.slice_mut(s![1, 1, ..]).assign(&array![1.3, 1.4, 1.5, 1.6]);
        batch_tokens.slice_mut(s![1, 2, ..]).assign(&array![1.7, 1.8, 1.9, 2.0]);

        let result = enc.batch_encode(batch_tokens.view(), Aggregation::Sum);
        assert_eq!(result.shape(), &[2, BUCKETS * DIM]);
        assert!(result.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_batch_encode_large_batch() {
        let enc = create_encoder();
        let batch_size = 100;
        let num_tokens = 5;

        // Create a large batch with random-like data
        let mut batch_tokens = Array3::<f32>::zeros((batch_size, num_tokens, DIM));
        for i in 0..batch_size {
            for j in 0..num_tokens {
                for k in 0..DIM {
                    batch_tokens[[i, j, k]] = (i + j + k) as f32 * 0.1;
                }
            }
        }

        let result = enc.batch_encode(batch_tokens.view(), Aggregation::Sum);
        assert_eq!(result.shape(), &[batch_size, BUCKETS * DIM]);

        // Verify not all results are zero
        assert!(result.iter().any(|&v| v != 0.0));

        // Verify all results are finite
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_batch_encode_aggregation_modes() {
        let enc = create_encoder();
        let batch_tokens = array![
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            [[3.0, 4.0, 5.0, 6.0], [4.0, 5.0, 6.0, 7.0]]
        ];

        let sum_result = enc.batch_encode(batch_tokens.view(), Aggregation::Sum);
        let avg_result = enc.batch_encode(batch_tokens.view(), Aggregation::Avg);

        // For sum aggregation, values should generally be larger than avg aggregation
        // (unless all tokens go to the same bucket)
        let sum_max: f32 = sum_result.fold(0.0_f32, |acc, &x| acc.max(x.abs()));
        let avg_max: f32 = avg_result.fold(0.0_f32, |acc, &x| acc.max(x.abs()));

        // This is a weak test, but should hold for most cases
        assert!(sum_max >= avg_max);
    }

    #[test]
    fn test_batch_encode_edge_cases() {
        let enc = create_encoder();

        // Test with very large values
        let batch_tokens = array![
            [[1e6, 2e6, 3e6, 4e6], [5e6, 6e6, 7e6, 8e6]],
            [[-1e6, -2e6, -3e6, -4e6], [-5e6, -6e6, -7e6, -8e6]]
        ];
        let result = enc.batch_encode(batch_tokens.view(), Aggregation::Sum);
        assert_eq!(result.shape(), &[2, BUCKETS * DIM]);
        assert!(result.iter().all(|&v| v.is_finite()));

        // Test with very small values
        let batch_tokens = array![
            [[1e-6, 2e-6, 3e-6, 4e-6], [5e-6, 6e-6, 7e-6, 8e-6]],
            [[-1e-6, -2e-6, -3e-6, -4e-6], [-5e-6, -6e-6, -7e-6, -8e-6]]
        ];
        let result = enc.batch_encode(batch_tokens.view(), Aggregation::Sum);
        assert_eq!(result.shape(), &[2, BUCKETS * DIM]);
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_encode_large_dim_bucket() {
        // Test with realistic large dimensions and buckets
        const LARGE_DIM: usize = 128;
        const LARGE_BUCKETS: usize = 128;
        const LARGE_SEED: u64 = 42;

        let encoder = FDEEncoder::new(LARGE_BUCKETS, LARGE_DIM, LARGE_SEED);
        
        // Create test data with large dimensions
        let tokens = Array2::from_shape_vec(
            (10, LARGE_DIM), 
            (0..10 * LARGE_DIM).map(|i| (i as f32) * 0.1).collect()
        ).unwrap();

        // Test both aggregation modes
        let sum_result = encoder.encode(tokens.view(), Aggregation::Sum);
        let avg_result = encoder.encode(tokens.view(), Aggregation::Avg);

        // Verify output shapes
        assert_eq!(sum_result.len(), LARGE_BUCKETS * LARGE_DIM);
        assert_eq!(avg_result.len(), LARGE_BUCKETS * LARGE_DIM);

        // Verify results are finite
        assert!(sum_result.iter().all(|&v| v.is_finite()));
        assert!(avg_result.iter().all(|&v| v.is_finite()));

        // Verify results are different (should be different due to different aggregation)
        assert!(sum_result != avg_result);

        // Verify not all results are zero
        assert!(sum_result.iter().any(|&v| v != 0.0));
        assert!(avg_result.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_batch_encode_large_dim_bucket() {
        // Test batch encoding with large dimensions and buckets
        const LARGE_DIM: usize = 128;
        const LARGE_BUCKETS: usize = 128;
        const LARGE_SEED: u64 = 42;

        let encoder = FDEEncoder::new(LARGE_BUCKETS, LARGE_DIM, LARGE_SEED);
        
        // Create batch test data
        let batch_tokens = Array3::from_shape_vec(
            (5, 10, LARGE_DIM), 
            (0..5 * 10 * LARGE_DIM).map(|i| (i as f32) * 0.1).collect()
        ).unwrap();

        // Test both aggregation modes
        let sum_result = encoder.batch_encode(batch_tokens.view(), Aggregation::Sum);
        let avg_result = encoder.batch_encode(batch_tokens.view(), Aggregation::Avg);

        // Verify output shapes
        assert_eq!(sum_result.shape(), &[5, LARGE_BUCKETS * LARGE_DIM]);
        assert_eq!(avg_result.shape(), &[5, LARGE_BUCKETS * LARGE_DIM]);

        // Verify results are finite
        assert!(sum_result.iter().all(|&v| v.is_finite()));
        assert!(avg_result.iter().all(|&v| v.is_finite()));

        // Verify results are different
        assert!(sum_result != avg_result);

        // Verify not all results are zero
        assert!(sum_result.iter().any(|&v| v != 0.0));
        assert!(avg_result.iter().any(|&v| v != 0.0));
    }
}
