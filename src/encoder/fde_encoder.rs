use crate::types::{Aggregation, FDEFloat};
use ndarray::{Array1, Array2, ArrayView2, s};
use rand;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

/// Trait for fixed-dimensional encoding from token embeddings.
pub trait FDEEncoding<T: FDEFloat> {
    fn encode(&self, tokens: ArrayView2<T>, mode: Aggregation) -> Array1<T>;

    // fn batch_encode(&self, tokens: ArrayView2<T>, mode: Aggregation) -> Vec<T>;

    fn encode_query(&self, tokens: ArrayView2<T>) -> Array1<T> {
        self.encode(tokens, Aggregation::Sum)
    }

    fn encode_doc(&self, tokens: ArrayView2<T>) -> Array1<T> {
        self.encode(tokens, Aggregation::Avg)
    }

}

pub struct FDEEncoder<T: FDEFloat> {
    pub buckets: usize, // number of the hyperplanes
    pub dim: usize,     // embedding size
    pub hyperplanes: Array2<T>,
}

impl<T: FDEFloat> FDEEncoder<T> {
    pub fn new(buckets: usize, dim: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Create a vector of size buckets * dim filled with standard normal distributed floats
        let data: Vec<T> = (0..buckets * dim)
            .map(|_| {
                let sample: f64 = StandardNormal.sample(&mut rng);
                T::from(sample).unwrap() // safe unwrap since f32/f64 supported
            })
            .collect();

        let hyperplanes = Array2::from_shape_vec((buckets, dim), data)
            .expect("Failed to create hyperplanes array");

        Self {
            buckets,
            dim,
            hyperplanes,
        }
    }
}

impl<T: FDEFloat> FDEEncoding<T> for FDEEncoder<T> {
    fn encode(&self, multi_vector_tokens: ArrayView2<T>, mode: Aggregation) -> Array1<T> {
        let embedding_dim = multi_vector_tokens.ncols();
        let buckets = self.buckets;

        assert_eq!(embedding_dim, self.dim);

        // 1) Project tokens onto hyperplanes (vectorized)
        // projections shape: (num_tokens, buckets)
        let projections: Array2<T> = multi_vector_tokens.dot(&self.hyperplanes.t());

        // 2) Convert projections > 0 to binary mask (u8)
        let bin_mask = projections.mapv(|x| if x > T::zero() { 1u8 } else { 0u8 });

        // 3) Cast mask to usize for bitwise summation
        let bin_mask_usize = bin_mask.mapv(|x| x as usize);

        // 4) Create powers of two vector (bit weights)
        let powers = Array1::from_iter((0..buckets).map(|i| 1 << i));

        // 5) Compute hashes as dot product between mask and powers (vectorized)
        let hashes: Array1<usize> = bin_mask_usize.dot(&powers);

        // 6) Modulo to get bucket indices
        let bucket_indices: Vec<usize> = hashes.iter().map(|h| h % buckets).collect();

        // Prepare accumulation buffers per bucket
        let mut bucket_sums = vec![ndarray::Array1::<T>::zeros(embedding_dim); buckets];
        let mut bucket_counts = vec![T::zero(); buckets];

        // 7) Accumulate token embeddings into buckets
        for (i, &bucket_index) in bucket_indices.iter().enumerate() {
            let token = multi_vector_tokens.row(i);
            bucket_sums[bucket_index] = &bucket_sums[bucket_index] + &token;
            bucket_counts[bucket_index] = bucket_counts[bucket_index] + T::one();
        }

        // 8) Final aggregation: sum or average per bucket
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
        assert_eq!(enc.hyperplanes.shape(), &[BUCKETS, DIM]);
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
        let projections = tokens.dot(&enc.hyperplanes.t());
        let bin_mask = projections.mapv(|x| if x > 0.0 { 1u8 } else { 0u8 });
        let bin_mask_usize = bin_mask.mapv(|x| x as usize);
        let powers = ndarray::Array1::from_iter((0..BUCKETS).map(|i| 1 << i));
        let hashes: ndarray::Array1<usize> = bin_mask_usize.dot(&powers);
        for h in hashes.iter() {
            let idx = h % BUCKETS;
            assert!(idx < BUCKETS);
        }
    }

    #[test]
    fn test_deterministic_output() {
        let enc = create_encoder();
        let tokens = array![[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]];
        let out1 = enc.encode(tokens.view(), Aggregation::Sum);
        let out2 = enc.encode(tokens.view(), Aggregation::Sum);
        assert_eq!(out1, out2);
    }
}
