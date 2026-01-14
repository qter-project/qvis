use std::{cmp::Ordering, collections::HashMap, sync::Arc};

use internment::ArcIntern;
use itertools::Itertools;
use kiddo::{KdTree, SquaredEuclidean};
use puzzle_theory::{
    permutations::{Permutation, PermutationGroup},
    puzzle_geometry::PuzzleGeometry,
};
use rand::Rng;

const CONFIDENCE_PERCENTILE: f64 = 0.2;
const MAX_NEAREST_N: usize = 10;
const MAX_FRACTION: usize = 8;

struct Pixel {
    idx: usize,
    kdtrees: HashMap<ArcIntern<str>, KdTree<f64, 3>>,
}

pub struct Inference {
    pixels_by_sticker: Box<[Box<[Pixel]>]>,
    group: Arc<PermutationGroup>,
    colors: Box<[ArcIntern<str>]>,
}

impl Inference {
    pub(crate) fn new(assignment: Box<[super::Pixel]>, puzzle: &PuzzleGeometry) -> Inference {
        let mut pixels_by_sticker: Vec<Vec<Pixel>> = Vec::new();

        let group = puzzle.permutation_group();

        for _ in 0..group.facelet_count() {
            pixels_by_sticker.push(Vec::new());
        }

        let colors: Box<[_]> = puzzle
            .permutation_group()
            .facelet_colors()
            .iter()
            .unique()
            .cloned()
            .collect();

        let empty_kdtrees: HashMap<ArcIntern<str>, KdTree<f64, 3>> = colors
            .iter()
            .cloned()
            .map(|a| (a, KdTree::<f64, 3>::new()))
            .collect();

        for (idx, pixel) in assignment.into_iter().enumerate() {
            let super::Pixel::Sticker(sticker) = pixel else {
                continue;
            };

            pixels_by_sticker[sticker].push(Pixel { idx, kdtrees: empty_kdtrees.clone() });
        }

        Inference {
            pixels_by_sticker: pixels_by_sticker.into_iter().map(|v| v.into()).collect(),
            group,
            colors,
        }
    }

    pub(crate) fn infer(&self, picture: &[(f64, f64, f64)]) -> Box<[HashMap<ArcIntern<str>, f64>]> {
        let mut rng = rand::rng();

        let mut confidences_by_pixel = self
            .colors
            .iter()
            .cloned()
            .map(|v| (v, Vec::<f64>::new()))
            .collect::<HashMap<_, _>>();

        self.pixels_by_sticker
            .iter()
            .map(|v| {
                // Maybe pick random subset
                for pixel in v {
                    for (color, kdtree) in &pixel.kdtrees {
                        let (r, g, b) = picture[pixel.idx];
                        let n = MAX_NEAREST_N
                            .min(kdtree.size() as usize / MAX_FRACTION)
                            .max(1);
                        let nn = kdtree.nearest_n::<SquaredEuclidean>(&[r, g, b], n);

                        // https://faculty.washington.edu/yenchic/18W_425/Lec7_knn_basis.pdf
                        // TODO: Try to account for non uniform distributions?
                        const UNIT_SPHERE: f64 = 4. / 3. * core::f64::consts::PI;
                        let density = n as f64 / kdtree.size() as f64
                            * (nn.last().unwrap().distance.powi(3) * UNIT_SPHERE).recip();

                        confidences_by_pixel.get_mut(color).unwrap().push(density);
                    }
                }

                confidences_by_pixel
                    .iter_mut()
                    .map(|(k, v)| {
                        if v.is_empty() {
                            return (ArcIntern::clone(k), 0.);
                        }

                        let n = (CONFIDENCE_PERCENTILE * v.len() as f64).floor() as usize;
                        quickselect(&mut rng, v, f64::total_cmp, n);
                        let confidence = v[n];
                        v.drain(..);
                        (ArcIntern::clone(k), confidence)
                    })
                    .collect()
            })
            .collect()
    }

    pub(crate) fn calibrate(&mut self, image: &[(f64, f64, f64)], state: Permutation) {
        for (sticker, pixels) in self.pixels_by_sticker.iter_mut().enumerate() {
            let color = &self.group.facelet_colors()[state.comes_from().get(sticker)];

            for pixel in pixels {
                let (r, g, b) = image[pixel.idx];
                pixel.kdtrees.get_mut(color).unwrap().add(&[r, g, b], 0);
            }
        }
    }
}

// This quickselect code is copied from <https://gitlab.com/hrovnyak/nmr-schedule>

fn partition<T, R: Rng + ?Sized>(
    rng: &mut R,
    slice: &mut [T],
    by: &impl Fn(&T, &T) -> Ordering,
) -> usize {
    slice.swap(0, rng.random_range(0..slice.len()));

    let mut i = 1;
    let mut j = slice.len() - 1;

    loop {
        while i < slice.len() && !matches!(by(&slice[i], &slice[0]), Ordering::Less) {
            i += 1;
        }

        while matches!(by(&slice[j], &slice[0]), Ordering::Less) {
            j -= 1;
        }

        // If the indices crossed, return
        if i > j {
            slice.swap(0, j);
            return j;
        }

        // Swap the elements at the left and right indices
        slice.swap(i, j);
        i += 1;
    }
}

/// Standard quickselect algorithm: https://en.wikipedia.org/wiki/Quickselect
/// Sorts in descending order
///
/// After calling this function, the value at index `find_spot` is guaranteed to be at the correctly sorted position and all values at indices less than `find_spot` are guaranteed to be greater than the value at `find_spot` and vice versa for indices greater.
pub(crate) fn quickselect<T, R: Rng + ?Sized>(
    rng: &mut R,
    mut slice: &mut [T],
    by: impl Fn(&T, &T) -> Ordering,
    mut find_spot: usize,
) {
    loop {
        let len = slice.len();

        if len < 2 {
            return;
        }

        let spot_found = partition(rng, slice, &by);

        match find_spot.cmp(&spot_found) {
            Ordering::Less => slice = &mut slice[0..spot_found],
            Ordering::Equal => return,
            Ordering::Greater => {
                slice = &mut slice[spot_found + 1..len];
                find_spot = find_spot - spot_found - 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::quickselect;

    #[test]
    fn test_quickselect() {
        fn verify<R: Rng + ?Sized>(rng: &mut R, pos: usize, slice: &[f64]) {
            let mut slice = slice
                .iter()
                .enumerate()
                .map(|(a, b)| (*b, a))
                .collect::<Vec<_>>();

            quickselect(rng, &mut slice, |a, b| a.0.total_cmp(&b.0), pos);

            for i in 0..pos {
                assert!(
                    slice[i].0 >= slice[pos].0,
                    "Pos: {pos}, Index: {i} - {slice:?}"
                );
            }

            for i in pos + 1..slice.len() {
                assert!(
                    slice[i].0 <= slice[pos].0,
                    "Pos: {pos}, Index: {i} - {slice:?}"
                );
            }

            let v = slice[pos];

            slice.sort_by(|a, b| b.0.total_cmp(&a.0));

            assert_eq!(slice[pos].0, v.0);
        }

        let mut rng = rand::rng();

        verify(&mut rng, 2, &[5., 4., 3., 2., 1.]);
        verify(&mut rng, 2, &[1., 2., 3., 4., 5.]);
        verify(&mut rng, 3, &[1., 2., 1., 4., 3.]);

        for i in 0..100 {
            let pos = rng.random_range(0..i + 1);
            let data = (0..i + 1).map(|_| rng.random()).collect::<Vec<_>>();
            verify(&mut rng, pos, &data);
        }
    }
}
