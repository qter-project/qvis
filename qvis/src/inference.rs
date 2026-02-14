use std::{cmp::Ordering, collections::HashMap, sync::OnceLock};

use internment::ArcIntern;
use itertools::Itertools;
use kiddo::{KdTree, SquaredEuclidean};
use puzzle_theory::{
    permutations::{Permutation, PermutationGroup},
    puzzle_geometry::PuzzleGeometry,
};
use rand::Rng;
use serde::{Deserialize, Serialize};

const CONFIDENCE_PERCENTILE: f64 = 0.2;
const MAX_NEAREST_N: usize = 10;
const MAX_FRACTION: usize = 8;

fn white_balance(mut color: (f64, f64, f64), neutral: (f64, f64, f64)) -> (f64, f64, f64) {
    color.0 /= neutral.0;
    color.1 /= neutral.1;
    color.2 /= neutral.2;

    color
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Pixel {
    pub(crate) idx: usize,
    kdtrees: HashMap<ArcIntern<str>, KdTree<f64, 3>>,
}

impl Pixel {
    fn density(kdtree: &KdTree<f64, 3>, (r, g, b): (f64, f64, f64)) -> Option<f64> {
        let n = MAX_NEAREST_N
            .min(kdtree.size() as usize / MAX_FRACTION)
            .max(1);
        let nn = kdtree.nearest_n::<SquaredEuclidean>(&[r, g, b], n);

        // https://faculty.washington.edu/yenchic/18W_425/Lec7_knn_basis.pdf
        // TODO: Try to account for non uniform distributions?
        const UNIT_SPHERE: f64 = 4. / 3. * core::f64::consts::PI;

        let last = nn.last()?;

        Some(n as f64 / kdtree.size() as f64 * (last.distance.sqrt().powi(3) * UNIT_SPHERE).recip())
    }

    fn densities(
        &self,
        at: (f64, f64, f64),
        wb: (f64, f64, f64),
    ) -> impl Iterator<Item = (&ArcIntern<str>, f64)> {
        self.kdtrees.iter().filter_map(move |(color, kdtree)| {
            let at = white_balance(at, wb);

            Some((color, Self::density(kdtree, at)?))
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inference {
    pub(crate) pixels_by_sticker: Box<[Box<[Pixel]>]>,
    pub(crate) white_balance_by_face: HashMap<ArcIntern<str>, Box<[usize]>>,
    colors: Box<[ArcIntern<str>]>,
    #[serde(skip)]
    max_confidence: OnceLock<f64>,
}

impl Inference {
    pub fn new(assignment: Box<[super::Pixel]>, puzzle: &PuzzleGeometry) -> Inference {
        let group = puzzle.permutation_group();

        let mut pixels_by_sticker: Vec<Vec<Pixel>> = Vec::new();

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

        let mut white_balance_by_face = colors
            .iter()
            .cloned()
            .map(|v| (v, Vec::<usize>::new()))
            .collect::<HashMap<_, _>>();

        for (idx, pixel) in assignment.into_iter().enumerate() {
            match pixel {
                crate::Pixel::Unassigned => {}
                crate::Pixel::WhiteBalance(arc_intern) => white_balance_by_face
                    .get_mut(&arc_intern)
                    .unwrap()
                    .push(idx),
                crate::Pixel::Sticker(sticker) => {
                    pixels_by_sticker[sticker].push(Pixel {
                        idx,
                        kdtrees: empty_kdtrees.clone(),
                    });
                }
            }
        }

        Inference {
            pixels_by_sticker: pixels_by_sticker.into_iter().map(|v| v.into()).collect(),
            white_balance_by_face: white_balance_by_face
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect(),
            colors,
            max_confidence: OnceLock::new(),
        }
    }

    fn white_balance(
        &self,
        picture: &[(f64, f64, f64)],
    ) -> HashMap<ArcIntern<str>, (f64, f64, f64)> {
        self.white_balance_by_face
            .iter()
            .map(|(k, v)| {
                let white = v
                    .iter()
                    .map(|idx| picture[*idx])
                    .tree_reduce(|(r1, g1, b1), (r2, g2, b2)| (r1 + r2, g1 + g2, b1 + b2));

                (
                    ArcIntern::clone(k),
                    match white {
                        Some((r, g, b)) => {
                            let len = v.len() as f64;

                            (r / len, g / len, b / len)
                        }
                        None => (1., 1., 1.),
                    },
                )
            })
            .collect()
    }

    pub fn infer(
        &self,
        picture: &[(f64, f64, f64)],
        group: &PermutationGroup,
    ) -> Box<[HashMap<ArcIntern<str>, f64>]> {
        let mut rng = rand::rng();

        let mut confidences_by_pixel = self
            .colors
            .iter()
            .cloned()
            .map(|v| (v, Vec::<f64>::new()))
            .collect::<HashMap<_, _>>();

        let wb = self.white_balance(picture);

        let facelet_count_adjust = self.pixels_by_sticker.len() as f64;
        let no_data = (self.colors.len() as f64 * facelet_count_adjust).recip();

        self.pixels_by_sticker
            .iter()
            .enumerate()
            .map(|(idx, v)| {
                let wb = *wb.get(&group.facelet_colors()[idx]).unwrap();

                // Maybe pick random subset
                for (color, density) in v
                    .iter()
                    .flat_map(|pixel| pixel.densities(picture[pixel.idx], wb))
                {
                    confidences_by_pixel.get_mut(color).unwrap().push(density)
                }

                let items = confidences_by_pixel
                    .iter_mut()
                    .map(|(k, v)| {
                        let confidence = representative_confidence(v, &mut rng);
                        v.drain(..);
                        (ArcIntern::clone(k), confidence)
                    })
                    .collect::<Vec<_>>();

                let mut normalization = items.iter().filter_map(|v| v.1).sum::<f64>();

                let len = items.len() as f64;
                normalization *= len;
                normalization /= items.iter().filter(|v| v.1.is_some()).count() as f64;

                normalization *= facelet_count_adjust;

                items
                    .into_iter()
                    .map(|(k, v)| {
                        (
                            k,
                            match v {
                                Some(v) => v / normalization,
                                None => no_data,
                            },
                        )
                    })
                    .collect()
            })
            .collect()
    }

    pub fn calibrate(
        &mut self,
        image: &[(f64, f64, f64)],
        state: &Permutation,
        group: &PermutationGroup,
    ) {
        self.max_confidence = OnceLock::new();

        let wb = self.white_balance(image);

        for (sticker, pixels) in self.pixels_by_sticker.iter_mut().enumerate() {
            let wb = *wb.get(&group.facelet_colors()[sticker]).unwrap();
            let color = &group.facelet_colors()[state.state().get(sticker)];

            for pixel in pixels {
                let (r, g, b) = white_balance(image[pixel.idx], wb);
                pixel.kdtrees.get_mut(color).unwrap().add(&[r, g, b], 0);
            }
        }
    }
}

fn representative_confidence<R: Rng + ?Sized>(confidences: &mut [f64], rng: &mut R) -> Option<f64> {
    if confidences.is_empty() {
        return None;
    }

    let n = (CONFIDENCE_PERCENTILE * confidences.len() as f64).floor() as usize;
    quickselect(rng, confidences, f64::total_cmp, n);
    Some(confidences[n])
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
    use std::{collections::HashMap, sync::LazyLock};

    use internment::ArcIntern;
    use puzzle_theory::{
        permutations::{Permutation, PermutationGroup, schreier_sims::StabilizerChain},
        puzzle_geometry::parsing::puzzle,
    };
    use rand::{Rng, SeedableRng};

    use crate::{inference::Inference, puzzle_matching::Matcher};

    use super::quickselect;

    static NATURAL_COLORS: LazyLock<HashMap<ArcIntern<str>, (f64, f64, f64)>> =
        LazyLock::new(|| {
            let mut map = HashMap::new();

            map.insert(ArcIntern::from("red"), (1., 0.2, 0.2));
            map.insert(ArcIntern::from("orange"), (1., 0.6, 0.2));
            map.insert(ArcIntern::from("white"), (1., 1., 1.));
            map.insert(ArcIntern::from("yellow"), (0.8, 0.8, 0.2));
            map.insert(ArcIntern::from("blue"), (0.2, 0.5, 1.));
            map.insert(ArcIntern::from("green"), (0.3, 1., 0.5));

            map
        });

    fn simulate_picture<R: Rng + ?Sized>(
        perm: &Permutation,
        group: &PermutationGroup,
        shadowful_noise: f64,
        colorful_noise: f64,
        rng: &mut R,
        out: &mut [(f64, f64, f64)],
    ) {
        assert_eq!(out.len(), (48 + 6) * 20);

        for face in 0..6 {
            // Simulate random lighting on this face to test our white balance code
            let white = (
                rng.random_range(0.2..1.2),
                rng.random_range(0.2..1.2),
                rng.random_range(0.2..1.2),
            );

            for spot in 0..8 {
                let idx = spot + face * 8;

                let is = perm.state().get(idx);
                let mut color = *NATURAL_COLORS.get(&group.facelet_colors()[is]).unwrap();
                color.0 *= white.0;
                color.1 *= white.1;
                color.2 *= white.2;

                out[idx * 20..(idx + 1) * 20].fill(color);
            }

            out[(48 + face) * 20..(48 + face + 1) * 20].fill(white);
        }

        for (r, g, b) in out.iter_mut() {
            // Add random noise to simulate differences in shadowing
            let noise = rng.random_range(((1. + shadowful_noise).recip())..(1. + shadowful_noise));
            *r *= noise;
            *g *= noise;
            *b *= noise;
        }

        for v in out.iter_mut().flat_map(|(r, g, b)| [r, g, b]) {
            // Add some random noise to simulate camera noise
            *v *= rng.random_range(((1. + colorful_noise).recip())..(1. + colorful_noise));
            // Clamp everything to be less than one to simulate overexposure
            *v = v.min(1.);
        }
    }

    fn conf_in(
        inference: &[HashMap<ArcIntern<str>, f64>],
        group: &PermutationGroup,
        perm: &Permutation,
    ) -> f64 {
        let mut ll = 0.;

        for v in 0..48 {
            ll += inference[v]
                .get(&group.facelet_colors()[perm.state().get(v)])
                .unwrap();
        }

        ll
    }

    #[test]
    fn test_inference() {
        let mut assignment = Vec::new();

        for i in 0..48 {
            for _ in 0..20 {
                assignment.push(crate::Pixel::Sticker(i));
            }
        }

        for color in ["white", "orange", "green", "red", "blue", "yellow"].map(ArcIntern::from) {
            for _ in 0..20 {
                assignment.push(crate::Pixel::WhiteBalance(ArcIntern::clone(&color)))
            }
        }

        let puzzle = puzzle("3x3");
        let group = puzzle.permutation_group();
        let stabchain = StabilizerChain::new(&group);

        let mut inference = Inference::new(assignment.into(), &puzzle);

        let mut rng = rand::rngs::SmallRng::from_seed(*b"Buying black on the black market");

        let mut img = [(0., 0., 0.); (48 + 6) * 20];

        for _ in 0..30 {
            let perm = stabchain.random(&mut rng);
            simulate_picture(&perm, &group, 0.3, 0.2, &mut rng, &mut img);
            inference.calibrate(&img, &perm, &group);
        }

        let matcher = Matcher::new(&puzzle);

        for _ in 0..1000 {
            let perm = stabchain.random(&mut rng);
            simulate_picture(&perm, &group, 0.3, 0.2, &mut rng, &mut img);
            let inference = inference.infer(&img, &group);
            let (perm_inferred, conf) = matcher.most_likely(&inference, &puzzle);

            assert!(0. <= conf, "{conf}");
            assert!(conf <= 1., "{conf}");
            assert_eq!(perm_inferred, perm);
        }

        for _ in 0..1000 {
            let perm = stabchain.random(&mut rng);
            simulate_picture(&perm, &group, 0.5, 0.4, &mut rng, &mut img);
            let inference = inference.infer(&img, &group);
            let (perm_inferred, conf) = matcher.most_likely(&inference, &puzzle);

            assert!(0. <= conf, "{conf}");
            assert!(conf <= 1., "{conf}");

            assert!(
                conf_in(&inference, &group, &perm) <= conf_in(&inference, &group, &perm_inferred)
            )
        }
    }

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
