use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap},
    sync::Arc,
};

use internment::ArcIntern;
use itertools::Itertools;
use ndarray::{Array2, Array3, ArrayRef2, ArrayRef3, Axis, s};
use puzzle_theory::{
    permutations::{Permutation, PermutationGroup, schreier_sims::StabilizerChain},
    puzzle_geometry::{OrbitData, OriNum, PuzzleGeometry},
};

use crate::puzzle_matching::hungarian_algorithm::maximum_matching;

mod hungarian_algorithm;

pub struct Matcher {
    orbits: Box<[OrbitMatcher]>,
    stab_chain: StabilizerChain,
}

impl Matcher {
    pub fn new(puzzle: Arc<PuzzleGeometry>) -> Matcher {
        let data = puzzle.pieces_data();

        let orbits = data
            .orbits()
            .iter()
            .map(|orbit| OrbitMatcher::new(Arc::clone(&puzzle), orbit))
            .collect();

        Matcher {
            orbits,
            stab_chain: StabilizerChain::new(&puzzle.permutation_group()),
        }
    }

    pub fn most_likely(
        &self,
        log_likelihoods: &[HashMap<ArcIntern<str>, f64>],
    ) -> (Permutation, f64) {
        let iters = self
            .orbits
            .iter()
            .map(|v| SavedIter {
                iter: v.most_likely_matchings(log_likelihoods),
                saved: Vec::new(),
            })
            .collect();

        let mut states = PuzzleIter::new(iters);

        states
            .find(|(v, _)| self.stab_chain.is_member(v.clone()))
            .unwrap()
    }
}

struct SavedIter<I: Iterator<Item = (Permutation, f64)>> {
    saved: Vec<(Permutation, f64)>,
    iter: I,
}

impl<I: Iterator<Item = (Permutation, f64)>> SavedIter<I> {
    fn get(&mut self, i: usize) -> (&Permutation, f64) {
        while self.saved.len() <= i {
            self.saved.push(self.iter.next().unwrap());
        }

        let (perm, ll) = self.saved.get(i).unwrap();
        (perm, *ll)
    }
}

struct PuzzleIter<I: Iterator<Item = (Permutation, f64)>> {
    heap: BinaryHeap<PuzzleHeapElt>,
    iters: Box<[SavedIter<I>]>,
    cache: Option<PuzzleHeapElt>,
}

impl<I: Iterator<Item = (Permutation, f64)>> PuzzleIter<I> {
    fn new(mut iters: Box<[SavedIter<I>]>) -> PuzzleIter<I> {
        let mut heap = BinaryHeap::new();

        heap.push(PuzzleHeapElt::new(vec![0; iters.len()].into(), &mut iters));

        PuzzleIter {
            heap,
            iters,
            cache: None,
        }
    }
}

impl<I: Iterator<Item = (Permutation, f64)>> Iterator for PuzzleIter<I> {
    type Item = (Permutation, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(prev) = self.cache.take() {
            let splitted = prev.split(&mut self.iters);

            self.heap.extend(splitted);
        }

        let item = self.heap.pop()?;

        while self.heap.peek().map(|v| &*v.idxs) == Some(&item.idxs) {
            self.heap.pop();
        }

        let mut ll = 0.;
        let cycles = item
            .idxs
            .iter()
            .zip(&mut self.iters)
            .flat_map(|(v, iter)| {
                let (perm, orbit_ll) = iter.get(*v);
                ll += orbit_ll;
                perm.cycles().iter()
            })
            .cloned()
            .collect();

        self.cache = Some(item);

        Some((Permutation::from_cycles(cycles), ll))
    }
}

struct PuzzleHeapElt {
    idxs: Box<[usize]>,
    log_likelihood: f64,
}

impl PuzzleHeapElt {
    fn new<I: Iterator<Item = (Permutation, f64)>>(
        idxs: Box<[usize]>,
        iters: &mut [SavedIter<I>],
    ) -> PuzzleHeapElt {
        let ll = idxs
            .iter()
            .zip(iters.iter_mut())
            .map(|(idx, iter)| iter.get(*idx).1)
            .sum::<f64>();

        PuzzleHeapElt {
            idxs,
            log_likelihood: ll,
        }
    }

    fn split<I: Iterator<Item = (Permutation, f64)>>(
        &self,
        iters: &mut [SavedIter<I>],
    ) -> Vec<PuzzleHeapElt> {
        (0..self.idxs.len())
            .map(|i| {
                let mut idxs = self.idxs.clone();
                idxs[i] += 1;
                PuzzleHeapElt::new(idxs, iters)
            })
            .collect_vec()
    }
}

impl PartialEq for PuzzleHeapElt {
    fn eq(&self, other: &Self) -> bool {
        self.idxs == other.idxs
    }
}

impl Eq for PuzzleHeapElt {}

impl PartialOrd for PuzzleHeapElt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PuzzleHeapElt {
    fn cmp(&self, other: &Self) -> Ordering {
        self.log_likelihood.total_cmp(&other.log_likelihood)
    }
}

struct OrbitMatcher {
    stab_chain: StabilizerChain,
    // Maps the observation (sticker orientation num, color) to all (piece, orientation) that would be consistent with it
    sticker_color_piece: HashMap<(OriNum, ArcIntern<str>), Vec<(usize, usize)>>,
    orbit: OrbitData,
    puzzle: Arc<PuzzleGeometry>,
}

impl OrbitMatcher {
    fn new(puzzle: Arc<PuzzleGeometry>, orbit: &OrbitData) -> OrbitMatcher {
        let pieces_data = puzzle.pieces_data();
        let ori_nums = pieces_data.orientation_numbers();
        let group = puzzle.permutation_group();

        let ori_count = orbit.orientation_count();

        let mut sticker_color_piece =
            HashMap::<(OriNum, ArcIntern<str>), Vec<(usize, usize)>>::new();

        let mut sticker_in_orbit = vec![false; group.facelet_count()];

        for (i, piece) in orbit.pieces().iter().enumerate() {
            for sticker in piece.stickers() {
                sticker_in_orbit[*sticker] = true;

                let mut current_sticker = *sticker;
                for ori in 0..ori_count {
                    let ori_num = ori_nums[current_sticker];
                    let color = ArcIntern::clone(&group.facelet_colors()[*sticker]);

                    let pieces = sticker_color_piece.entry((ori_num, color)).or_default();
                    pieces.push((i, ori));

                    current_sticker = piece.twist().goes_to().get(current_sticker);
                }
            }
        }

        let subgroup = PermutationGroup::new(
            group.facelet_colors().to_owned(),
            group.piece_assignments().to_owned(),
            group
                .generators()
                .map(|(name, perm)| {
                    let new_perm = Permutation::from_mapping(
                        perm.goes_to()
                            .minimal()
                            .iter()
                            .enumerate()
                            .map(|(i, v)| if sticker_in_orbit[i] { *v } else { i })
                            .collect(),
                    );

                    (name, new_perm)
                })
                .collect(),
        );

        OrbitMatcher {
            stab_chain: StabilizerChain::new(&Arc::new(subgroup)),
            sticker_color_piece,
            orbit: orbit.to_owned(),
            puzzle,
        }
    }

    fn most_likely_matchings(
        &self,
        log_likelihoods: &[HashMap<ArcIntern<str>, f64>],
    ) -> impl Iterator<Item = (Permutation, f64)> {
        // Data for matching piece i to piece j where piece j gives the cost for each possible orientation
        let mut cost_matrix = Array3::zeros([
            self.orbit.pieces().len(),
            self.orbit.pieces().len(),
            self.orbit.orientation_count(),
        ]);

        for (piece, mut cost_row) in self
            .orbit
            .pieces()
            .iter()
            .zip(cost_matrix.axis_iter_mut(Axis(0)))
        {
            let pieces_data = self.puzzle.pieces_data();
            let ori_nums = pieces_data.orientation_numbers();

            for sticker in piece.stickers() {
                let ori_num = ori_nums[*sticker];

                for (color, log_likelihood) in log_likelihoods[*sticker]
                    .iter()
                    .map(|(a, b)| (ArcIntern::clone(a), *b))
                {
                    for (piece, ori) in self.sticker_color_piece.get(&(ori_num, color)).unwrap() {
                        cost_row[[*piece, *ori]] += log_likelihood;
                    }
                }
            }
        }

        println!("{cost_matrix}");

        let mut heap = BinaryHeap::new();
        heap.push(OrbitHeapElt::new(&cost_matrix));

        MatchIter {
            orbit_matcher: self,
            cost_matrix,
            heap,
            cache: None,
            facelet_count: self.puzzle.permutation_group().facelet_count(),
        }
        .filter(|(perm, _)| self.stab_chain.is_member(perm.clone()))
    }
}

struct MatchIter<'a> {
    orbit_matcher: &'a OrbitMatcher,
    cost_matrix: Array3<f64>,
    heap: BinaryHeap<OrbitHeapElt>,
    facelet_count: usize,
    // Save the HeapElt we just returned instead of splitting it and putting it in the heap immediately
    cache: Option<OrbitHeapElt>,
}

impl<'a> Iterator for MatchIter<'a> {
    type Item = (Permutation, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(item) = self.cache.take() {
            self.heap.extend(item.split(&self.cost_matrix));
        }

        let item = self.heap.pop()?;

        while self.heap.peek().map(|v| &v.allowed) == Some(&item.allowed) {
            self.heap.pop();
        }

        let data = self.orbit_matcher.puzzle.pieces_data();
        let ori_nums = data.orientation_numbers();

        let mut mapping_comes_from = (0..self.facelet_count).collect_vec();
        for (spot, (is, ori)) in item.matching.iter().enumerate() {
            for sticker_spot in self.orbit_matcher.orbit.pieces()[spot].stickers() {
                let target_ori = ori_nums[*sticker_spot] + *ori;
                mapping_comes_from[*sticker_spot] = *self.orbit_matcher.orbit.pieces()[*is]
                    .stickers()
                    .iter()
                    .find(|v| ori_nums[**v] == target_ori)
                    .unwrap();
            }
        }

        let ll = item.log_likelihood;

        self.cache = Some(item);

        Some((Permutation::from_state(mapping_comes_from), ll))
    }
}

#[derive(Debug)]
struct OrbitHeapElt {
    allowed: Array3<bool>,
    cost_matrix_2d: Array2<Option<f64>>,
    oris_chosen: Array2<Option<usize>>,
    log_likelihood: f64,
    matching: Vec<(usize, usize)>,
}

impl OrbitHeapElt {
    fn new(cost_matrix_3d: &ArrayRef3<f64>) -> OrbitHeapElt {
        let maxima = cost_matrix_3d.map_axis(Axis(2), |v| {
            v.into_iter()
                .copied()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap()
        });

        let cost_matrix_2d = maxima.map(|v| Some(v.1));
        let oris_chosen = maxima.map(|v| Some(v.0));

        let (matching, log_likelihood) = Self::mk_matching(&cost_matrix_2d, &oris_chosen).unwrap();

        OrbitHeapElt {
            allowed: Array3::from_elem(cost_matrix_3d.raw_dim(), true),
            cost_matrix_2d,
            oris_chosen,
            log_likelihood,
            matching,
        }
    }

    fn mk_matching(
        cost_matrix_2d: &ArrayRef2<Option<f64>>,
        oris_chosen: &ArrayRef2<Option<usize>>,
    ) -> Option<(Vec<(usize, usize)>, f64)> {
        let matching = maximum_matching(cost_matrix_2d)?
            .into_iter()
            .enumerate()
            .map(|(i, j)| (j, oris_chosen[[i, j]].unwrap()))
            .collect_vec();

        let log_likelihood = matching
            .iter()
            .enumerate()
            .map(|(i, (j, _))| cost_matrix_2d[[i, *j]].unwrap())
            .sum();

        Some((matching, log_likelihood))
    }

    fn split(&self, cost_matrix_3d: &ArrayRef3<f64>) -> impl Iterator<Item = OrbitHeapElt> {
        self.matching
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(i, (j, ori))| {
                let mut allowed = self.allowed.clone();
                let mut cost_matrix_2d = self.cost_matrix_2d.clone();
                let mut oris_chosen = self.oris_chosen.clone();

                allowed[[i, j, ori]] = false;

                let maybe_ori = cost_matrix_3d
                    .slice(s![i, j, ..])
                    .iter()
                    .zip(allowed.slice(s![i, j, ..]))
                    .enumerate()
                    .filter(|(_, (_, v))| **v)
                    .max_by(|(_, (a, _)), (_, (b, _))| a.total_cmp(b))
                    .map(|(a, (b, _))| (a, *b));

                cost_matrix_2d[[i, j]] = maybe_ori.map(|(_, v)| v);
                oris_chosen[[i, j]] = maybe_ori.map(|(v, _)| v);

                let (matching, log_likelihood) = Self::mk_matching(&cost_matrix_2d, &oris_chosen)?;

                Some(OrbitHeapElt {
                    allowed,
                    cost_matrix_2d,
                    oris_chosen,
                    log_likelihood,
                    matching,
                })
            })
    }
}

impl PartialEq for OrbitHeapElt {
    fn eq(&self, other: &Self) -> bool {
        self.allowed == other.allowed
    }
}

impl Eq for OrbitHeapElt {}

impl PartialOrd for OrbitHeapElt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrbitHeapElt {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.allowed == other.allowed {
            return Ordering::Equal;
        }

        self.log_likelihood.total_cmp(&other.log_likelihood)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use internment::ArcIntern;
    use itertools::Itertools;
    use ndarray::array;
    use puzzle_theory::{
        permutations::{Algorithm, Permutation},
        puzzle_geometry::parsing::puzzle,
    };

    use crate::puzzle_matching::{Matcher, OrbitHeapElt, PuzzleIter, SavedIter};

    #[test]
    fn heap_elt() {
        let cost_matrix_3d = array![
            [[-8., -10.], [-4., -10.], [-7., -10.],],
            [[-6., -10.], [-2., -10.], [-3., -10.],],
            [[-9., -10.], [-4., -10.], [-8., -10.],]
        ];

        let elt = OrbitHeapElt::new(&cost_matrix_3d);

        assert_eq!(elt.log_likelihood, -15.);
        assert_eq!(
            elt.cost_matrix_2d,
            array![
                [Some(-8.), Some(-4.), Some(-7.)],
                [Some(-6.), Some(-2.), Some(-3.)],
                [Some(-9.), Some(-4.), Some(-8.)],
            ]
        );
        assert_eq!(
            elt.oris_chosen,
            array![[0, 0, 0], [0, 0, 0], [0, 0, 0]].mapv(Some)
        );
        assert_eq!(
            elt.allowed,
            array![
                [[true, true], [true, true], [true, true]],
                [[true, true], [true, true], [true, true]],
                [[true, true], [true, true], [true, true]],
            ]
        );
        assert_eq!(elt.matching, vec![(0, 0), (2, 0), (1, 0)]);

        let splits = elt.split(&cost_matrix_3d).collect_vec();

        assert_eq!(splits.len(), 3);

        assert_eq!(
            splits[0].cost_matrix_2d,
            array![
                [Some(-10.), Some(-4.), Some(-7.)],
                [Some(-6.), Some(-2.), Some(-3.)],
                [Some(-9.), Some(-4.), Some(-8.)],
            ]
        );
        assert_eq!(
            splits[0].oris_chosen,
            array![[1, 0, 0], [0, 0, 0], [0, 0, 0]].mapv(Some)
        );
        assert_eq!(
            splits[0].allowed,
            array![
                [[false, true], [true, true], [true, true]],
                [[true, true], [true, true], [true, true]],
                [[true, true], [true, true], [true, true]],
            ]
        );
        assert_eq!(splits[0].matching, vec![(1, 0), (2, 0), (0, 0)]);
        assert_eq!(splits[0].log_likelihood, -16.);

        assert_eq!(
            splits[1].cost_matrix_2d,
            array![
                [Some(-8.), Some(-4.), Some(-7.)],
                [Some(-6.), Some(-2.), Some(-10.)],
                [Some(-9.), Some(-4.), Some(-8.)],
            ]
        );
        assert_eq!(
            splits[1].oris_chosen,
            array![[0, 0, 0], [0, 0, 1], [0, 0, 0]].mapv(Some)
        );
        assert_eq!(
            splits[1].allowed,
            array![
                [[true, true], [true, true], [true, true]],
                [[true, true], [true, true], [false, true]],
                [[true, true], [true, true], [true, true]],
            ]
        );
        assert_eq!(splits[1].matching, vec![(2, 0), (0, 0), (1, 0)]);
        assert_eq!(splits[1].log_likelihood, -17.);

        assert_eq!(
            splits[2].cost_matrix_2d,
            array![
                [Some(-8.), Some(-4.), Some(-7.)],
                [Some(-6.), Some(-2.), Some(-3.)],
                [Some(-9.), Some(-10.), Some(-8.)],
            ]
        );
        assert_eq!(
            splits[2].oris_chosen,
            array![[0, 0, 0], [0, 0, 0], [0, 1, 0]].mapv(Some)
        );
        assert_eq!(
            splits[2].allowed,
            array![
                [[true, true], [true, true], [true, true]],
                [[true, true], [true, true], [true, true]],
                [[true, true], [false, true], [true, true]],
            ]
        );
        assert_eq!(splits[2].matching, vec![(1, 0), (2, 0), (0, 0)]);
        assert_eq!(splits[2].log_likelihood, -16.);

        let splits2 = splits[2].split(&cost_matrix_3d).collect_vec();

        assert_eq!(splits2.len(), 3);

        assert_eq!(
            splits2[0].cost_matrix_2d,
            array![
                [Some(-8.), Some(-10.), Some(-7.)],
                [Some(-6.), Some(-2.), Some(-3.)],
                [Some(-9.), Some(-10.), Some(-8.)],
            ]
        );
        assert_eq!(
            splits2[0].oris_chosen,
            array![[0, 1, 0], [0, 0, 0], [0, 1, 0]].mapv(Some)
        );
        assert_eq!(
            splits2[0].allowed,
            array![
                [[true, true], [false, true], [true, true]],
                [[true, true], [true, true], [true, true]],
                [[true, true], [false, true], [true, true]],
            ]
        );
        assert!(
            [vec![(0, 0), (1, 0), (2, 0)], vec![(2, 0), (1, 0), (0, 0)]]
                .contains(&splits2[0].matching)
        );
        assert_eq!(splits2[0].log_likelihood, -18.);

        assert_eq!(
            splits2[1].cost_matrix_2d,
            array![
                [Some(-8.), Some(-4.), Some(-7.)],
                [Some(-6.), Some(-2.), Some(-10.)],
                [Some(-9.), Some(-10.), Some(-8.)],
            ]
        );
        assert_eq!(
            splits2[1].oris_chosen,
            array![[0, 0, 0], [0, 0, 1], [0, 1, 0]].mapv(Some)
        );
        assert_eq!(
            splits2[1].allowed,
            array![
                [[true, true], [true, true], [true, true]],
                [[true, true], [true, true], [false, true]],
                [[true, true], [false, true], [true, true]],
            ]
        );
        assert!(
            [
                vec![(0, 0), (1, 0), (2, 0)],
                vec![(1, 0), (0, 0), (2, 0)],
                vec![(2, 0), (1, 0), (0, 0)]
            ]
            .contains(&splits2[0].matching)
        );
        assert_eq!(splits2[1].log_likelihood, -18.);

        assert_eq!(
            splits2[2].cost_matrix_2d,
            array![
                [Some(-8.), Some(-4.), Some(-7.)],
                [Some(-6.), Some(-2.), Some(-3.)],
                [Some(-10.), Some(-10.), Some(-8.)],
            ]
        );
        assert_eq!(
            splits2[2].oris_chosen,
            array![[0, 0, 0], [0, 0, 0], [1, 1, 0]].mapv(Some)
        );
        assert_eq!(
            splits2[2].allowed,
            array![
                [[true, true], [true, true], [true, true]],
                [[true, true], [true, true], [true, true]],
                [[false, true], [false, true], [true, true]],
            ]
        );
        assert_eq!(splits2[2].matching, vec![(1, 0), (2, 0), (0, 1)]);
        assert_eq!(splits2[2].log_likelihood, -17.);

        let splits3 = splits2[2].split(&cost_matrix_3d).collect_vec();

        assert_eq!(
            splits3[2].cost_matrix_2d,
            array![
                // 8 2 8 18
                // 4 6 8 18
                [Some(-8.), Some(-4.), Some(-7.)],
                [Some(-6.), Some(-2.), Some(-3.)],
                [None, Some(-10.), Some(-8.)],
            ]
        );
        assert_eq!(
            splits3[2].oris_chosen,
            array![
                [Some(0), Some(0), Some(0)],
                [Some(0), Some(0), Some(0)],
                [None, Some(1), Some(0)]
            ]
        );
        assert_eq!(
            splits3[2].allowed,
            array![
                [[true, true], [true, true], [true, true]],
                [[true, true], [true, true], [true, true]],
                [[false, false], [false, true], [true, true]],
            ]
        );
        assert!(
            [vec![(0, 0), (1, 0), (2, 0)], vec![(1, 0), (0, 0), (2, 0)]]
                .contains(&splits3[2].matching)
        );
        assert_eq!(splits3[2].log_likelihood, -18.);
    }

    #[test]
    fn saved_iter() {
        let mut iter = [
            (Permutation::from_cycles(vec![vec![1, 2, 3]]), 1.),
            (Permutation::from_cycles(vec![vec![2, 3]]), 2.),
            (Permutation::from_cycles(vec![vec![1, 2]]), 3.),
        ]
        .into_iter();

        let mut saved_iter = SavedIter {
            saved: Vec::new(),
            iter: iter.by_ref(),
        };

        assert_eq!(
            saved_iter.get(0),
            (&Permutation::from_cycles(vec![vec![1, 2, 3]]), 1.)
        );
        assert_eq!(
            saved_iter.get(1),
            (&Permutation::from_cycles(vec![vec![2, 3]]), 2.)
        );
        assert_eq!(
            saved_iter.get(0),
            (&Permutation::from_cycles(vec![vec![1, 2, 3]]), 1.)
        );
        assert_eq!(
            saved_iter.get(1),
            (&Permutation::from_cycles(vec![vec![2, 3]]), 2.)
        );

        assert_eq!(
            iter.next(),
            Some((Permutation::from_cycles(vec![vec![1, 2]]), 3.))
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn puzzle_iter() {
        let a = [
            (Permutation::from_cycles(vec![vec![0, 1]]), -1.),
            (Permutation::from_cycles(vec![vec![1, 2]]), -3.),
            (Permutation::from_cycles(vec![vec![0, 2]]), -100.),
        ];
        let b = [
            (Permutation::from_cycles(vec![vec![10, 11]]), -2.),
            (Permutation::from_cycles(vec![vec![11, 12]]), -5.),
            (Permutation::from_cycles(vec![vec![10, 12]]), -100.),
        ];

        let mut puzzle_iter = PuzzleIter::new(Box::from([
            SavedIter {
                saved: Vec::new(),
                iter: a.into_iter(),
            },
            SavedIter {
                saved: Vec::new(),
                iter: b.into_iter(),
            },
        ]));

        assert_eq!(
            puzzle_iter.next(),
            Some((
                Permutation::from_cycles(vec![vec![0, 1], vec![10, 11]]),
                -3.
            ))
        );
        assert_eq!(
            puzzle_iter.next(),
            Some((
                Permutation::from_cycles(vec![vec![1, 2], vec![10, 11]]),
                -5.
            ))
        );
        assert_eq!(
            puzzle_iter.next(),
            Some((
                Permutation::from_cycles(vec![vec![0, 1], vec![11, 12]]),
                -6.
            ))
        );
        assert_eq!(
            puzzle_iter.next(),
            Some((
                Permutation::from_cycles(vec![vec![1, 2], vec![11, 12]]),
                -8.
            ))
        );
    }

    // #[test]
    fn solved() {
        let geometry = puzzle("3x3").into_inner();

        let mut baseline = HashMap::new();

        for color in geometry.permutation_group().facelet_colors() {
            baseline.insert(ArcIntern::clone(color), -2.);
        }

        let mut observation = vec![baseline; 48];

        let purduehackers =
            Algorithm::parse_from_string(geometry.permutation_group(), "R2 L2 D2 R2").unwrap();

        for (spot, is) in purduehackers
            .permutation()
            .comes_from()
            .iter_infinite()
            .enumerate()
            .take(48)
        {
            let v = observation[spot]
                .get_mut(&geometry.permutation_group().facelet_colors()[is])
                .unwrap();
            *v += 1.;
        }

        println!("{observation:?}");

        let matcher = Matcher::new(geometry);

        for orbit in &matcher.orbits {
            println!(
                "{:?}",
                orbit
                    .sticker_color_piece
                    .iter()
                    .sorted_by_key(|((a, b), v)| (a.num(), a.orbit(), b, *v))
                    .collect_vec()
            );
        }

        println!(
            "{:?}\n{:?}",
            matcher.most_likely(&observation),
            (purduehackers.permutation().clone(), -48.)
        );

        panic!()
    }
}
