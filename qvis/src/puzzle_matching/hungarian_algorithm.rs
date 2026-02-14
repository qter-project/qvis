use std::mem;

use ndarray::{Array2, ArrayRef2};

#[derive(Default, Clone, Copy, Debug)]
struct Node {
    potential: f64,
    matches_with: Option<usize>,
    bfs_comes_from: Option<usize>,
    visited: bool,
}

/// Allows storing the left and right nodes of the bipartite graph in the same list
#[derive(Default, Clone, Copy, Debug)]
struct Element {
    left: Node,
    right: Node,
}

/// Return a maximum cost matching where the number at index `i` is the index that `i` matches with. The `costs[i][j]` represents the cost of matching `i` with `j`. If the cost is `None`, then we consider matching those two elements to be disallowed. In this case, the function will return `None`.
///
/// <https://timroughgarden.org/w16/l/l5.pdf>
pub fn maximum_matching(costs: &ArrayRef2<Option<f64>>) -> Option<Vec<usize>> {
    assert!(costs.is_square());

    if costs.is_empty() {
        return Some(Vec::new());
    }

    let mut is_tight = Array2::from_shape_fn(costs.raw_dim(), |_| false);

    // Each value is a tuple of `(left potential, right potential, left matches to, right matches to, bfs depth)`
    let mut data: Box<[_]> = Box::from(vec![Element::default(); costs.shape()[0]]);

    // We need the reduced cost to be <=0 and we can make that happen in the case of negative costs by setting all of the potentials on the left to the min cost.
    let min_cost = costs
        .iter()
        .filter_map(|v| *v)
        .max_by(|a, b| a.total_cmp(b))
        .unwrap();

    for elt in &mut data {
        elt.left.potential = min_cost;
    }

    while let Some((i, _)) = data
        .iter()
        .enumerate()
        .find(|(_, elt)| elt.left.matches_with.is_none())
    {
        match find_augmenting_path(i, &mut data, &is_tight, costs) {
            Some(endpoint) => toggle_augmenting_path(endpoint, &mut data),
            None => {
                if !relax_potentials(&mut data, &mut is_tight, costs) {
                    return None;
                }
            }
        }
    }

    Some(
        data.into_iter()
            .map(|elt| elt.left.matches_with.unwrap())
            .collect(),
    )
}

/// Attempt to find an augmenting (good) path that we can use to increase the number of matched nodes by one. If there exists one, then this will return the right index and the information to recover the path is stored in the `bfs_comes_from` fields. Otherwise, the BFS data will still be stored and can be used to relax node prices along the path.
fn find_augmenting_path(
    start_from: usize,
    data: &mut [Element],
    is_tight: &ArrayRef2<bool>,
    costs: &ArrayRef2<Option<f64>>,
) -> Option<usize> {
    // Reset the BFS tracker
    for elt in &mut *data {
        elt.left.bfs_comes_from = None;
        elt.left.visited = false;
        elt.right.bfs_comes_from = None;
        elt.right.visited = false;
    }

    // These are always items on the left side of the bipartite graph
    let mut current_level = vec![start_from];
    data[start_from].left.visited = true;
    let mut next_level = vec![];

    while !current_level.is_empty() {
        for left_idx in current_level.drain(..) {
            for right_idx in 0..costs.shape()[0] {
                // Search any nodes on the right that are unvisited and where the reduced cost is zero
                if let Some(_) = costs[[left_idx, right_idx]]
                    && !data[right_idx].right.visited
                    && is_tight[[left_idx, right_idx]]
                {
                    data[right_idx].right.bfs_comes_from = Some(left_idx);
                    data[right_idx].right.visited = true;

                    match data[right_idx].right.matches_with {
                        Some(new_left_idx) => {
                            // If this is matched with something on the left, then we must search that node in the next layer if it is unvisited
                            if !data[new_left_idx].left.visited {
                                data[new_left_idx].left.bfs_comes_from = Some(right_idx);
                                data[new_left_idx].left.visited = true;
                                next_level.push(new_left_idx);
                            }
                        }
                        None => {
                            // If this node is unmatched, then we have a good path and can quit the search
                            return Some(right_idx);
                        }
                    }
                }
            }
        }

        mem::swap(&mut current_level, &mut next_level);
    }

    None
}

/// Set the matching to the xor of the current matching with the augmenting path
fn toggle_augmenting_path(mut endpoint: usize, data: &mut [Element]) {
    loop {
        let left_side = data[endpoint].right.bfs_comes_from.unwrap();
        data[endpoint].right.matches_with = Some(left_side);
        data[left_side].left.matches_with = Some(endpoint);

        if let Some(next_endpoint) = data[left_side].left.bfs_comes_from {
            endpoint = next_endpoint;
        } else {
            return;
        }
    }
}

/// Relax the potentials along the path to make at least one more edge tight
///
/// Returns whether anything was able to be relaxed
fn relax_potentials(
    data: &mut [Element],
    is_tight: &mut ArrayRef2<bool>,
    costs: &ArrayRef2<Option<f64>>,
) -> bool {
    let Some(((i, j), δ)) = costs
        .indexed_iter()
        .filter_map(|(idxs, v)| v.map(|v| (idxs, v)))
        .filter(|((i, j), _)| data[*i].left.visited && !data[*j].right.visited)
        .map(|((i, j), c)| ((i, j), data[i].left.potential + data[j].right.potential - c))
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
    else {
        return false;
    };

    for i in 0..data.len() {
        for j in 0..data.len() {
            if data[i].left.visited && !data[j].right.visited {
                is_tight[[i, j]] = false;
            }
        }
    }

    is_tight[[i, j]] = true;

    for elt in data {
        if elt.left.visited {
            elt.left.potential -= δ;
        }

        if elt.right.visited {
            elt.right.potential += δ;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::maximum_matching;

    #[test]
    fn example() {
        assert_eq!(
            maximum_matching(&array![
                [Some(-8.), Some(-4.), Some(-7.)],
                [Some(-6.), Some(-2.), Some(-3.)],
                [Some(-9.), Some(-4.), Some(-8.)],
            ]),
            Some(vec![0, 2, 1])
        );

        assert_eq!(
            maximum_matching(&array![
                [None, Some(-4.), Some(-7.)],
                [Some(-6.), Some(-2.), Some(-3.)],
                [Some(-9.), Some(-4.), Some(-8.)],
            ]),
            Some(vec![1, 2, 0])
        );

        assert_eq!(
            maximum_matching(&array![
                [None, Some(-4.), Some(-7.)],
                [None, Some(-2.), Some(-3.)],
                [None, Some(-4.), Some(-8.)],
            ]),
            None
        );

        assert_eq!(
            maximum_matching(&array![
                [Some(100.), Some(110.), Some(90.)],
                [Some(95.), Some(130.), Some(75.)],
                [Some(95.), Some(140.), Some(65.)],
            ]),
            Some(vec![2, 0, 1])
        );
    }

    #[test]
    fn tightness_not_through_epsilon() {
        // This matching leads to the relaxing of potentials not working properly due to floating point rounding error because the precise value of the tightness is never close enough to zero to be considered zero under ε=1e-9. The solution is to keep track of tightness in a separate array.
        assert_eq!(
            maximum_matching(&array![
                [
                    Some(3052265.763914855),
                    Some(3051048.084988203),
                    Some(45.073006316285735),
                    Some(1294345.8137656434),
                    Some(5898072.435256591),
                    Some(3052675.829981847),
                    Some(1552774.9128819676),
                    Some(1552728.4503640207)
                ],
                [
                    Some(1156951.093342854),
                    Some(1.134599964850414),
                    Some(7649154.094641632),
                    Some(555734.4444284381),
                    Some(1157008.9535065796),
                    Some(7649155.83921888),
                    Some(7649157.015021505),
                    Some(60.438339297708175)
                ],
                [
                    Some(5319458.202466325),
                    Some(926169.1991026127),
                    Some(926220.7540678747),
                    Some(4295463.453554934),
                    Some(4295465.153555874),
                    Some(97878.14460299305),
                    Some(704.6096895474138),
                    Some(4295464.157698463)
                ],
                [
                    Some(63461.42078957725),
                    Some(36361925.9918591),
                    Some(47703556.83654001),
                    Some(11278226.089127451),
                    Some(52.97836939994223),
                    Some(36361927.55345198),
                    Some(36361925.568258174),
                    Some(11278278.652790288)
                ],
                [
                    Some(7517468.676308601),
                    Some(7517450.04143544),
                    Some(18214.102036218326),
                    Some(4310.718371037171),
                    Some(51338675.91309436),
                    Some(58874333.48451123),
                    Some(51338675.4767505),
                    Some(51338699.67340185)
                ],
                [
                    Some(1147.390857123671),
                    Some(6201064.561333844),
                    Some(40616550.60643597),
                    Some(40616608.0936402),
                    Some(591904.5930478168),
                    Some(6201064.099499533),
                    Some(47409452.10109716),
                    Some(40617694.52826714)
                ],
                [
                    Some(2676939.97975629),
                    Some(1677575.6585671527),
                    Some(2651885.0775300157),
                    Some(7006362.661739242),
                    Some(2676942.307682288),
                    Some(461.4718209297044),
                    Some(2651920.0537068467),
                    Some(2676938.803695033)
                ],
                [
                    Some(575002.1259626774),
                    Some(92.45961702099193),
                    Some(439769.85429266735),
                    Some(575000.5004559389),
                    Some(8948930.829434488),
                    Some(8949021.402547736),
                    Some(8948930.640305543),
                    Some(9963609.14817566)
                ]
            ]),
            Some(vec![4, 1, 0, 2, 5, 6, 3, 7])
        );
    }
}
