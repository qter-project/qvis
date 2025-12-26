use std::mem;

use itertools::Itertools;

const E: f64 = 1e-9;

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
pub fn maximum_matching(costs: &[&[Option<f64>]]) -> Option<Vec<usize>> {
    if costs.is_empty() {
        return Some(Vec::new());
    }

    assert!(costs.iter().map(|v| v.len()).all_equal());
    assert_eq!(costs.len(), costs[0].len());

    // Each value is a tuple of `(left potential, right potential, left matches to, right matches to, bfs depth)`
    let mut data: Box<[_]> = Box::from(vec![Element::default(); costs.len()]);

    // We need the reduced cost to be <=0 and we can make that happen in the case of negative costs by setting all of the potentials on the left to the min cost.
    let min_cost = costs.iter().flat_map(|v| v.iter()).filter_map(|v| *v).max_by(|a, b| a.total_cmp(b)).unwrap();

    for elt in &mut data {
        elt.left.potential = min_cost;
    }

    while let Some((i, _)) = data
        .iter()
        .enumerate()
        .find(|(_, elt)| elt.left.matches_with.is_none())
    {
        match find_augmenting_path(i, &mut data, costs) {
            Some(endpoint) => toggle_augmenting_path(endpoint, &mut data),
            None => if !relax_potentials(&mut data, costs) {
                return None;
            },
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
    costs: &[&[Option<f64>]],
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
            for right_idx in 0..costs.len() {
                // Search any nodes on the right that are unvisited and where the reduced cost is zero
                if let Some(cost) = costs[left_idx][right_idx]
                    && !data[right_idx].right.visited
                    && (data[left_idx].left.potential + data[right_idx].right.potential - cost)
                        .abs()
                        < E
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
fn relax_potentials(data: &mut [Element], costs: &[&[Option<f64>]]) -> bool {
    let Some(δ) = costs
        .iter()
        .enumerate()
        .flat_map(|(i, v)| v.iter().enumerate().map(move |(j, v)| ((i, j), v)))
        .filter_map(|(idxs, v)| v.map(|v| (idxs, v)))
        .filter(|((i, j), _)| {
            data[*i].left.visited && !data[*j].right.visited
        })
        .map(|((i, j), c)| data[i].left.potential + data[j].right.potential - c)
        .min_by(|a, b| a.total_cmp(b))
    else {
        return false;
    };

    if δ.abs() < E {
        return false;
    }

    println!("{δ}");

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
    use crate::hungarian_algorithm::maximum_matching;

    #[test]
    fn example() {
        assert_eq!(maximum_matching(&[
            &[Some(-8.), Some(-4.), Some(-7.)],
            &[Some(-6.), Some(-2.), Some(-3.)],
            &[Some(-9.), Some(-4.), Some(-8.)],
        ]), Some(vec![0, 2, 1]));

        assert_eq!(maximum_matching(&[
            &[None, Some(-4.), Some(-7.)],
            &[Some(-6.), Some(-2.), Some(-3.)],
            &[Some(-9.), Some(-4.), Some(-8.)],
        ]), Some(vec![1, 2, 0]));

        assert_eq!(maximum_matching(&[
            &[None, Some(-4.), Some(-7.)],
            &[None, Some(-2.), Some(-3.)],
            &[None, Some(-4.), Some(-8.)],
        ]), None);

        assert_eq!(maximum_matching(&[
            &[Some(100.), Some(110.), Some(90.)],
            &[Some(95.), Some(130.), Some(75.)],
            &[Some(95.), Some(140.), Some(65.)],
        ]), Some(vec![2, 0, 1]));
    }
}
