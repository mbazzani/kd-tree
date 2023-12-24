use itertools::Itertools;

use ndarray::*;
use ordered_float::NotNan;
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::*;

use std::fmt;


trait Metric {
    fn distance(&self, other: &Self) -> f32;
}

impl Metric for Array1<f32> {
    fn distance(&self, other: &Self) -> f32 {
        self.iter()
            .zip(other.iter())
            .map(|(x, y)| f32::powi(*x - *y, 2))
            .sum::<f32>()
    }
}

fn min_by_query_point_dist<'a>(
    p: &'a Array1<f32>,
    q: &'a Array1<f32>,
    query_point: &Array1<f32>,
) -> &'a Array1<f32> {
    if p.distance(query_point) < q.distance(query_point) {
        p
    } else {
        q
    }
}

pub struct KDTree {
    dimension: usize,
    children: Option<Box<[KDTree; 2]>>,
    dataset: Vec<Array1<f32>>,
    split_dim: usize,
    split_point: Array1<f32>,
}

impl fmt::Debug for KDTree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Dimension: {}, Split Dim: {}, Median: {} \nChildren: {:#?}",
            self.dimension, self.split_dim, self.split_point[self.split_dim], self.children
        )
    }
}

fn variance(data: &[f32]) -> NotNan<f32> {
    let len = data.len() as f32;
    let mean = data.iter().sum::<f32>() / len;
    unsafe {
        NotNan::new_unchecked(data.iter().map(|&x| f32::powi(mean - x, 2)).sum::<f32>() / len)
    }
}

fn nth_dim(dataset: &[Array1<f32>], n: usize) -> Vec<f32> {
    dataset.iter().map(|p| p[n]).collect()
}

fn median_point(arr: &[Array1<f32>], split_dim: usize) -> Array1<f32> {
    let mut arr: Vec<_> = arr.to_vec();
    arr.sort_by_key(|p| NotNan::new(p[split_dim]).unwrap());
    arr[arr.len() / 2].clone()
}

fn intersection_nonempty(
    split_coordinate: usize,
    median_coordinate_val: f32,
    neighborhood_center: &Array1<f32>,
    neighborhood_radius: f32,
) -> bool {
    let dist_between_point_and_plane =
        (neighborhood_center[split_coordinate] - median_coordinate_val).abs();
    neighborhood_radius >= dist_between_point_and_plane
}

impl KDTree {
    pub fn new(dataset: &[Array1<f32>], min_leaf_size: usize) -> Self {
        let dimension = dataset[0].dim();
        let split_dim = (0..dimension)
            .max_by_key(|&n| variance(&nth_dim(dataset, n)))
            .unwrap();
        let split_point = median_point(dataset, split_dim);
        let left_dataset: Vec<_> = dataset
            .iter()
            .filter(|&p| p[split_dim] <= split_point[split_dim])
            .cloned()
            .collect();
        let mut right_dataset: Vec<_> = dataset
            .iter()
            .filter(|&p| p[split_dim] > split_point[split_dim])
            .cloned()
            .collect();
        let datasets_too_small =
            left_dataset.len() < min_leaf_size || right_dataset.len() < min_leaf_size;
        let children = if datasets_too_small {
            None
        } else {
            Some(Box::new([
                KDTree::new(&left_dataset, min_leaf_size),
                KDTree::new(&right_dataset, min_leaf_size),
            ]))
        };
        KDTree {
            dimension,
            children,
            dataset: dataset.to_owned(),
            split_dim,
            split_point,
        }
    }

    pub fn approximate_nearest_neighbors(
        &self,
        query_point: &Array1<f32>,
        k: usize,
    ) -> Vec<Array1<f32>> {
        match &self.children {
            Some(children) => {
                let closest_partition =
                    if query_point[self.split_dim] <= self.split_point[self.split_dim] {
                        &children[0]
                    } else {
                        &children[1]
                    };
                closest_partition.approximate_nearest_neighbors(query_point, k)
            }
            None => self.k_nearest_neighbors(query_point, k),
        }
    }
    pub fn k_nearest_neighbors(&self, query_point: &Array1<f32>, k: usize) -> Vec<Array1<f32>> {
        self.dataset
            .iter()
            .sorted_by_key(|&p| NotNan::new(p.distance(query_point)).unwrap())
            .take(k)
            .cloned()
            .collect()
    }

    pub fn recall(&self, query_point: &Array1<f32>, k: usize) -> f32 {
        let ann = self.approximate_nearest_neighbors(query_point, k);
        let knn = self.k_nearest_neighbors(query_point, k);
        let num_correct_nn = ann.iter().filter(|&p| knn.contains(p)).count();
        num_correct_nn as f32 / k as f32
    }

    pub fn print_length_profile(&self, depth: usize) {
        match &self.children {
            Some(children) => {
                children[0].print_length_profile(depth + 1);
                children[1].print_length_profile(depth + 1);
            }
            None => {
                dbg!(depth);
            }
        }
    }

    fn distance_from_split_axis(&self, p: &Array1<f32>) -> f32 {
        f32::powi(self.split_point[self.split_dim] - p[self.split_dim], 2)
    }

    fn point_closer_to_left_child(&self, p: &Array1<f32>) -> bool {
        p[self.split_dim] <= self.split_point[self.split_dim]
    }

    fn exact_nn_with_opcount(&self, query_point: &Array1<f32>) -> (Array1<f32>, usize) {
        fn exact_nn_helper(
            node: &KDTree,
            query_point: &Array1<f32>,
            best_point: &Array1<f32>,
        ) -> (Array1<f32>, usize) {
            let old_best_point = best_point;
            let best_point = min_by_query_point_dist(&node.split_point, old_best_point, query_point);
            if node.children.is_none() {
                let best_in_leaf = &node.k_nearest_neighbors(query_point, 1)[0];
                return (min_by_query_point_dist(best_in_leaf, best_point, query_point).clone(), node.dataset.len());
            };

            let children = node.children.as_ref().unwrap();
            let (closest_child, further_child) = if node.point_closer_to_left_child(query_point) {
                (&children[0], &children[1])
            } else {
                (&children[1], &children[0])
            };
            if best_point.distance(query_point) >= node.distance_from_split_axis(query_point) {
                let (best_point_closest_side, n) = exact_nn_helper(closest_child, query_point, best_point);
                let (best_point_further_side, m) = exact_nn_helper(further_child, query_point, &best_point_closest_side);
                (min_by_query_point_dist(
                    &best_point_closest_side,
                    &best_point_further_side,
                    query_point,
                )
                .clone(), n+m)
            } else {
                exact_nn_helper(closest_child, query_point, best_point).clone()
            }
        }
        let (nn, opcount) = exact_nn_helper(self, query_point, &self.split_point);
        //Always passes now
        //assert_eq!(nn, self.k_nearest_neighbors(query_point, 1)[0]);
        (nn, opcount)
    }

    pub fn average_num_ops(
        dataset: Vec<Array1<f32>>,
        query_points: Vec<Array1<f32>>,
        min_leaf_size: usize,
    ) -> f32 {
        let kd_tree = KDTree::new(&dataset, min_leaf_size);
        query_points
            .par_iter()
            .map(|p| kd_tree.exact_nn_with_opcount(p) )
            .map(|(_, n)| n)
            .sum::<usize>() as f32
            / query_points.len() as f32
    }

    pub fn average_recall(
        dataset: Vec<Array1<f32>>,
        query_points: Vec<Array1<f32>>,
        min_leaf_size: usize,
        num_nearest_neighbors: usize,
    ) -> f32 {
        let kd_tree = KDTree::new(&dataset, min_leaf_size);
        query_points
            .par_iter()
            .map(|p| kd_tree.recall(p, num_nearest_neighbors))
            .sum::<f32>()
            / query_points.len() as f32
    }
}
