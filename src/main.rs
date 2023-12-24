#![allow(dead_code)]
use ndarray::*;
mod sample;
use sample::*;

mod kd_tree;
use kd_tree::KDTree;

struct Benchmark {
    dimension: usize,
    dataset_size: usize,
    data_source: Box<dyn Fn(usize) -> Vec<Array1<f32>>>,
    num_query_points: usize,
    num_nearest_neighbors: usize,
    min_leaf_size: usize,
    num_test_iterations: usize,
}

impl Benchmark {
    fn run_benchmark(
        &mut self,
        update_fn: fn(&mut Self),
        get_independent_var: fn(&Self) -> f32,
    ) -> Vec<(f32, f32, f32)> {
        let mut data = vec![];
        while self.num_test_iterations > 0 {
            let recall = KDTree::average_recall(
                (self.data_source)(self.dataset_size),
                (self.data_source)(self.num_query_points),
                self.min_leaf_size,
                self.num_nearest_neighbors,
            );
            let num_ops = KDTree::average_num_ops(
                (self.data_source)(self.dataset_size),
                (self.data_source)(self.num_query_points),
                self.min_leaf_size,
            );
            let independent_var = get_independent_var(self);
            data.push((independent_var, recall, num_ops));
            self.num_test_iterations -= 1;
            update_fn(self);
        }
        data
    }

    fn run_with_standard_params(
        data_source: Box<dyn Fn(usize) -> Vec<Array1<f32>>>,
        update_fn: fn(&mut Benchmark),
        independent_var_fn: fn(&Benchmark) -> f32,
        data_source_description: &str,
        independent_var_name: &str,
    ) {
        let dimension = 1;
        let dataset_size = 100000;
        let num_query_points = 200;
        let num_nearest_neighbors = 5;
        let min_leaf_size = 100;
        let num_test_iterations = 50;

        let mut bench = Benchmark {
            dimension,
            dataset_size,
            data_source,
            num_query_points,
            num_nearest_neighbors,
            min_leaf_size,
            num_test_iterations,
        };

        println!("Data Source: {data_source_description}");
        println!("Left entry: {independent_var_name}");
        println!("Middle entry: Average Recall");
        println!("Right entry: Average Number of Point Comparisons for Exact 1-NN");
        for (x, y, z) in bench.run_benchmark(update_fn, independent_var_fn) {
            print!("{:.0}\t", x);
            print!("{:.3},\t", y);
            println!("{:.3}", z);
        }
    }
}

fn main() {
    let dimension = 1;
    let dataset_size = 100000;
    let num_query_points = 200;
    let num_nearest_neighbors = 5;
    let min_leaf_size = 100;
    let num_test_iterations = 50;
    println!("Starting dimension: {dimension}");
    println!("Dataset Size: {dataset_size}");
    println!("Number of Query Points: {num_query_points}");
    println!("Number of Nearest Neighbors: {num_nearest_neighbors}");
    println!("Min Leaf Size: {min_leaf_size}");
    println!("Num Test Iterations: {num_test_iterations}\n\n");

    let starting_dimension: usize = 5;
    let independent_var_fn = |b: &Benchmark| b.dimension as f32;
    let independent_var_name = "Dimension";

    let data_source_name = "Uniform Distribution on Cell";
    let data_source = Box::new(move |n| from_uniform_box(starting_dimension, n));
    let update_fn = |benchmark: &mut Benchmark| {
        benchmark.dimension += 1;
        let d = benchmark.dimension;
        benchmark.data_source = Box::new(move |n| from_uniform_box(d, n));
    };
    Benchmark::run_with_standard_params(
        data_source,
        update_fn,
        independent_var_fn,
        data_source_name,
        independent_var_name,
    );

//    let data_source_name = "Uniform on Unit Sphere";
//    let data_source = Box::new(move |n| from_sphere(starting_dimension, n, 1.));
//    let update_fn = |benchmark: &mut Benchmark| {
//        benchmark.dimension += 1;
//        let d = benchmark.dimension;
//        benchmark.data_source = Box::new(move |n| from_sphere(d, n, 1.));
//    };
//    Benchmark::run_with_standard_params(
//        data_source,
//        update_fn,
//        independent_var_fn,
//        data_source_name,
//        independent_var_name,
//    );
//
//    let data_source_name = "5 Guassian Mixture with R=1";
//    let data_source = Box::new(move |n| from_gaussian_mixture(starting_dimension, 1., 5, n));
//    let update_fn = |benchmark: &mut Benchmark| {
//        benchmark.dimension += 1;
//        let d = benchmark.dimension;
//        benchmark.data_source = Box::new(move |n| from_gaussian_mixture(d, 1., 5, n));
//    };
//    Benchmark::run_with_standard_params(
//        data_source,
//        update_fn,
//        independent_var_fn,
//        data_source_name,
//        independent_var_name,
//    );
//
//    let data_source_name = "Cross";
//    let data_source = Box::new(move |n| from_cross(starting_dimension, n));
//    let update_fn = |benchmark: &mut Benchmark| {
//        benchmark.dimension += 1;
//        let d = benchmark.dimension;
//        benchmark.data_source = Box::new(move |n| from_cross(d, n));
//    };
//    Benchmark::run_with_standard_params(
//        data_source,
//        update_fn,
//        independent_var_fn,
//        data_source_name,
//        independent_var_name,
//    );
//
//    let data_source_name = "Number Line";
//    let data_source = Box::new(move |n| from_number_line(starting_dimension, n));
//    let update_fn = |benchmark: &mut Benchmark| {
//        benchmark.dimension += 1;
//        let d = benchmark.dimension;
//        benchmark.data_source = Box::new(move |n| from_number_line(d, n));
//    };
//    Benchmark::run_with_standard_params(
//        data_source,
//        update_fn,
//        independent_var_fn,
//        data_source_name,
//        independent_var_name,
//    );
}
