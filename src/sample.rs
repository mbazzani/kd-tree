use ndarray::*;
use ndarray_linalg::Norm;
use rand::thread_rng;
use rand_distr::*;
use std::iter::from_fn;

trait FromDistribution<DistType> {
    type DistType;
    fn from_distribution(dist: impl Distribution<DistType>, num_samples: usize) -> Self;
}

impl FromDistribution<f32> for Array1<f32> {
    type DistType = f32;
    fn from_distribution(distribution: impl Distribution<f32>, len: usize) -> Self {
        let mut rng = thread_rng();
        distribution.sample_iter(&mut rng).take(len).collect()
    }
}

pub fn from_sphere(dimension: usize, num_samples: usize, radius: f32) -> Vec<Array1<f32>> {
    let standard_normal = Normal::new(0., 1.).unwrap();
    from_fn(|| Some(Array1::from_distribution(standard_normal, dimension)))
        .map(|p| &p * radius / p.norm_l2())
        .take(num_samples)
        .collect()
}

pub fn from_spherical_guassian(
    dimension: usize,
    num_samples: usize,
    center: Array1<f32>,
) -> Vec<Array1<f32>> {
    let standard_normal = Normal::new(0., 1.).unwrap();
    from_fn(|| Some(&center + Array1::from_distribution(standard_normal, dimension)))
        .take(num_samples)
        .collect()
}

pub fn from_uniform_box(dimension: usize, num_samples: usize) -> Vec<Array1<f32>> {
    let uniform = Uniform::new(-1e4, 1e4);
    from_fn(|| Some(Array1::from_distribution(uniform, dimension)))
        .take(num_samples)
        .collect()
}

pub fn from_number_line(dimension: usize, num_samples: usize) -> Vec<Array1<f32>> {
   let uniform = Uniform::new(-1e4, 1e4);
    let mut rng = thread_rng();
    from_fn(|| {
        Some(Array1::from_shape_fn(dimension, |n| {
            if n == 0 {
                uniform.sample(&mut rng)
            } else {
                0.
            }
        }))
    })
    .take(num_samples)
    .collect()
}
pub fn from_gaussian_mixture(
    dimension: usize,
    radius: f32,
    num_gaussians: usize,
    num_samples: usize,
) -> Vec<Array1<f32>> {
    let centers = from_sphere(dimension, num_gaussians, radius);
    let samples_per_gaussian = num_samples / num_gaussians;
    centers
        .into_iter()
        .flat_map(|center| from_spherical_guassian(dimension, samples_per_gaussian, center))
        .collect()
}

pub fn from_cross(dimension: usize, num_samples: usize) -> Vec<Array1<f32>> {
    let mut rng = thread_rng();
    let coordinate_distribution = Uniform::new(0, dimension);
    let point_distribution = Uniform::new(-0.1, 0.1);
    from_fn(|| {
        let mut point = Array1::from_shape_fn(dimension, |_| point_distribution.sample(&mut rng));
        let stretch_direction = coordinate_distribution.sample(&mut rng);
        point[stretch_direction] *= 1e5f32;
        Some(point)
    })
    .take(num_samples)
    .collect()
}
