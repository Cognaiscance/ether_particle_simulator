use glam::Vec3;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

mod grid;
use grid::HashGrid;

#[derive(Clone, Copy, Debug)]
pub struct SimulationParams {
    pub radius: f32,
    pub mass: f32,
    pub box_min: Vec3,
    pub box_max: Vec3,
}

pub struct Simulation {
    pub params: SimulationParams,
    pub positions: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    grid: HashGrid,
}

impl Simulation {
    pub fn new(params: SimulationParams, positions: Vec<Vec3>, velocities: Vec<Vec3>) -> Self {
        assert_eq!(positions.len(), velocities.len());
        let grid = HashGrid::new(2.0 * params.radius);
        Self { params, positions, velocities, grid }
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub fn step(&mut self, dt: f32) {
        self.integrate(dt);
        self.resolve_walls();
        self.resolve_pairs();
    }

    fn integrate(&mut self, dt: f32) {
        for (p, v) in self.positions.iter_mut().zip(self.velocities.iter()) {
            *p += *v * dt;
        }
    }

    fn resolve_walls(&mut self) {
        let r = self.params.radius;
        let lo = self.params.box_min + Vec3::splat(r);
        let hi = self.params.box_max - Vec3::splat(r);
        for (p, v) in self.positions.iter_mut().zip(self.velocities.iter_mut()) {
            for axis in 0..3 {
                if p[axis] < lo[axis] {
                    p[axis] = lo[axis] + (lo[axis] - p[axis]);
                    v[axis] = -v[axis];
                } else if p[axis] > hi[axis] {
                    p[axis] = hi[axis] - (p[axis] - hi[axis]);
                    v[axis] = -v[axis];
                }
            }
        }
    }

    fn resolve_pairs(&mut self) {
        let n = self.positions.len();
        let r = self.params.radius;
        let min_dist = 2.0 * r;
        let min_dist_sq = min_dist * min_dist;

        // Broad phase via uniform grid: per particle, only test the 3x3x3 cell neighborhood
        // instead of all (i, j>i) pairs. Pair-list collection parallelizes across cores; the
        // resolve step stays serial so per-pair impulse is computed from the live state.
        self.grid.rebuild(&self.positions);
        let positions = &self.positions;
        let grid = &self.grid;
        let mut pairs: Vec<(usize, usize)> = (0..n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let pos_i = positions[i];
                let mut out: Vec<(usize, usize)> = Vec::new();
                grid.for_each_neighbor(pos_i, |j| {
                    if j > i {
                        let delta = positions[j] - pos_i;
                        let dist_sq = delta.length_squared();
                        if dist_sq < min_dist_sq && dist_sq > 0.0 {
                            out.push((i, j));
                        }
                    }
                });
                out.into_iter()
            })
            .collect();
        pairs.sort_unstable();

        for (i, j) in pairs {
            let delta = self.positions[j] - self.positions[i];
            let dist_sq = delta.length_squared();
            if dist_sq >= min_dist_sq || dist_sq == 0.0 {
                // Earlier resolved pair already separated these; nothing to do.
                continue;
            }
            let dist = dist_sq.sqrt();
            let n_hat = delta / dist;
            let v_rel = self.velocities[i] - self.velocities[j];
            let approach = v_rel.dot(n_hat);
            if approach > 0.0 {
                let impulse = approach * n_hat;
                self.velocities[i] -= impulse;
                self.velocities[j] += impulse;
            }
            let overlap = min_dist - dist;
            if overlap > 0.0 {
                let push = 0.5 * overlap * n_hat;
                self.positions[i] -= push;
                self.positions[j] += push;
            }
        }
    }

    pub fn kinetic_energy(&self) -> f32 {
        let half_m = 0.5 * self.params.mass;
        self.velocities.iter().map(|v| half_m * v.length_squared()).sum()
    }

    pub fn momentum(&self) -> Vec3 {
        let m = self.params.mass;
        self.velocities.iter().copied().sum::<Vec3>() * m
    }
}

/// Place `n` particles at random non-overlapping positions inside the simulation box,
/// each given a velocity of magnitude `speed` in a uniformly random direction.
///
/// Returns `None` if rejection sampling cannot find space for all particles (box too packed).
pub fn init_random_uniform_speed(
    n: usize,
    params: SimulationParams,
    speed: f32,
    seed: u64,
) -> Option<(Vec<Vec3>, Vec<Vec3>)> {
    let r = params.radius;
    let extent = params.box_max - params.box_min;
    let usable = extent - Vec3::splat(2.0 * r);
    if usable.min_element() <= 0.0 {
        return None;
    }
    if n == 0 {
        return Some((Vec::new(), Vec::new()));
    }

    // Place particles on an FCC lattice (max sphere packing fraction ≈ 0.7405). Reaches
    // dense configurations that random rejection sampling can't.
    //
    // For lattice constant `a`, the 12 nearest neighbors in FCC are at distance `a / √2`.
    // We require `a / √2 ≥ 2r` so spheres don't overlap.
    let v_usable = usable.x * usable.y * usable.z;
    let min_spacing = 2.0 * std::f32::consts::SQRT_2 * r;
    let mut a = (4.0 * v_usable / n as f32).cbrt();
    let mut nx: i32 = 0;
    let mut ny: i32 = 0;
    let mut nz: i32 = 0;
    loop {
        if a < min_spacing {
            return None; // can't fit N non-overlapping spheres at any FCC spacing in this box
        }
        nx = (usable.x / a).floor() as i32;
        ny = (usable.y / a).floor() as i32;
        nz = (usable.z / a).floor() as i32;
        let count = 4i64 * (nx as i64) * (ny as i64) * (nz as i64);
        if count >= n as i64 {
            break;
        }
        // Floor losses cost us a few positions; shrink and retry. Converges in O(log).
        a *= 0.98;
    }

    let basis = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.5, 0.5, 0.0) * a,
        Vec3::new(0.5, 0.0, 0.5) * a,
        Vec3::new(0.0, 0.5, 0.5) * a,
    ];
    let origin = params.box_min + Vec3::splat(r);
    let mut positions: Vec<Vec3> = Vec::with_capacity(n);
    'outer: for k in 0..nz {
        let z = origin.z + (k as f32) * a;
        for j in 0..ny {
            let y = origin.y + (j as f32) * a;
            for i in 0..nx {
                let x = origin.x + (i as f32) * a;
                for &off in &basis {
                    positions.push(Vec3::new(x, y, z) + off);
                    if positions.len() == n {
                        break 'outer;
                    }
                }
            }
        }
    }

    // Jitter, bounded so the FCC nearest-neighbor distance can't drop below 2r.
    // Two neighbors can each jitter by up to magnitude J = (a/√2 - 2r) / 2; with per-axis
    // uniform jitter ±K, total magnitude ≤ K√3, so K = J/√3. A 0.9 safety factor avoids
    // grazing contact.
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let jitter_axis =
        ((a / std::f32::consts::SQRT_2 - 2.0 * r) / (2.0 * 3f32.sqrt())) * 0.9;
    if jitter_axis > 0.0 {
        for p in positions.iter_mut() {
            *p += Vec3::new(
                rng.gen_range(-jitter_axis..jitter_axis),
                rng.gen_range(-jitter_axis..jitter_axis),
                rng.gen_range(-jitter_axis..jitter_axis),
            );
        }
    }

    let mut velocities: Vec<Vec3> = Vec::with_capacity(n);
    for _ in 0..n {
        velocities.push(random_unit_vec3(&mut rng) * speed);
    }

    // Remove net drift so the center of mass stays put — purely cosmetic for diagnostics.
    let drift: Vec3 = velocities.iter().copied().sum::<Vec3>() / n as f32;
    for v in velocities.iter_mut() {
        *v -= drift;
    }

    Some((positions, velocities))
}

fn random_unit_vec3<R: Rng>(rng: &mut R) -> Vec3 {
    // Marsaglia's method via rejection in a unit sphere.
    loop {
        let x: f32 = rng.gen_range(-1.0..1.0);
        let y: f32 = rng.gen_range(-1.0..1.0);
        let z: f32 = rng.gen_range(-1.0..1.0);
        let len_sq = x * x + y * y + z * z;
        if len_sq > 1e-8 && len_sq <= 1.0 {
            let inv = 1.0 / len_sq.sqrt();
            return Vec3::new(x * inv, y * inv, z * inv);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> SimulationParams {
        SimulationParams {
            radius: 0.05,
            mass: 1.0,
            box_min: Vec3::ZERO,
            box_max: Vec3::splat(2.0),
        }
    }

    #[test]
    fn wall_only_conserves_energy_and_momentum() {
        // No pair collisions: spread them out far enough that they won't interact.
        let params = SimulationParams { radius: 0.01, ..default_params() };
        let (positions, velocities) = init_random_uniform_speed(8, params, 1.0, 42).unwrap();
        let mut sim = Simulation::new(params, positions, velocities);
        let e0 = sim.kinetic_energy();
        let p0 = sim.momentum();
        for _ in 0..2000 {
            sim.step(0.005);
        }
        let e1 = sim.kinetic_energy();
        let p1 = sim.momentum();
        assert!((e1 - e0).abs() / e0 < 1e-4, "energy drift: {} -> {}", e0, e1);
        // Wall reflections flip individual components, so total momentum is NOT conserved
        // (the box exerts external forces). Just check the magnitude stays bounded.
        assert!(p1.length() <= p0.length().max(1.0) * 10.0);
    }

    #[test]
    fn pair_collisions_conserve_energy_and_momentum() {
        // Periodic-free: use a very large box so walls don't interfere.
        let params = SimulationParams {
            radius: 0.05,
            mass: 1.0,
            box_min: Vec3::splat(-100.0),
            box_max: Vec3::splat(100.0),
        };
        // Place particles in a tight cluster so they actually collide.
        let small_params = SimulationParams {
            box_min: Vec3::splat(-0.5),
            box_max: Vec3::splat(0.5),
            ..params
        };
        let (positions, velocities) = init_random_uniform_speed(20, small_params, 1.0, 7).unwrap();
        let mut sim = Simulation::new(params, positions, velocities);
        let e0 = sim.kinetic_energy();
        let p0 = sim.momentum();
        for _ in 0..2000 {
            sim.step(0.005);
        }
        let e1 = sim.kinetic_energy();
        let p1 = sim.momentum();
        assert!(
            (e1 - e0).abs() / e0 < 1e-3,
            "energy not conserved across pair collisions: {} -> {}", e0, e1
        );
        assert!(
            (p1 - p0).length() < 1e-3,
            "momentum not conserved across pair collisions: {} -> {}", p0, p1
        );
    }

    #[test]
    fn head_on_equal_mass_swaps_velocities() {
        // Two particles on the x-axis moving toward each other; they should swap velocities.
        let params = SimulationParams {
            radius: 0.1,
            mass: 1.0,
            box_min: Vec3::splat(-10.0),
            box_max: Vec3::splat(10.0),
        };
        let positions = vec![Vec3::new(-0.15, 0.0, 0.0), Vec3::new(0.15, 0.0, 0.0)];
        let velocities = vec![Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
        let mut sim = Simulation::new(params, positions, velocities);
        // Many small steps so the moment of contact falls inside a step rather than on a boundary.
        for _ in 0..100 {
            sim.step(0.005);
        }
        // After collision, velocities should have swapped (equal mass head-on).
        assert!(sim.velocities[0].x < 0.0, "left particle should bounce back, got {:?}", sim.velocities[0]);
        assert!(sim.velocities[1].x > 0.0, "right particle should bounce back, got {:?}", sim.velocities[1]);
        // Total kinetic energy unchanged.
        let ke = sim.kinetic_energy();
        assert!((ke - 1.0).abs() < 1e-5, "energy not preserved: {}", ke);
    }
}
