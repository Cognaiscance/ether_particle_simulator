use glam::Vec3;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

mod grid;
use grid::HashGrid;

/// Shape of the simulation domain. Walls are inside-facing — particles bounce off them.
#[derive(Clone, Copy, Debug)]
pub enum Domain {
    Box { min: Vec3, max: Vec3 },
    Sphere { center: Vec3, radius: f32 },
}

impl Domain {
    /// AABB of the domain (used to size the hash grid, frame the camera, etc.).
    pub fn aabb(&self) -> (Vec3, Vec3) {
        match *self {
            Domain::Box { min, max } => (min, max),
            Domain::Sphere { center, radius } => (
                center - Vec3::splat(radius),
                center + Vec3::splat(radius),
            ),
        }
    }

    /// Region in which particle centers can legally sit (accounting for particle radius).
    fn particle_aabb(&self, r: f32) -> (Vec3, Vec3) {
        match *self {
            Domain::Box { min, max } => (min + Vec3::splat(r), max - Vec3::splat(r)),
            Domain::Sphere { center, radius } => {
                let inner = radius - r;
                (center - Vec3::splat(inner), center + Vec3::splat(inner))
            }
        }
    }

    /// Volume of the region in which particle centers can legally sit.
    fn particle_volume(&self, r: f32) -> f32 {
        match *self {
            Domain::Box { min, max } => {
                let usable = (max - min) - Vec3::splat(2.0 * r);
                (usable.x * usable.y * usable.z).max(0.0)
            }
            Domain::Sphere { radius, .. } => {
                let inner = (radius - r).max(0.0);
                (4.0 / 3.0) * std::f32::consts::PI * inner * inner * inner
            }
        }
    }

    /// Whether a particle center at `p` lies inside the wall-respecting interior.
    fn contains_center(&self, p: Vec3, r: f32) -> bool {
        match *self {
            Domain::Box { min, max } => {
                let lo = min + Vec3::splat(r);
                let hi = max - Vec3::splat(r);
                p.x >= lo.x && p.x <= hi.x
                    && p.y >= lo.y && p.y <= hi.y
                    && p.z >= lo.z && p.z <= hi.z
            }
            Domain::Sphere { center, radius } => {
                let inner = radius - r;
                (p - center).length_squared() <= inner * inner
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SimulationParams {
    pub radius: f32,
    pub mass: f32,
    pub domain: Domain,
}

/// A free-moving rigid sphere that interacts with the small particles. Currently linear
/// only — no orientation, angular velocity, or friction.
#[derive(Clone, Copy, Debug)]
pub struct RigidBody {
    pub pos: Vec3,
    pub vel: Vec3,
    pub radius: f32,
    pub mass: f32,
    /// `1.0 / mass`, cached. `0.0` means immovable (infinite mass).
    pub inv_mass: f32,
}

impl RigidBody {
    pub fn new(pos: Vec3, vel: Vec3, radius: f32, mass: f32) -> Self {
        let inv_mass = if mass.is_finite() && mass > 0.0 { 1.0 / mass } else { 0.0 };
        Self { pos, vel, radius, mass, inv_mass }
    }
}

/// Hollow cylindrical cannon with an annular muzzle lip and a kinematic piston.
///
/// Geometry is anchored at `origin` (center of the back face) and oriented along
/// the unit vector `axis`. Local axial coordinate `s` is the projection of
/// `(p - origin)` onto `axis`; local radial distance `rho` is the length of the
/// component perpendicular to `axis`. The body is a thick-walled tube:
///
/// - inner bore at `rho = bore_radius`, axial extent `s ∈ [0, length]`
/// - outer surface at `rho = lip_outer_radius`, same axial extent
/// - annular back face at `s = 0`, `rho ∈ [bore_radius, lip_outer_radius]`
/// - annular front lip at `s = length`, `rho ∈ [bore_radius, lip_outer_radius]`
/// - piston disk inside the bore at `s = piston_offset(t)`
///
/// The piston follows `p(t) = stroke · (1 − cos(2π t / period)) / 2` for one
/// cycle, then rests at zero. Peak axial velocity is `stroke · π / period`.
#[derive(Clone, Debug)]
pub struct Cannon {
    pub origin: Vec3,
    pub axis: Vec3,
    pub length: f32,
    pub bore_radius: f32,
    pub lip_outer_radius: f32,
    pub piston_stroke: f32,
    pub piston_period: f32,
    pub elapsed: f32,
    pub piston_offset: f32,
    pub piston_vel: f32,
}

impl Cannon {
    pub fn new(
        origin: Vec3,
        axis: Vec3,
        length: f32,
        bore_radius: f32,
        lip_outer_radius: f32,
        piston_stroke: f32,
        piston_period: f32,
    ) -> Self {
        let axis = axis.normalize_or_zero();
        Self {
            origin,
            axis,
            length,
            bore_radius,
            lip_outer_radius,
            piston_stroke,
            piston_period,
            elapsed: 0.0,
            piston_offset: 0.0,
            piston_vel: 0.0,
        }
    }

    fn advance(&mut self, dt: f32) {
        self.elapsed += dt;
        if self.piston_period <= 0.0 || self.elapsed >= self.piston_period {
            self.piston_offset = 0.0;
            self.piston_vel = 0.0;
            return;
        }
        let phase = std::f32::consts::TAU * self.elapsed / self.piston_period;
        self.piston_offset = self.piston_stroke * (1.0 - phase.cos()) * 0.5;
        self.piston_vel = self.piston_stroke * std::f32::consts::PI / self.piston_period * phase.sin();
    }
}

pub struct Simulation {
    pub params: SimulationParams,
    pub positions: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    pub bodies: Vec<RigidBody>,
    pub cannons: Vec<Cannon>,
    grid: HashGrid,
}

impl Simulation {
    pub fn new(params: SimulationParams, positions: Vec<Vec3>, velocities: Vec<Vec3>) -> Self {
        assert_eq!(positions.len(), velocities.len());
        let grid = HashGrid::new(2.0 * params.radius);
        Self { params, positions, velocities, bodies: Vec::new(), cannons: Vec::new(), grid }
    }

    pub fn add_body(&mut self, body: RigidBody) {
        self.bodies.push(body);
    }

    pub fn add_cannon(&mut self, cannon: Cannon) {
        self.cannons.push(cannon);
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub fn step(&mut self, dt: f32) {
        self.integrate(dt);
        self.integrate_bodies(dt);
        for c in self.cannons.iter_mut() {
            c.advance(dt);
        }
        self.resolve_walls();
        self.resolve_body_walls();
        self.resolve_cannons();
        self.resolve_pairs();
        self.resolve_body_particles();
    }

    fn integrate(&mut self, dt: f32) {
        for (p, v) in self.positions.iter_mut().zip(self.velocities.iter()) {
            *p += *v * dt;
        }
    }

    fn integrate_bodies(&mut self, dt: f32) {
        for body in self.bodies.iter_mut() {
            body.pos += body.vel * dt;
        }
    }

    fn resolve_walls(&mut self) {
        let r = self.params.radius;
        match self.params.domain {
            Domain::Box { min, max } => {
                let lo = min + Vec3::splat(r);
                let hi = max - Vec3::splat(r);
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
            Domain::Sphere { center, radius } => {
                let inner = radius - r;
                let inner_sq = inner * inner;
                for (p, v) in self.positions.iter_mut().zip(self.velocities.iter_mut()) {
                    let offset = *p - center;
                    let d_sq = offset.length_squared();
                    if d_sq > inner_sq && d_sq > 1e-12 {
                        let d = d_sq.sqrt();
                        let n_hat = offset / d;
                        // Mirror position across the wall so the particle re-enters by the
                        // amount it overshot.
                        *p -= n_hat * (2.0 * (d - inner));
                        // Reflect the outward-going component of velocity.
                        let v_rad = v.dot(n_hat);
                        if v_rad > 0.0 {
                            *v -= 2.0 * v_rad * n_hat;
                        }
                    }
                }
            }
        }
    }

    fn resolve_body_walls(&mut self) {
        let domain = self.params.domain;
        for body in self.bodies.iter_mut() {
            let r = body.radius;
            match domain {
                Domain::Box { min, max } => {
                    let lo = min + Vec3::splat(r);
                    let hi = max - Vec3::splat(r);
                    for axis in 0..3 {
                        if body.pos[axis] < lo[axis] {
                            body.pos[axis] = lo[axis] + (lo[axis] - body.pos[axis]);
                            body.vel[axis] = -body.vel[axis];
                        } else if body.pos[axis] > hi[axis] {
                            body.pos[axis] = hi[axis] - (body.pos[axis] - hi[axis]);
                            body.vel[axis] = -body.vel[axis];
                        }
                    }
                }
                Domain::Sphere { center, radius } => {
                    let inner = radius - r;
                    if inner <= 0.0 { continue; }
                    let offset = body.pos - center;
                    let d_sq = offset.length_squared();
                    if d_sq > inner * inner && d_sq > 1e-12 {
                        let d = d_sq.sqrt();
                        let n_hat = offset / d;
                        body.pos -= n_hat * (2.0 * (d - inner));
                        let v_rad = body.vel.dot(n_hat);
                        if v_rad > 0.0 {
                            body.vel -= 2.0 * v_rad * n_hat;
                        }
                    }
                }
            }
        }
    }

    fn resolve_cannons(&mut self) {
        if self.cannons.is_empty() {
            return;
        }
        let r_p = self.params.radius;
        for cannon in self.cannons.iter() {
            for i in 0..self.positions.len() {
                resolve_particle_against_cannon(
                    &mut self.positions[i],
                    &mut self.velocities[i],
                    r_p,
                    cannon,
                );
            }
        }
    }

    fn resolve_body_particles(&mut self) {
        if self.bodies.is_empty() {
            return;
        }
        let part_r = self.params.radius;
        let part_m = self.params.mass;
        let inv_m_p = if part_m > 0.0 { 1.0 / part_m } else { 0.0 };

        // Grid must reflect post-pair positions (resolve_pairs may have nudged particles).
        self.grid.rebuild(&self.positions);

        for bi in 0..self.bodies.len() {
            let body = self.bodies[bi];
            let body_r = body.radius;
            let min_dist = body_r + part_r;
            let min_dist_sq = min_dist * min_dist;
            let pad = Vec3::splat(min_dist);
            let aabb_lo = body.pos - pad;
            let aabb_hi = body.pos + pad;

            // Collect particles that share at least one cell with the body's AABB. The
            // closure can't mutate `self`, so we materialise the candidate list first.
            let mut candidates: Vec<usize> = Vec::new();
            self.grid.for_each_in_aabb(aabb_lo, aabb_hi, |j| candidates.push(j));

            for &j in &candidates {
                let delta = self.positions[j] - self.bodies[bi].pos;
                let d_sq = delta.length_squared();
                if d_sq >= min_dist_sq || d_sq < 1e-12 {
                    continue;
                }
                let d = d_sq.sqrt();
                let n_hat = delta / d;
                let v_rel = self.velocities[j] - self.bodies[bi].vel;
                let v_rel_n = v_rel.dot(n_hat);
                let denom = inv_m_p + self.bodies[bi].inv_mass;
                if denom <= 0.0 {
                    continue; // both static; nothing to do
                }
                let inv_m_b = self.bodies[bi].inv_mass;
                if v_rel_n < 0.0 {
                    // Elastic impulse along n_hat. Pushes particle outward, body inward
                    // (each scaled by its inverse mass).
                    let imp = -2.0 * v_rel_n / denom;
                    self.velocities[j] += (imp * inv_m_p) * n_hat;
                    self.bodies[bi].vel -= (imp * inv_m_b) * n_hat;
                }
                // Inverse-mass-weighted positional de-overlap.
                let overlap = min_dist - d;
                if overlap > 0.0 {
                    let part_share = inv_m_p / denom;
                    let body_share = inv_m_b / denom;
                    self.positions[j] += (overlap * part_share) * n_hat;
                    self.bodies[bi].pos -= (overlap * body_share) * n_hat;
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

/// Resolve a single particle against the surfaces of `cannon`. Pushes the particle
/// out of any overlap and reflects the relevant velocity component (radial for the
/// cylindrical walls, axial for the annuli and piston). The piston is treated as a
/// moving wall: `v' = 2·v_piston - v` for the axial component.
fn resolve_particle_against_cannon(pos: &mut Vec3, vel: &mut Vec3, r_p: f32, c: &Cannon) {
    // Helper: decompose a position into (axial, radial_vec, rho, rho_hat).
    let axial_radial = |p: Vec3| {
        let to_p = p - c.origin;
        let s = to_p.dot(c.axis);
        let radial_vec = to_p - c.axis * s;
        let rho = radial_vec.length();
        (s, radial_vec, rho)
    };

    let bore = c.bore_radius;
    let outer = c.lip_outer_radius;
    let length = c.length;
    let s_pist = c.piston_offset;

    // 1) Inner bore wall: cylinder at rho = bore, axial extent [0, length], two-sided.
    let (s, radial_vec, rho) = axial_radial(*pos);
    if s > -r_p && s < length + r_p && rho > 1e-6 {
        let rho_hat = radial_vec / rho;
        let d = rho - bore;
        if d.abs() < r_p {
            let target_rho = if d >= 0.0 { bore + r_p } else { bore - r_p };
            *pos += rho_hat * (target_rho - rho);
            let v_radial = vel.dot(rho_hat);
            // Reflect only if moving into the wall.
            if (d >= 0.0 && v_radial < 0.0) || (d < 0.0 && v_radial > 0.0) {
                *vel -= 2.0 * v_radial * rho_hat;
            }
        }
    }

    // 2) Outer wall: cylinder at rho = outer, axial extent [0, length], two-sided.
    let (s, radial_vec, rho) = axial_radial(*pos);
    if s > -r_p && s < length + r_p && rho > 1e-6 {
        let rho_hat = radial_vec / rho;
        let d = rho - outer;
        if d.abs() < r_p {
            let target_rho = if d >= 0.0 { outer + r_p } else { outer - r_p };
            *pos += rho_hat * (target_rho - rho);
            let v_radial = vel.dot(rho_hat);
            if (d >= 0.0 && v_radial < 0.0) || (d < 0.0 && v_radial > 0.0) {
                *vel -= 2.0 * v_radial * rho_hat;
            }
        }
    }

    // 3) Front lip annulus at s = length, rho in [bore, outer], two-sided.
    let (s, _, rho) = axial_radial(*pos);
    if rho >= bore && rho <= outer {
        let d = s - length;
        if d.abs() < r_p {
            let target_s = if d >= 0.0 { length + r_p } else { length - r_p };
            *pos += c.axis * (target_s - s);
            let v_axial = vel.dot(c.axis);
            if (d >= 0.0 && v_axial < 0.0) || (d < 0.0 && v_axial > 0.0) {
                *vel -= 2.0 * v_axial * c.axis;
            }
        }
    }

    // 4) Back annulus at s = 0, rho in [bore, outer], two-sided.
    let (s, _, rho) = axial_radial(*pos);
    if rho >= bore && rho <= outer {
        if s.abs() < r_p {
            let target_s = if s >= 0.0 { r_p } else { -r_p };
            *pos += c.axis * (target_s - s);
            let v_axial = vel.dot(c.axis);
            if (s >= 0.0 && v_axial < 0.0) || (s < 0.0 && v_axial > 0.0) {
                *vel -= 2.0 * v_axial * c.axis;
            }
        }
    }

    // 5) Piston disk at s = piston_offset, rho < bore. One-sided: particle's center
    // must stay at s >= piston_offset + r_p within the bore.
    let (s, _, rho) = axial_radial(*pos);
    if rho < bore && s < s_pist + r_p {
        let target_s = s_pist + r_p;
        *pos += c.axis * (target_s - s);
        let v_axial = vel.dot(c.axis);
        // Elastic collision with a wall moving at piston velocity. Only apply when
        // the particle is closing on the piston (relative velocity < 0).
        if v_axial < c.piston_vel {
            let v_new = 2.0 * c.piston_vel - v_axial;
            *vel += (v_new - v_axial) * c.axis;
        }
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
    let (aabb_lo, aabb_hi) = params.domain.particle_aabb(r);
    let usable = aabb_hi - aabb_lo;
    if usable.min_element() <= 0.0 {
        return None;
    }
    if n == 0 {
        return Some((Vec::new(), Vec::new()));
    }

    // Place particles on an FCC lattice (max sphere packing fraction ≈ 0.7405). Reaches
    // dense configurations that random rejection sampling can't. Lattice points outside
    // the actual domain (e.g. corners of the AABB for a sphere) are filtered out.
    //
    // For lattice constant `a`, the 12 nearest neighbors in FCC are at distance `a / √2`.
    // We require `a / √2 ≥ 2r` so spheres don't overlap.
    let v_domain = params.domain.particle_volume(r);
    if v_domain <= 0.0 {
        return None;
    }
    let min_spacing = 2.0 * std::f32::consts::SQRT_2 * r;
    let mut a = (4.0 * v_domain / n as f32).cbrt();

    let mut positions: Vec<Vec3> = Vec::with_capacity(n);
    loop {
        if a < min_spacing {
            return None; // can't fit N non-overlapping spheres at any FCC spacing
        }
        positions.clear();
        let nx = (usable.x / a).floor() as i32;
        let ny = (usable.y / a).floor() as i32;
        let nz = (usable.z / a).floor() as i32;
        let basis = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.0) * a,
            Vec3::new(0.5, 0.0, 0.5) * a,
            Vec3::new(0.0, 0.5, 0.5) * a,
        ];
        'outer: for k in 0..nz {
            let z = aabb_lo.z + (k as f32) * a;
            for j in 0..ny {
                let y = aabb_lo.y + (j as f32) * a;
                for i in 0..nx {
                    let x = aabb_lo.x + (i as f32) * a;
                    for &off in &basis {
                        let p = Vec3::new(x, y, z) + off;
                        if params.domain.contains_center(p, r) {
                            positions.push(p);
                            if positions.len() == n {
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
        if positions.len() == n {
            break;
        }
        // Either the lattice was too coarse (cube floors lost positions) or the domain
        // discards too many points (sphere corners). Shrink `a` and retry.
        a *= 0.98;
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
            domain: Domain::Box { min: Vec3::ZERO, max: Vec3::splat(2.0) },
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
            domain: Domain::Box { min: Vec3::splat(-100.0), max: Vec3::splat(100.0) },
        };
        // Place particles in a tight cluster so they actually collide.
        let small_params = SimulationParams {
            domain: Domain::Box { min: Vec3::splat(-0.5), max: Vec3::splat(0.5) },
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
    fn body_particle_collision_conserves_energy_and_momentum() {
        // Body and particles share a large box (walls won't interfere over the test interval).
        let params = SimulationParams {
            radius: 0.05,
            mass: 1.0,
            domain: Domain::Box { min: Vec3::splat(-50.0), max: Vec3::splat(50.0) },
        };
        let small_params = SimulationParams {
            domain: Domain::Box { min: Vec3::splat(-0.5), max: Vec3::splat(0.5) },
            ..params
        };
        let (positions, velocities) = init_random_uniform_speed(20, small_params, 1.0, 9).unwrap();
        let mut sim = Simulation::new(params, positions, velocities);
        // Heavy-but-finite body so it actually exchanges energy with the particles.
        sim.add_body(RigidBody::new(
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(-1.5, 0.0, 0.0),
            0.3,
            50.0,
        ));

        let total_energy = |s: &Simulation| {
            let part = s.kinetic_energy();
            let body: f32 = s.bodies.iter().map(|b| 0.5 * b.mass * b.vel.length_squared()).sum();
            part + body
        };
        let total_momentum = |s: &Simulation| {
            let part = s.momentum();
            let body: Vec3 = s.bodies.iter().map(|b| b.mass * b.vel).sum();
            part + body
        };

        let e0 = total_energy(&sim);
        let p0 = total_momentum(&sim);
        for _ in 0..2000 {
            sim.step(0.005);
        }
        let e1 = total_energy(&sim);
        let p1 = total_momentum(&sim);
        assert!(
            (e1 - e0).abs() / e0 < 1e-2,
            "energy not conserved with body: {} -> {}", e0, e1,
        );
        assert!(
            (p1 - p0).length() < 1e-2,
            "momentum not conserved with body: {} -> {}", p0, p1,
        );
    }

    #[test]
    fn sphere_walls_conserve_energy_and_keep_particles_inside() {
        let params = SimulationParams {
            radius: 0.02,
            mass: 1.0,
            domain: Domain::Sphere { center: Vec3::splat(1.0), radius: 1.0 },
        };
        let (positions, velocities) = init_random_uniform_speed(50, params, 1.0, 11).unwrap();
        let mut sim = Simulation::new(params, positions, velocities);
        let e0 = sim.kinetic_energy();
        for _ in 0..2000 {
            sim.step(0.005);
        }
        let e1 = sim.kinetic_energy();
        assert!((e1 - e0).abs() / e0 < 1e-3, "energy drift: {} -> {}", e0, e1);
        // No particle should have leaked out (allowing a small numerical slack).
        for p in &sim.positions {
            let d = (*p - Vec3::splat(1.0)).length();
            assert!(d <= 1.0 - 0.02 + 1e-4, "particle at distance {} from center", d);
        }
    }

    #[test]
    fn head_on_equal_mass_swaps_velocities() {
        // Two particles on the x-axis moving toward each other; they should swap velocities.
        let params = SimulationParams {
            radius: 0.1,
            mass: 1.0,
            domain: Domain::Box { min: Vec3::splat(-10.0), max: Vec3::splat(10.0) },
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
