use glam::Vec3;

/// Spatial hash grid for broad-phase pair queries.
///
/// Cell size should be at least `2 * radius` so that any colliding pair lies in the same
/// cell or an immediate (3x3x3) neighbor. The table is hashed (not dense) so memory stays
/// O(N) regardless of how spread out the particles are — this matters when the simulation
/// box is much larger than the actual particle cluster.
pub struct HashGrid {
    cell_size: f32,
    inv_cell_size: f32,
    table_size: usize,        // power of 2
    table_mask: u64,          // = table_size - 1
    // After `rebuild`, particles are sorted by hash bucket. `cell_starts` is a CSR offset
    // array; `cell_ids[k]` is the packed 3D cell coord of `indices[k]`, used to filter
    // hash collisions in `for_each_neighbor`.
    cell_starts: Vec<u32>,    // length table_size + 1
    cell_ids: Vec<u64>,       // length N (after rebuild)
    indices: Vec<u32>,        // length N (after rebuild)
}

impl HashGrid {
    pub fn new(cell_size: f32) -> Self {
        let table_size = 16usize;
        Self {
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            table_size,
            table_mask: (table_size - 1) as u64,
            cell_starts: vec![0; table_size + 1],
            cell_ids: Vec::new(),
            indices: Vec::new(),
        }
    }

    pub fn rebuild(&mut self, positions: &[Vec3]) {
        let n = positions.len();

        // Keep load factor ~0.5 so the average bucket holds ~1 cell_id, ~2 particles.
        let target = (n.max(8) * 2).next_power_of_two();
        if target != self.table_size {
            self.table_size = target;
            self.table_mask = (target - 1) as u64;
        }

        if n == 0 {
            self.cell_starts.clear();
            self.cell_starts.resize(self.table_size + 1, 0);
            self.cell_ids.clear();
            self.indices.clear();
            return;
        }

        // Compute (cell_id, particle_index) for each particle.
        let cell_ids_in: Vec<u64> = positions
            .iter()
            .map(|&p| {
                let c = cell_coord_of(p, self.inv_cell_size);
                pack_cell(c)
            })
            .collect();

        // Phase 1: bucket counts (offset by +1 so prefix sum runs in place).
        self.cell_starts.clear();
        self.cell_starts.resize(self.table_size + 1, 0);
        for &cid in &cell_ids_in {
            let b = (hash64(cid) & self.table_mask) as usize;
            self.cell_starts[b + 1] += 1;
        }

        // Phase 2: in-place prefix sum.
        for i in 1..self.cell_starts.len() {
            self.cell_starts[i] += self.cell_starts[i - 1];
        }

        // Phase 3: scatter.
        self.cell_ids.clear();
        self.cell_ids.resize(n, 0);
        self.indices.clear();
        self.indices.resize(n, 0);
        let mut cursors: Vec<u32> = self.cell_starts[..self.table_size].to_vec();
        for (i, &cid) in cell_ids_in.iter().enumerate() {
            let b = (hash64(cid) & self.table_mask) as usize;
            let slot = cursors[b] as usize;
            self.cell_ids[slot] = cid;
            self.indices[slot] = i as u32;
            cursors[b] += 1;
        }
    }

    /// Call `f` once for every particle whose cell lies in the 3x3x3 neighborhood of `p`.
    /// May yield the particle at `p` itself; callers must filter if that matters.
    pub fn for_each_neighbor<F: FnMut(usize)>(&self, p: Vec3, mut f: F) {
        let c = cell_coord_of(p, self.inv_cell_size);
        for oz in -1..=1 {
            for oy in -1..=1 {
                for ox in -1..=1 {
                    let query = [c[0] + ox, c[1] + oy, c[2] + oz];
                    let cid = pack_cell(query);
                    let b = (hash64(cid) & self.table_mask) as usize;
                    let start = self.cell_starts[b] as usize;
                    let end = self.cell_starts[b + 1] as usize;
                    for k in start..end {
                        // Filter hash collisions: the bucket may hold particles from
                        // unrelated cells that happened to hash to the same slot.
                        if self.cell_ids[k] == cid {
                            f(self.indices[k] as usize);
                        }
                    }
                }
            }
        }
    }
}

#[inline]
fn cell_coord_of(p: Vec3, inv_cell_size: f32) -> [i32; 3] {
    // `floor` (not truncation) so negative coords round down consistently.
    [
        (p.x * inv_cell_size).floor() as i32,
        (p.y * inv_cell_size).floor() as i32,
        (p.z * inv_cell_size).floor() as i32,
    ]
}

#[inline]
fn pack_cell(c: [i32; 3]) -> u64 {
    // 21 bits per axis, biased into unsigned range. Supports cell coords in
    // [-2^20, 2^20) — i.e. world coords up to ±(2^20 * cell_size) per axis.
    const BIAS: i32 = 1 << 20;
    const MASK: u64 = (1 << 21) - 1;
    let x = ((c[0] + BIAS) as u64) & MASK;
    let y = ((c[1] + BIAS) as u64) & MASK;
    let z = ((c[2] + BIAS) as u64) & MASK;
    x | (y << 21) | (z << 42)
}

#[inline]
fn hash64(x: u64) -> u64 {
    let h = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^ (h >> 32)
}
