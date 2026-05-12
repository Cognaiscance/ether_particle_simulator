use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use glam::Vec3;
use physics_core::{init_random_uniform_speed, Simulation, SimulationParams};
use serde::Deserialize;

#[derive(Deserialize)]
struct Config {
    particles: ParticlesConfig,
    #[serde(rename = "box")]
    box_: BoxConfig,
    sim: SimConfig,
}

#[derive(Deserialize)]
struct ParticlesConfig {
    count: usize,
    radius: f32,
    mass: f32,
    speed: f32,
}

#[derive(Deserialize)]
struct BoxConfig {
    min: [f32; 3],
    max: [f32; 3],
}

#[derive(Deserialize)]
struct SimConfig {
    dt: f32,
    steps: u64,
    seed: u64,
    report_every: u64,
}

fn main() -> Result<()> {
    let config_path: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("config/example.toml"));

    let raw = std::fs::read_to_string(&config_path)
        .with_context(|| format!("reading config {}", config_path.display()))?;
    let cfg: Config = toml::from_str(&raw)
        .with_context(|| format!("parsing config {}", config_path.display()))?;

    let params = SimulationParams {
        radius: cfg.particles.radius,
        mass: cfg.particles.mass,
        box_min: Vec3::from_array(cfg.box_.min),
        box_max: Vec3::from_array(cfg.box_.max),
    };

    let (positions, velocities) = init_random_uniform_speed(
        cfg.particles.count,
        params,
        cfg.particles.speed,
        cfg.sim.seed,
    )
    .ok_or_else(|| anyhow!("could not place {} particles without overlap (box too small?)", cfg.particles.count))?;

    let mut sim = Simulation::new(params, positions, velocities);

    let e0 = sim.kinetic_energy();
    let p0 = sim.momentum();
    println!(
        "starting: N={} dt={} steps={} E0={:.6} |P0|={:.6e}",
        sim.len(), cfg.sim.dt, cfg.sim.steps, e0, p0.length()
    );
    println!("{:>10} {:>14} {:>14} {:>14}", "step", "energy", "energy_drift", "|momentum|");

    for step in 0..cfg.sim.steps {
        sim.step(cfg.sim.dt);
        if cfg.sim.report_every > 0 && (step + 1) % cfg.sim.report_every == 0 {
            let e = sim.kinetic_energy();
            let p = sim.momentum();
            let drift = if e0 > 0.0 { (e - e0) / e0 } else { 0.0 };
            println!("{:>10} {:>14.6} {:>14.3e} {:>14.3e}", step + 1, e, drift, p.length());
        }
    }

    Ok(())
}
