mod camera;
mod renderer;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use glam::Vec3;
use physics_core::{init_random_uniform_speed, Cannon, Domain, RigidBody, Simulation, SimulationParams};
use serde::Deserialize;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::camera::OrbitCamera;
use crate::renderer::{Instance, Renderer};

#[derive(Deserialize)]
struct Config {
    particles: ParticlesConfig,
    #[serde(rename = "box", default)]
    box_: Option<BoxConfig>,
    #[serde(default)]
    sphere: Option<SphereConfig>,
    sim: SimConfig,
    #[serde(default)]
    view: ViewConfig,
    #[serde(default)]
    bodies: Vec<BodyConfig>,
    #[serde(default)]
    slug: Option<SlugConfig>,
    #[serde(default)]
    cannons: Vec<CannonConfig>,
}

#[derive(Deserialize)]
struct CannonConfig {
    origin: [f32; 3],
    axis: [f32; 3],
    length: f32,
    bore_radius: f32,
    lip_outer_radius: f32,
    piston_stroke: f32,
    piston_period: f32,
}

/// A cylindrical region whose particles are launched with a uniform directed velocity.
/// Used to seed a vortex-cannon-style ejection without needing piston geometry.
#[derive(Deserialize)]
struct SlugConfig {
    center: [f32; 3],
    axis: [f32; 3],
    radius: f32,
    length: f32,
    speed: f32,
}

#[derive(Deserialize)]
struct BodyConfig {
    pos: [f32; 3],
    #[serde(default)]
    vel: [f32; 3],
    radius: f32,
    mass: f32,
    /// Render color for this body. Defaults to a neutral gray so it visually
    /// distinguishes from the warm particle color ramp.
    #[serde(default = "default_body_color")]
    color: [f32; 3],
}

fn default_body_color() -> [f32; 3] { [0.7, 0.75, 0.85] }

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
struct SphereConfig {
    center: [f32; 3],
    radius: f32,
}

#[derive(Deserialize)]
struct SimConfig {
    dt: f32,
    #[allow(dead_code)]
    steps: u64,
    seed: u64,
    #[allow(dead_code)]
    report_every: u64,
}

/// `particles.radius` (in physics-core) controls collision size — the world-space radius used
/// for elastic pair contacts and wall reflection. `view.particle_pixel_size` here controls only
/// the rendered dot size on screen, independent of physics. Rendering smaller than the physics
/// radius gives a "see the centers" look without the dots obstructing each other.
#[derive(Deserialize)]
struct ViewConfig {
    #[serde(default = "default_pixel_size")]
    particle_pixel_size: f32,
    #[serde(default)]
    color_mode: ColorMode,
    #[serde(default = "one")]
    subsample: usize,
    #[serde(default)]
    slice: Option<SliceConfig>,
    /// Color used in `Uniform` mode and as the warm end of `Velocity`.
    #[serde(default = "default_color")]
    color: [f32; 3],
    /// 0 = every particle the same size on screen; 1 = closer particles drawn bigger
    /// (full perspective). Values in between give subtle depth cues.
    #[serde(default = "default_depth_scale")]
    depth_scale: f32,
    /// When true, ignore `particle_pixel_size` and draw each dot as a world-space billboard
    /// matching the physics radius — the rendered size shrinks naturally with distance and
    /// corresponds to the actual collision sphere.
    #[serde(default)]
    render_at_physical_size: bool,
}

impl Default for ViewConfig {
    fn default() -> Self {
        Self {
            particle_pixel_size: default_pixel_size(),
            color_mode: ColorMode::default(),
            subsample: one(),
            slice: None,
            color: default_color(),
            depth_scale: default_depth_scale(),
            render_at_physical_size: false,
        }
    }
}

fn default_pixel_size() -> f32 { 6.0 }
fn one() -> usize { 1 }
fn default_color() -> [f32; 3] { [0.9, 0.6, 0.2] }
fn default_depth_scale() -> f32 { 0.0 }

#[derive(Deserialize, Default, Copy, Clone)]
#[serde(rename_all = "snake_case")]
enum ColorMode {
    #[default]
    Velocity,
    Uniform,
}

#[derive(Deserialize, Copy, Clone)]
struct SliceConfig {
    axis: Axis,
    min: f32,
    max: f32,
}

#[derive(Deserialize, Copy, Clone)]
#[serde(rename_all = "snake_case")]
enum Axis { X, Y, Z }

fn load_config() -> Result<Config> {
    let path: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("config/example.toml"));
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("reading config {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parsing config {}", path.display()))
}

struct App {
    cfg: Config,
    sim: Simulation,
    camera: OrbitCamera,
    renderer: Option<Renderer>,
    paused: bool,
    speed_for_color: f32,
    instances_scratch: Vec<Instance>,
}

impl App {
    fn new(cfg: Config) -> Result<Self> {
        let domain = match (cfg.box_.as_ref(), cfg.sphere.as_ref()) {
            (Some(b), None) => Domain::Box {
                min: Vec3::from_array(b.min),
                max: Vec3::from_array(b.max),
            },
            (None, Some(s)) => Domain::Sphere {
                center: Vec3::from_array(s.center),
                radius: s.radius,
            },
            (Some(_), Some(_)) => return Err(anyhow!("config must specify either [box] or [sphere], not both")),
            (None, None) => return Err(anyhow!("config must specify [box] or [sphere]")),
        };
        let params = SimulationParams {
            radius: cfg.particles.radius,
            mass: cfg.particles.mass,
            domain,
        };
        let (positions, velocities) = init_random_uniform_speed(
            cfg.particles.count,
            params,
            cfg.particles.speed,
            cfg.sim.seed,
        )
        .ok_or_else(|| anyhow!("could not place {} particles without overlap", cfg.particles.count))?;
        let mut sim = Simulation::new(params, positions, velocities);
        for b in &cfg.bodies {
            sim.add_body(RigidBody::new(
                Vec3::from_array(b.pos),
                Vec3::from_array(b.vel),
                b.radius,
                b.mass,
            ));
        }
        for c in &cfg.cannons {
            sim.add_cannon(Cannon::new(
                Vec3::from_array(c.origin),
                Vec3::from_array(c.axis),
                c.length,
                c.bore_radius,
                c.lip_outer_radius,
                c.piston_stroke,
                c.piston_period,
            ));
        }
        if let Some(s) = &cfg.slug {
            let center = Vec3::from_array(s.center);
            let axis_unit = Vec3::from_array(s.axis).normalize_or_zero();
            let half_l = s.length * 0.5;
            let r2 = s.radius * s.radius;
            for (p, v) in sim.positions.iter().zip(sim.velocities.iter_mut()) {
                let rel = *p - center;
                let along = rel.dot(axis_unit);
                if along.abs() > half_l { continue; }
                let radial2 = (rel - axis_unit * along).length_squared();
                if radial2 > r2 { continue; }
                *v = axis_unit * s.speed;
            }
        }
        // Color scale: if a slug is present, normalize by its (much higher) speed so
        // the quiescent medium and the slug span the cool/warm gradient meaningfully.
        let speed_for_color = cfg.slug.as_ref().map(|s| s.speed).unwrap_or(cfg.particles.speed);
        let (aabb_min, aabb_max) = params.domain.aabb();
        let camera = OrbitCamera::looking_at_box(aabb_min, aabb_max);
        Ok(Self {
            speed_for_color,
            cfg,
            sim,
            camera,
            renderer: None,
            paused: false,
            instances_scratch: Vec::new(),
        })
    }

    fn build_instances(&mut self) {
        self.instances_scratch.clear();
        let n = self.sim.len();
        let sub = self.cfg.view.subsample.max(1);
        let slice = self.cfg.view.slice;
        let mode = self.cfg.view.color_mode;
        let warm = self.cfg.view.color;
        // Speed scale for velocity coloring: half/double the initial speed bracket roughly maps 0..1.
        let scale = self.speed_for_color.max(1e-6);
        // Particles render either in screen-pixel mode (`world_radius = 0`) or world-space
        // mode where each dot is a billboard the size of its physics collision sphere.
        let particle_world_radius = if self.cfg.view.render_at_physical_size {
            self.sim.params.radius
        } else {
            0.0
        };

        for i in (0..n).step_by(sub) {
            let p = self.sim.positions[i];
            if let Some(s) = slice {
                let v = match s.axis {
                    Axis::X => p.x,
                    Axis::Y => p.y,
                    Axis::Z => p.z,
                };
                if v < s.min || v > s.max {
                    continue;
                }
            }
            let color = match mode {
                ColorMode::Uniform => warm,
                ColorMode::Velocity => velocity_color(self.sim.velocities[i].length() / scale, warm),
            };
            self.instances_scratch.push(Instance {
                pos: [p.x, p.y, p.z],
                color,
                world_radius: particle_world_radius,
            });
        }

        // Rigid bodies render as world-space billboards regardless of the global flag,
        // since their radius is per-body. Per-body colour comes from config.
        for (body, cfg_body) in self.sim.bodies.iter().zip(self.cfg.bodies.iter()) {
            self.instances_scratch.push(Instance {
                pos: [body.pos.x, body.pos.y, body.pos.z],
                color: cfg_body.color,
                world_radius: body.radius,
            });
        }
    }
}

/// Blue -> warm color ramp. `t` typically in [0, ~2]; clamped.
fn velocity_color(t: f32, warm: [f32; 3]) -> [f32; 3] {
    let t = t.clamp(0.0, 1.5) / 1.5;
    let cool = [0.1, 0.3, 1.0];
    [
        cool[0] + (warm[0] - cool[0]) * t,
        cool[1] + (warm[1] - cool[1]) * t,
        cool[2] + (warm[2] - cool[2]) * t,
    ]
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("physics-viewer")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 800));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("create window"),
        );
        let renderer = pollster::block_on(Renderer::new(
            window.clone(),
            self.cfg.view.particle_pixel_size,
            self.cfg.view.depth_scale,
        ))
        .expect("init renderer");
        self.renderer = Some(renderer);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                if let Some(r) = self.renderer.as_mut() { r.resize(new_size); }
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(code), repeat, .. },
                ..
            } => {
                let rot = 0.05_f32;
                let zoom_in = 0.9_f32;
                let zoom_out = 1.0 / 0.9;
                match code {
                    KeyCode::Escape => event_loop.exit(),
                    KeyCode::Space if !repeat => self.paused = !self.paused,
                    KeyCode::KeyR if !repeat => self.camera.reset(),
                    KeyCode::ArrowLeft  => self.camera.rotate(-rot, 0.0),
                    KeyCode::ArrowRight => self.camera.rotate( rot, 0.0),
                    KeyCode::ArrowUp    => self.camera.rotate(0.0,  rot),
                    KeyCode::ArrowDown  => self.camera.rotate(0.0, -rot),
                    KeyCode::Equal | KeyCode::NumpadAdd      => self.camera.zoom(zoom_in),
                    KeyCode::Minus | KeyCode::NumpadSubtract => self.camera.zoom(zoom_out),
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested => {
                if self.renderer.is_none() {
                    return;
                }
                if !self.paused {
                    self.sim.step(self.cfg.sim.dt);
                }
                self.build_instances();
                let aspect = self.renderer.as_ref().unwrap().aspect();
                let view_proj = self.camera.view_proj(aspect);
                let proj_xy = self.camera.proj_scale(aspect);
                let count = self.instances_scratch.len() as u32;
                let status = match self.sim.bodies.first() {
                    Some(b) => format!("ball |v| = {:.2}", b.vel.length()),
                    None => String::new(),
                };
                let r = self.renderer.as_mut().unwrap();
                r.update_camera(view_proj, proj_xy);
                r.update_instances(&self.instances_scratch);
                r.set_status_text(status);
                match r.render(count) {
                    Ok(()) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = r.window.inner_size();
                        r.resize(size);
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(e) => eprintln!("render error: {e:?}"),
                }
                r.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    let cfg = load_config()?;
    let mut app = App::new(cfg)?;
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut app)?;
    Ok(())
}
