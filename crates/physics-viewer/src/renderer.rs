use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use glyphon::{
    Attrs, Buffer as GlyphonBuffer, Cache as GlyphonCache, Color as GlyphonColor, Family,
    FontSystem, Metrics, Resolution, Shaping, SwashCache, TextArea, TextAtlas, TextBounds,
    TextRenderer, Viewport,
};
use wgpu::util::DeviceExt;
use winit::window::Window;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Instance {
    pub pos: [f32; 3],
    pub color: [f32; 3],
    /// World-space radius. When `> 0`, the dot is drawn as a perspective-correct
    /// billboard of that radius. When `0`, falls back to camera-uniform pixel sizing.
    pub world_radius: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    // xy: 2*pixel_size/screen_size, z: depth_scale, w: unused.
    px_size: [f32; 4],
    // xy: proj[0][0], proj[1][1] — used to size per-instance world-space billboards.
    proj_xy: [f32; 4],
}

const QUAD_VERTICES: [[f32; 2]; 4] = [
    [-1.0, -1.0],
    [ 1.0, -1.0],
    [-1.0,  1.0],
    [ 1.0,  1.0],
];

pub struct Renderer {
    pub window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    pipeline: wgpu::RenderPipeline,
    quad_vbuf: wgpu::Buffer,
    instance_vbuf: wgpu::Buffer,
    instance_capacity: usize,
    camera_buf: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    depth_view: wgpu::TextureView,
    particle_pixel_size: f32,
    depth_scale: f32,
    text: TextOverlay,
}

struct TextOverlay {
    font_system: FontSystem,
    swash_cache: SwashCache,
    atlas: TextAtlas,
    viewport: Viewport,
    text_renderer: TextRenderer,
    buffer: GlyphonBuffer,
    label: String,
}

impl Renderer {
    pub async fn new(
        window: Arc<Window>,
        particle_pixel_size: f32,
        depth_scale: f32,
    ) -> Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance
            .create_surface(window.clone())
            .context("create surface")?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("request adapter")?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .context("request device")?;

        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera-bg"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline-layout"),
            bind_group_layouts: &[&camera_bgl],
            push_constant_ranges: &[],
        });

        let quad_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 2]>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            }],
        };
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Instance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 24,
                    shader_location: 3,
                },
            ],
        };

        let depth_format = wgpu::TextureFormat::Depth32Float;
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particle-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[quad_layout, instance_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let quad_vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad-verts"),
            contents: bytemuck::cast_slice(&QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let instance_capacity = 1024;
        let instance_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instances"),
            size: (instance_capacity * std::mem::size_of::<Instance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let depth_view = create_depth_view(&device, size, depth_format);

        let text = TextOverlay::new(&device, &queue, format, size);

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            pipeline,
            quad_vbuf,
            instance_vbuf,
            instance_capacity,
            camera_buf,
            camera_bind_group,
            depth_view,
            particle_pixel_size,
            depth_scale: depth_scale.clamp(0.0, 1.0),
            text,
        })
    }

    pub fn set_status_text(&mut self, text: impl Into<String>) {
        self.text.label = text.into();
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = create_depth_view(&self.device, new_size, wgpu::TextureFormat::Depth32Float);
    }

    pub fn aspect(&self) -> f32 {
        self.size.width as f32 / self.size.height.max(1) as f32
    }

    pub fn update_camera(&self, view_proj: glam::Mat4, proj_xy: [f32; 2]) {
        // Pixel mode size (NDC offset per quad corner). Per-instance world_radius > 0 lets
        // an individual dot opt into world-space sizing via the proj_xy uniform instead.
        let px = self.particle_pixel_size;
        let px_xy = [
            2.0 * px / self.size.width.max(1) as f32,
            2.0 * px / self.size.height.max(1) as f32,
        ];
        let uniform = CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            px_size: [px_xy[0], px_xy[1], self.depth_scale, 0.0],
            proj_xy: [proj_xy[0], proj_xy[1], 0.0, 0.0],
        };
        self.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&uniform));
    }

    pub fn update_instances(&mut self, instances: &[Instance]) {
        if instances.is_empty() {
            return;
        }
        if instances.len() > self.instance_capacity {
            let new_cap = instances.len().next_power_of_two().max(1024);
            self.instance_vbuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instances"),
                size: (new_cap * std::mem::size_of::<Instance>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_capacity = new_cap;
        }
        self.queue
            .write_buffer(&self.instance_vbuf, 0, bytemuck::cast_slice(instances));
    }

    pub fn render(&mut self, instance_count: u32) -> Result<(), wgpu::SurfaceError> {
        self.text.prepare(&self.device, &self.queue, self.size);

        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("encoder") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("particle-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.05, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            if instance_count > 0 {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_vertex_buffer(0, self.quad_vbuf.slice(..));
                pass.set_vertex_buffer(1, self.instance_vbuf.slice(..));
                pass.draw(0..4, 0..instance_count);
            }
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("text-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            self.text.render(&mut pass);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }
}

impl TextOverlay {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        size: winit::dpi::PhysicalSize<u32>,
    ) -> Self {
        let mut font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let cache = GlyphonCache::new(device);
        let mut viewport = Viewport::new(device, &cache);
        viewport.update(
            queue,
            Resolution { width: size.width.max(1), height: size.height.max(1) },
        );
        let mut atlas = TextAtlas::new(device, queue, &cache, format);
        let text_renderer = TextRenderer::new(
            &mut atlas,
            device,
            wgpu::MultisampleState::default(),
            None,
        );
        let mut buffer = GlyphonBuffer::new(&mut font_system, Metrics::new(20.0, 24.0));
        buffer.set_size(
            &mut font_system,
            Some(size.width.max(1) as f32),
            Some(40.0),
        );
        Self {
            font_system,
            swash_cache,
            atlas,
            viewport,
            text_renderer,
            buffer,
            label: String::new(),
        }
    }

    fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: winit::dpi::PhysicalSize<u32>,
    ) {
        let w = size.width.max(1);
        let h = size.height.max(1);
        self.viewport.update(queue, Resolution { width: w, height: h });
        self.buffer
            .set_size(&mut self.font_system, Some(w as f32), Some(40.0));
        self.buffer.set_text(
            &mut self.font_system,
            &self.label,
            Attrs::new().family(Family::SansSerif),
            Shaping::Advanced,
        );
        self.buffer.shape_until_scroll(&mut self.font_system, false);
        let top = (h as f32) - 30.0;
        let area = TextArea {
            buffer: &self.buffer,
            left: 12.0,
            top,
            scale: 1.0,
            bounds: TextBounds { left: 0, top: 0, right: w as i32, bottom: h as i32 },
            default_color: GlyphonColor::rgb(220, 220, 220),
            custom_glyphs: &[],
        };
        let _ = self.text_renderer.prepare(
            device,
            queue,
            &mut self.font_system,
            &mut self.atlas,
            &self.viewport,
            [area],
            &mut self.swash_cache,
        );
    }

    fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        let _ = self.text_renderer.render(&self.atlas, &self.viewport, pass);
    }
}

fn create_depth_view(
    device: &wgpu::Device,
    size: winit::dpi::PhysicalSize<u32>,
    format: wgpu::TextureFormat,
) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d {
            width: size.width.max(1),
            height: size.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}
