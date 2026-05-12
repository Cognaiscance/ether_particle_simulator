struct Camera {
    view_proj: mat4x4<f32>,
    // xy: 2*pixel_size/screen_size (NDC offset per quad corner).
    // z: depth_scale in [0,1] — 0 = constant pixel size, 1 = fully perspective-correct (1/depth).
    // w: unused.
    px_size: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

struct VertexInput {
    // Per-vertex: quad corner offset in {-1, +1}.
    @location(0) quad: vec2<f32>,
    // Per-instance: particle world position and color.
    @location(1) world_pos: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) quad: vec2<f32>,
    @location(1) color: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let center_clip = camera.view_proj * vec4<f32>(in.world_pos, 1.0);
    // Blend between constant-pixel-size (multiply by w to cancel perspective divide) and
    // perspective-correct (multiply by 1, so /w stays in NDC and far things shrink).
    let w_factor = mix(center_clip.w, 1.0, camera.px_size.z);
    let offset = vec4<f32>(in.quad * camera.px_size.xy, 0.0, 0.0) * w_factor;
    out.clip = center_clip + offset;
    out.quad = in.quad;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (dot(in.quad, in.quad) > 1.0) {
        discard;
    }
    return vec4<f32>(in.color, 1.0);
}
