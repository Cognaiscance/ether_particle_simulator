use glam::{Mat4, Vec3};

pub struct OrbitCamera {
    pub target: Vec3,
    pub radius: f32,
    pub yaw: f32,   // around +Y, radians
    pub pitch: f32, // up from XZ plane, radians
    pub fov_y: f32, // radians
    pub znear: f32,
    pub zfar: f32,
    initial_radius: f32,
    initial_yaw: f32,
    initial_pitch: f32,
}

impl OrbitCamera {
    pub fn looking_at_box(box_min: Vec3, box_max: Vec3) -> Self {
        let center = 0.5 * (box_min + box_max);
        let extent = (box_max - box_min).length();
        let radius = extent * 1.4;
        let yaw = std::f32::consts::FRAC_PI_4;
        let pitch = std::f32::consts::FRAC_PI_6;
        Self {
            target: center,
            radius,
            yaw,
            pitch,
            fov_y: 60_f32.to_radians(),
            znear: 0.05,
            zfar: extent * 20.0,
            initial_radius: radius,
            initial_yaw: yaw,
            initial_pitch: pitch,
        }
    }

    pub fn reset(&mut self) {
        self.radius = self.initial_radius;
        self.yaw = self.initial_yaw;
        self.pitch = self.initial_pitch;
    }

    pub fn rotate(&mut self, d_yaw: f32, d_pitch: f32) {
        self.yaw += d_yaw;
        self.pitch = (self.pitch + d_pitch)
            .clamp(-std::f32::consts::FRAC_PI_2 + 0.05, std::f32::consts::FRAC_PI_2 - 0.05);
    }

    pub fn zoom(&mut self, factor: f32) {
        self.radius = (self.radius * factor).max(self.znear * 2.0);
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let dir = Vec3::new(cp * sy, sp, cp * cy);
        let eye = self.target + dir * self.radius;
        let view = Mat4::look_at_rh(eye, self.target, Vec3::Y);
        let proj = Mat4::perspective_rh(self.fov_y, aspect, self.znear, self.zfar);
        proj * view
    }
}
