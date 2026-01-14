use bytemuck::{Pod, Zeroable};
use cgmath::{InnerSpace, Point3};
use std::mem;
use std::sync::Arc;
use wgpu::util::DeviceExt;

use winit::{
    application::ApplicationHandler, event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent}, event_loop::{ActiveEventLoop, EventLoop}, keyboard::{self, NamedKey}, window::{Window, WindowId}
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

fn vertex(pos: [i8; 3], color: [f32; 3]) -> Vertex {
    Vertex {
        position: [pos[0] as f32, pos[1] as f32, pos[2] as f32],
        color,
    }
}

fn create_cube_vertices() -> Vec<Vertex> {
    let r = [1.0, 0.0, 0.0];
    let g = [0.0, 1.0, 0.0];
    let b = [0.0, 0.0, 1.0];
    let y = [1.0, 1.0, 0.0];
    let c = [0.0, 1.0, 1.0];
    let m = [1.0, 0.0, 1.0];

    vec![
        // front
        vertex([-1, -1, 1], r),
        vertex([1, -1, 1], r),
        vertex([1, 1, 1], r),
        vertex([-1, 1, 1], r),
        // back
        vertex([-1, -1, -1], g),
        vertex([1, -1, -1], g),
        vertex([1, 1, -1], g),
        vertex([-1, 1, -1], g),
        // right
        vertex([1, -1, -1], b),
        vertex([1, -1, 1], b),
        vertex([1, 1, 1], b),
        vertex([1, 1, -1], b),
        // left
        vertex([-1, -1, -1], y),
        vertex([-1, -1, 1], y),
        vertex([-1, 1, 1], y),
        vertex([-1, 1, -1], y),
        // top
        vertex([-1, 1, -1], c),
        vertex([1, 1, -1], c),
        vertex([1, 1, 1], c),
        vertex([-1, 1, 1], c),
        // bottom
        vertex([-1, -1, -1], m),
        vertex([1, -1, -1], m),
        vertex([1, -1, 1], m),
        vertex([-1, -1, 1], m),
    ]
}

fn cube_indices() -> Vec<u16> {
    let mut indices = Vec::new();
    for i in 0..6 {
        let start = i * 4;
        indices.extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
    }
    indices
}

fn create_axis_geometry() -> (Vec<Vertex>, Vec<Vertex>) {
    let stem_len = 80.0;
    let arrow_len = 20.0;
    let arrow_radius = 8.0;

    let mut line_vertices = Vec::new();
    let mut arrow_vertices = Vec::new();

    let add_axis = |line_vertices: &mut Vec<Vertex>,
                    arrow_vertices: &mut Vec<Vertex>,
                    dir: [f32; 3],
                    color: [f32; 3]| {
        // Line segment (stem)
        line_vertices.push(Vertex {
            position: [0.0, 0.0, 0.0],
            color,
        });
        line_vertices.push(Vertex {
            position: [dir[0] * stem_len, dir[1] * stem_len, dir[2] * stem_len],
            color,
        });

        // Arrow tip
        let tip = [
            dir[0] * (stem_len + arrow_len),
            dir[1] * (stem_len + arrow_len),
            dir[2] * (stem_len + arrow_len),
        ];
        let base = [dir[0] * stem_len, dir[1] * stem_len, dir[2] * stem_len];

        // Find perpendicular vectors
        let fallback_up = if dir[0].abs() < 0.9 && dir[1].abs() < 0.9 {
            [0.0, 0.0, 1.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let dir_vec = cgmath::Vector3::new(dir[0], dir[1], dir[2]);
        let up_vec = cgmath::Vector3::new(fallback_up[0], fallback_up[1], fallback_up[2]);
        let right = dir_vec.cross(up_vec).normalize();
        let up = dir_vec.cross(right).normalize();

        let p1 = [
            base[0] + right.x * arrow_radius,
            base[1] + right.y * arrow_radius,
            base[2] + right.z * arrow_radius,
        ];
        let p2 = [
            base[0] - right.x * arrow_radius,
            base[1] - right.y * arrow_radius,
            base[2] - right.z * arrow_radius,
        ];
        let p3 = [
            base[0] + up.x * arrow_radius,
            base[1] + up.y * arrow_radius,
            base[2] + up.z * arrow_radius,
        ];
        let p4 = [
            base[0] - up.x * arrow_radius,
            base[1] - up.y * arrow_radius,
            base[2] - up.z * arrow_radius,
        ];

        // 4 triangles for arrowhead
        arrow_vertices.extend_from_slice(&[
            Vertex {
                position: tip,
                color,
            },
            Vertex {
                position: p1,
                color,
            },
            Vertex {
                position: p3,
                color,
            },
            Vertex {
                position: tip,
                color,
            },
            Vertex {
                position: p3,
                color,
            },
            Vertex {
                position: p2,
                color,
            },
            Vertex {
                position: tip,
                color,
            },
            Vertex {
                position: p2,
                color,
            },
            Vertex {
                position: p4,
                color,
            },
            Vertex {
                position: tip,
                color,
            },
            Vertex {
                position: p4,
                color,
            },
            Vertex {
                position: p1,
                color,
            },
        ]);
    };

    add_axis(
        &mut line_vertices,
        &mut arrow_vertices,
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ); // X
    add_axis(
        &mut line_vertices,
        &mut arrow_vertices,
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ); // Y
    add_axis(
        &mut line_vertices,
        &mut arrow_vertices,
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ); // Z

    (line_vertices, arrow_vertices)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline_triangle: wgpu::RenderPipeline,
    render_pipeline_line: wgpu::RenderPipeline,
    cube_vertex_buf: wgpu::Buffer,
    cube_index_buf: wgpu::Buffer,
    cube_index_count: u32,
    axis_line_vertex_buf: wgpu::Buffer,
    axis_line_vertex_count: u32,
    axis_arrow_vertex_buf: wgpu::Buffer,
    axis_arrow_vertex_count: u32,
    uniform_buf_scene: wgpu::Buffer,
    uniform_buf_overlay: wgpu::Buffer,
    window: Arc<Window>,
    rotation_x: f32,
    rotation_y: f32,
    scale: f32,
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = unsafe { instance.create_surface(&window).unwrap() };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let uniform_buf_scene = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scene Uniform Buffer"),
            size: mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_buf_overlay = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Overlay Uniform Buffer"),
            size: mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            label: Some("bind_group_layout"),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline_triangle =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Triangle Render Pipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                    }],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(config.format.into())],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

        let render_pipeline_line = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Line Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(config.format.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let cube_vertices = create_cube_vertices();
        let cube_indices = cube_indices();
        let cube_vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cube Vertex Buffer"),
            contents: bytemuck::cast_slice(&cube_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let cube_index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cube Index Buffer"),
            contents: bytemuck::cast_slice(&cube_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let cube_index_count = cube_indices.len() as u32;

        let (axis_line_vertices, axis_arrow_vertices) = create_axis_geometry();
        let axis_line_vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Axis Line Buffer"),
            contents: bytemuck::cast_slice(&axis_line_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let axis_arrow_vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Axis Arrow Buffer"),
            contents: bytemuck::cast_slice(&axis_arrow_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline_triangle,
            render_pipeline_line,
            cube_vertex_buf,
            cube_index_buf,
            cube_index_count,
            axis_line_vertex_buf,
            axis_line_vertex_count: axis_line_vertices.len() as u32,
            axis_arrow_vertex_buf,
            axis_arrow_vertex_count: axis_arrow_vertices.len() as u32,
            uniform_buf_scene,
            uniform_buf_overlay,
            window,
            rotation_x: 0.0,
            rotation_y: 0.0,
            scale: 5.0,
            mouse_pressed: false,
            last_mouse_pos: None,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn update_scene_uniforms(&mut self) {
        use cgmath::{Deg, Matrix4, Vector3, perspective};
        let aspect = self.config.width as f32 / self.config.height as f32;
        let proj = perspective(Deg(60.0), aspect, 0.1, 100.0);
        let view = Matrix4::<f32>::look_at_rh(
            Point3::<f32>::new(0.0, 0.0, self.scale),
            Point3::<f32>::new(0.0, 0.0, 0.0),
            Vector3::<f32>::unit_y(),
        );
        let rot_y = Matrix4::from_angle_y(cgmath::Rad(self.rotation_y));
        let rot_x = Matrix4::from_angle_x(cgmath::Rad(self.rotation_x));
        let model = rot_y * rot_x;
        let view_proj = proj * view * model;

        self.queue.write_buffer(
            &self.uniform_buf_scene,
            0,
            bytemuck::cast_slice(&[view_proj.into()]),
        );
    }

    fn update_overlay_uniforms(&mut self) {
        use cgmath::{Matrix4, ortho};
        let width = self.config.width as f32;
        let height = self.config.height as f32;
        let margin = 100.0;
        let left = width - margin;
        let right = width;
        let bottom = 0.0;
        let top = margin;
        let proj = ortho(left, right, bottom, top, -1.0, 1.0);
        let scale = Matrix4::from_nonuniform_scale(margin, margin, 1.0);
        let translate = Matrix4::from_translation(cgmath::Vector3::new(left, bottom, 0.0));
        let model = translate * scale;
        let view_proj = proj * model;

        self.queue.write_buffer(
            &self.uniform_buf_overlay,
            0,
            bytemuck::cast_slice(&[view_proj.into()]),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.update_scene_uniforms();
        self.update_overlay_uniforms();

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Main scene pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            let scene_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.render_pipeline_triangle.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buf_scene.as_entire_binding(),
                }],
                label: None,
            });

            render_pass.set_pipeline(&self.render_pipeline_triangle);
            render_pass.set_bind_group(0, &scene_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.cube_vertex_buf.slice(..));
            render_pass.set_index_buffer(self.cube_index_buf.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.cube_index_count, 0, 0..1);
        }

        // Overlay pass (axes)
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Overlay Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            let overlay_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.render_pipeline_triangle.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buf_overlay.as_entire_binding(),
                }],
                label: None,
            });

            // Draw arrowheads (triangles)
            render_pass.set_pipeline(&self.render_pipeline_triangle);
            render_pass.set_bind_group(0, &overlay_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.axis_arrow_vertex_buf.slice(..));
            render_pass.draw(0..self.axis_arrow_vertex_count, 0..1);

            // Draw stems (lines)
            render_pass.set_pipeline(&self.render_pipeline_line);
            render_pass.set_bind_group(0, &overlay_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.axis_line_vertex_buf.slice(..));
            render_pass.draw(0..self.axis_line_vertex_count, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }

    fn mouse_drag(&mut self, x: f64, y: f64) {
        if let Some((last_x, last_y)) = self.last_mouse_pos {
            let dx = (x - last_x) as f32 * 0.01;
            let dy = (y - last_y) as f32 * 0.01;
            self.rotation_y += dx;
            self.rotation_x += dy;
            self.rotation_x = self.rotation_x.clamp(-1.4, 1.4);
        }
        self.last_mouse_pos = Some((x, y));
    }

    fn handle_scroll(&mut self, delta: MouseScrollDelta) {
        let delta = match delta {
            MouseScrollDelta::LineDelta(_, y) => y * 20.0,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
        };
        self.scale = (self.scale - delta * 0.1).max(1.0).min(20.0);
    }
}
