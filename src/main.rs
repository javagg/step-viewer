use bevy::{
    app::AppExit,
    asset::RenderAssetUsages,
    camera::{Viewport, visibility::RenderLayers},
    log::LogPlugin,
    prelude::MessageWriter,
    prelude::*,
    render::render_resource::PrimitiveTopology,
    window::{PresentMode, WindowTheme},
    winit::WinitSettings,
};
use bevy_egui::{
    EguiContexts, EguiGlobalSettings, EguiPlugin, EguiPrimaryContextPass, PrimaryEguiContext, egui,
};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use std::sync::Mutex;
use std::sync::mpsc::Receiver;
use step_viewer::{
    LoadMessage, Parameter, StepMetadata, StepScene, StepShell, load_step_file_streaming,
};
use truck_meshalgo::prelude::PolygonMesh;

#[derive(Clone, Copy, PartialEq, Eq)]
enum AxesMode {
    World,
    ScreenTriad,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SelectionMode {
    Point,
    Edge,
    Face,
}

#[derive(Resource)]
struct ViewerState {
    pending_path: Option<std::path::PathBuf>,
    loaded_path: Option<std::path::PathBuf>,
    metadata: Option<StepMetadata>,
    shells: Vec<ShellRecord>,
    faces: Vec<FaceRecord>,
    error: Option<String>,
    loading_job: Option<LoadJob>,
    pending_bounds: Option<Bounds>,
    panel_width: f32,
    // Viewport overlay toggles
    show_random_colors: bool,
    show_bounding_box: bool,
    show_wireframe: bool,
    show_axes: bool,
    axes_mode: AxesMode,
    show_grids: bool,
    show_grid_xy: bool,
    show_grid_yz: bool,
    show_grid_xz: bool,
    show_scale_ruler: bool,
    scene_data: Option<StepScene>,
    needs_mesh_rebuild: bool,
    current_bounds: Option<Bounds>,
    /// Tessellation density factor (smaller = more triangles). Range: 0.0005 to 0.02
    tessellation_factor: f64,
    /// Tessellation factor used for currently loaded scene (to detect changes)
    applied_tessellation_factor: f64,
    /// Flag to trigger visibility update (avoids costly is_changed() checks)
    visibility_changed: bool,
    /// Scene normalization: original center (for wireframe rendering)
    scene_center: Vec3,
    /// Scene normalization: scale factor (for wireframe rendering)
    scene_scale: f32,
    /// Selection mode (point/edge/face)
    selection_mode: SelectionMode,
}

impl Default for ViewerState {
    fn default() -> Self {
        Self {
            pending_path: None,
            loaded_path: None,
            metadata: None,
            shells: Vec::new(),
            faces: Vec::new(),
            error: None,
            loading_job: None,
            pending_bounds: None,
            panel_width: 340.0,
            show_random_colors: false,
            show_bounding_box: false,
            show_wireframe: false,
            show_axes: true,
            axes_mode: AxesMode::ScreenTriad,
            show_grids: true,
            show_grid_xy: true,
            show_grid_yz: true,
            show_grid_xz: true,
            show_scale_ruler: true,
            scene_data: None,
            needs_mesh_rebuild: false,
            current_bounds: None,
            tessellation_factor: 0.001, // Default: matches original hardcoded value
            applied_tessellation_factor: 0.001,
            visibility_changed: false,
            scene_center: Vec3::ZERO,
            scene_scale: 1.0,
            selection_mode: SelectionMode::Face,
        }
    }
}

struct FaceRecord {
    id: usize,
    shell_id: usize,
    name: String,
    triangles: usize,
    visible: bool,
    ui_color: [f32; 3],
    mesh_handle: Handle<Mesh>,
    material_handle: Handle<StandardMaterial>,
}

struct ShellRecord {
    id: usize,
    name: String,
    expanded: bool,
    face_ids: Vec<usize>, // indices into ViewerState.faces
}

#[derive(Resource, Default)]
struct SelectionState {
    selected_faces: std::collections::HashSet<usize>,
    selected_edges: Vec<(Vec3, Vec3)>,
    selected_points: Vec<Vec3>,
}

#[derive(Component)]
struct FaceMesh {
    face_id: usize,
}

struct LoadJob {
    path: std::path::PathBuf,
    receiver: Mutex<Receiver<LoadMessage>>,
    current_shell: usize,
    total_shells: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct Bounds {
    center: Vec3,
    min: Vec3,
    max: Vec3,
}

#[derive(Component)]
struct MainCamera;

/// Render a STEP Parameter as a collapsible tree in egui.
fn parameter_ui(ui: &mut egui::Ui, param: &Parameter, label: &str, depth: usize) {
    match param {
        Parameter::List(items) if !items.is_empty() => {
            egui::CollapsingHeader::new(format!("{} ({})", label, items.len()))
                .id_salt(format!("{}_{}", label, depth))
                .default_open(depth < 1)
                .show(ui, |ui| {
                    for (i, item) in items.iter().enumerate() {
                        parameter_ui(ui, item, &format!("[{}]", i), depth + 1);
                    }
                });
        }
        Parameter::List(_) => {
            ui.label(format!("{}: []", label));
        }
        Parameter::String(s) if s.is_empty() => {
            ui.label(format!("{}: (empty)", label));
        }
        Parameter::String(s) => {
            ui.horizontal_wrapped(|ui| {
                ui.label(format!("{}:", label));
                ui.add(egui::Label::new(s.as_str()).wrap());
            });
        }
        Parameter::Integer(n) => {
            ui.label(format!("{}: {}", label, n));
        }
        Parameter::Real(x) => {
            ui.label(format!("{}: {}", label, x));
        }
        Parameter::Enumeration(e) => {
            ui.label(format!("{}: .{}.", label, e));
        }
        Parameter::Typed { keyword, parameter } => {
            egui::CollapsingHeader::new(format!("{}: {}", label, keyword))
                .id_salt(format!("{}_typed_{}", label, depth))
                .default_open(depth < 2)
                .show(ui, |ui| {
                    parameter_ui(ui, parameter, "value", depth + 1);
                });
        }
        Parameter::Ref(name) => {
            ui.label(format!("{}: {:?}", label, name));
        }
        Parameter::NotProvided => {
            ui.label(format!("{}: $", label));
        }
        Parameter::Omitted => {
            ui.label(format!("{}: *", label));
        }
    }
}

fn main() {
    let cli_path = std::env::args().nth(1).map(std::path::PathBuf::from);

    App::new()
        .insert_resource(ViewerState {
            pending_path: cli_path.clone(),
            ..Default::default()
        })
        .insert_resource(SelectionState::default())
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "STEP Viewer (Bevy + egui)".into(),
                        present_mode: PresentMode::AutoVsync,
                        fit_canvas_to_parent: true,
                        prevent_default_event_handling: false,
                        window_theme: Some(WindowTheme::Dark),
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                .set(LogPlugin {
                    filter: "info,wgpu_core=warn,wgpu_hal=warn".into(),
                    level: bevy::log::Level::INFO,
                    ..Default::default()
                }),
        )
        .add_plugins(EguiPlugin::default())
        .add_plugins(PanOrbitCameraPlugin)
        .insert_resource(WinitSettings::desktop_app())
        .add_systems(Startup, setup_scene)
        .add_systems(Update, process_load_requests)
        .add_systems(Update, rebuild_meshes_on_toggle)
        .add_systems(EguiPrimaryContextPass, ui_system)
        .add_systems(Update, normalize_scene_and_setup_camera)
        .add_systems(Update, apply_face_visibility)
        .add_systems(Update, disable_camera_when_egui_wants_input)
        .add_systems(Update, draw_gizmos)
        .add_systems(Update, draw_selection_gizmos)
        .add_systems(Update, handle_picking_clicks)
        .run();
}

fn setup_scene(mut commands: Commands, mut egui_global_settings: ResMut<EguiGlobalSettings>) {
    // Disable auto egui context - we create our own camera for it
    egui_global_settings.auto_create_primary_context = false;

    // Ambient light - low for more contrast
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 200.0,
        affects_lightmapped_meshes: false,
    });

    // Main 3D camera with lights as children (so lights move with camera)
    // Camera at ~2 units from origin for viewing unit-sized normalized scene
    commands
        .spawn((
            MainCamera,
            Camera3d::default(),
            Transform::from_xyz(1.5, 1.0, 1.5).looking_at(Vec3::ZERO, Vec3::Y),
            PanOrbitCamera {
                focus: Vec3::ZERO,
                radius: Some(2.0),
                ..Default::default()
            },
        ))
        .with_children(|parent| {
            // Key light - main directional light from top-left (relative to camera)
            parent.spawn((
                DirectionalLight {
                    illuminance: 15000.0,
                    shadows_enabled: true,
                    ..Default::default()
                },
                Transform::from_rotation(Quat::from_euler(
                    EulerRot::YXZ,
                    std::f32::consts::PI * 0.25,  // 45° rotated left
                    std::f32::consts::PI * -0.3,  // 54° down from horizontal
                    0.0,
                )),
            ));

            // Back light - from bottom-right-back (relative to camera)
            parent.spawn((
                DirectionalLight {
                    illuminance: 2000.0,
                    shadows_enabled: false,
                    ..Default::default()
                },
                Transform::from_rotation(Quat::from_euler(
                    EulerRot::YXZ,
                    std::f32::consts::PI * -0.7,  // 126° rotated right (behind)
                    std::f32::consts::PI * 0.15,  // 27° up from horizontal
                    0.0,
                )),
            ));
        });

    // Egui-only camera for UI overlay
    commands.spawn((
        PrimaryEguiContext,
        Camera3d::default(),
        RenderLayers::none(),
        Camera {
            order: 1,
            ..Default::default()
        },
    ));
}

fn process_load_requests(
    mut commands: Commands,
    mut state: ResMut<ViewerState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    existing_meshes: Query<Entity, With<FaceMesh>>,
) {
    // Start a new load if requested
    if let Some(path) = state.pending_path.take() {
        for entity in existing_meshes.iter() {
            commands.entity(entity).despawn();
        }
        state.shells.clear();
        state.faces.clear();
        state.metadata = None;
        state.loaded_path = None;
        state.error = None;
        state.scene_data = None;

        let receiver = load_step_file_streaming(path.clone(), state.tessellation_factor);
        state.loading_job = Some(LoadJob {
            path,
            receiver: Mutex::new(receiver),
            current_shell: 0,
            total_shells: 0,
        });
        info!("Started loading STEP file");
    }

    // Poll the loading job for new messages
    let Some(job) = state.loading_job.as_mut() else {
        return;
    };

    // Collect all available messages first (to avoid borrow issues)
    let messages: Vec<_> = {
        let receiver = job.receiver.lock().unwrap();
        receiver.try_iter().collect()
    };

    // Process collected messages
    for msg in messages {
        // Re-borrow job mutably for each message
        let Some(job) = state.loading_job.as_mut() else {
            return;
        };

        match msg {
            LoadMessage::Metadata(meta) => {
                state.metadata = Some(meta);
            }
            LoadMessage::TotalShells(total) => {
                job.total_shells = total;
            }
            LoadMessage::Progress(current, _total) => {
                job.current_shell = current;
            }
            LoadMessage::Shell(shell) => {
                // Store shell in scene_data - don't spawn meshes yet (need bounds first)
                if let Some(scene) = state.scene_data.as_mut() {
                    scene.shells.push(shell);
                } else {
                    state.scene_data = Some(StepScene {
                        metadata: state.metadata.clone().unwrap_or_default(),
                        shells: vec![shell],
                    });
                }
            }
            LoadMessage::Done => {
                let path = job.path.clone();
                state.loaded_path = Some(path);
                state.loading_job = None;

                // Compute bounds for ENTIRE scene
                let bounds = state.scene_data.as_ref().and_then(compute_bounds);

                if let Some(bounds) = bounds {
                    let size = bounds.max - bounds.min;
                    let max_dim = size.x.max(size.y).max(size.z);
                    let scale = if max_dim > 0.0 { 1.0 / max_dim } else { 1.0 };

                    // Store normalization params for wireframe rendering
                    state.scene_center = bounds.center;
                    state.scene_scale = scale;

                    info!(
                        "Scene bounds: center=({:.2}, {:.2}, {:.2}), max_dim={:.2}, scale={:.4}",
                        bounds.center.x, bounds.center.y, bounds.center.z, max_dim, scale
                    );

                    // Now spawn all meshes with normalization applied
                    // Take scene_data temporarily to avoid borrow conflict
                    if let Some(scene) = state.scene_data.take() {
                        for shell in &scene.shells {
                            spawn_shell_faces_normalized(
                                shell,
                                &mut commands,
                                &mut meshes,
                                &mut materials,
                                &mut state,
                                bounds.center,
                                scale,
                            );
                        }
                        state.scene_data = Some(scene);
                    }
                    state.current_bounds = Some(Bounds {
                        center: Vec3::ZERO,
                        min: (bounds.min - bounds.center) * scale,
                        max: (bounds.max - bounds.center) * scale,
                    });
                }

                // Track the tessellation factor used for this load
                state.applied_tessellation_factor = state.tessellation_factor;

                info!(
                    "Finished loading {} shells, {} faces",
                    state.shells.len(),
                    state.faces.len()
                );
                return;
            }
            LoadMessage::Error(err) => {
                state.error = Some(err);
                state.loading_job = None;
                return;
            }
        }
    }
}

/// Spawn faces for a single shell with normalization applied
fn spawn_shell_faces_normalized(
    shell: &StepShell,
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    state: &mut ResMut<ViewerState>,
    scene_center: Vec3,
    scale: f32,
) {
    let use_random_colors = state.show_random_colors;
    let base_face_id = state.faces.len();
    let mut face_ids = Vec::new();

    // Shell color from STEP file (if defined)
    let step_color = shell.color;

    for (idx, face) in shell.faces.iter().enumerate() {
        let global_face_id = base_face_id + idx;

        // For random colors: each face gets its own color based on global_face_id
        // For STEP colors: all faces in shell use the STEP-defined color
        // Otherwise: neutral gray (handled in mesh function)
        let ui_rgb = if let Some(color) = step_color {
            color
        } else {
            let (_, rgb) = color_for_index(global_face_id);
            rgb
        };

        let (mesh, tri_count) = bevy_mesh_from_polygon_normalized(
            &face.mesh,
            ui_rgb,
            use_random_colors || step_color.is_some(),
            scene_center,
            scale,
        );
        let mesh_handle = meshes.add(mesh);

        // Use white base color to show vertex colors
        let material = materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.4,
            metallic: 0.0,
            ..Default::default()
        });
        let material_handle = material.clone();

        commands.spawn((
            FaceMesh {
                face_id: global_face_id,
            },
            Mesh3d(mesh_handle.clone()),
            MeshMaterial3d(material),
            Transform::default(),
            Visibility::Visible,
        ));

        state.faces.push(FaceRecord {
            id: global_face_id,
            shell_id: shell.id,
            name: face.name.clone(),
            triangles: tri_count,
            visible: true,
            ui_color: ui_rgb,
            mesh_handle,
            material_handle,
        });

        face_ids.push(global_face_id);
    }

    state.shells.push(ShellRecord {
        id: shell.id,
        name: shell.name.clone(),
        expanded: true,
        face_ids,
    });
}

fn compute_bounds(scene: &StepScene) -> Option<Bounds> {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    let mut has_points = false;

    for shell in &scene.shells {
        for face in &shell.faces {
            for p in face.mesh.positions() {
                let pos = Vec3::new(p.x as f32, p.y as f32, p.z as f32);
                min = min.min(pos);
                max = max.max(pos);
                has_points = true;
            }
        }
    }

    if !has_points {
        return None;
    }

    let center = (min + max) * 0.5;
    let size = max - min;
    log::info!(
        "Scene bounds: min=({:.2}, {:.2}, {:.2}), max=({:.2}, {:.2}, {:.2}), size=({:.2}, {:.2}, {:.2})",
        min.x,
        min.y,
        min.z,
        max.x,
        max.y,
        max.z,
        size.x,
        size.y,
        size.z
    );
    Some(Bounds { center, min, max })
}

fn bevy_mesh_from_polygon_normalized(
    mesh: &PolygonMesh,
    shell_color: [f32; 3],
    use_random_colors: bool,
    scene_center: Vec3,
    scale: f32,
) -> (Mesh, usize) {
    // Apply normalization: (pos - center) * scale
    let positions: Vec<[f32; 3]> = mesh
        .positions()
        .iter()
        .map(|p| {
            let pos = Vec3::new(p.x as f32, p.y as f32, p.z as f32);
            let normalized = (pos - scene_center) * scale;
            [normalized.x, normalized.y, normalized.z]
        })
        .collect();

    let normals: Vec<[f32; 3]> = mesh
        .normals()
        .iter()
        .map(|n| [n.x as f32, n.y as f32, n.z as f32])
        .collect();

    // Collect vertices as (pos_idx, nor_idx) tuples
    let mut vertices: Vec<(usize, Option<usize>)> = Vec::new();

    for tri in mesh.tri_faces() {
        vertices.extend([
            (tri[0].pos, tri[0].nor),
            (tri[1].pos, tri[1].nor),
            (tri[2].pos, tri[2].nor),
        ]);
    }

    for quad in mesh.quad_faces() {
        vertices.extend([
            (quad[0].pos, quad[0].nor),
            (quad[1].pos, quad[1].nor),
            (quad[2].pos, quad[2].nor),
            (quad[0].pos, quad[0].nor),
            (quad[2].pos, quad[2].nor),
            (quad[3].pos, quad[3].nor),
        ]);
    }

    for face in mesh.other_faces() {
        if face.len() < 3 {
            continue;
        }
        let first = (face[0].pos, face[0].nor);
        face.windows(2).skip(1).for_each(|w| {
            vertices.extend([first, (w[0].pos, w[0].nor), (w[1].pos, w[1].nor)]);
        });
    }

    // Expand indexed geometry to flat arrays
    let (flat_positions, flat_normals): (Vec<[f32; 3]>, Vec<[f32; 3]>) = vertices
        .iter()
        .map(|(pos_idx, nor_idx)| {
            let pos = positions[*pos_idx];
            let nor = nor_idx.map(|ni| normals[ni]).unwrap_or([0.0, 0.0, 1.0]); // Fallback normal
            (pos, nor)
        })
        .unzip();

    // Uniform color per shell: distinct color if random colors enabled, gray otherwise
    let color = if use_random_colors {
        [shell_color[0], shell_color[1], shell_color[2], 1.0]
    } else {
        [0.7, 0.7, 0.7, 1.0]
    };
    let colors: Vec<[f32; 4]> = vec![color; flat_positions.len()];

    let mut bevy_mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    bevy_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, flat_positions);
    bevy_mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, flat_normals);
    bevy_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);

    (bevy_mesh, vertices.len() / 3)
}

fn ui_system(
    mut contexts: EguiContexts,
    mut state: ResMut<ViewerState>,
    mut exit: MessageWriter<AppExit>,
    windows: Query<&Window>,
    mut camera_query: Query<&mut Camera, With<MainCamera>>,
    cam_tf_query: Query<&Transform, With<MainCamera>>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };

    egui::TopBottomPanel::top("menu").show(ctx, |ui| {
        ui.horizontal(|ui| {
            if ui.button("Open STEP file…").clicked() {
                #[cfg(not(target_arch = "wasm32"))]
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("STEP", &["stp", "step"])
                    .pick_file()
                {
                    state.pending_path = Some(path);
                }

                #[cfg(target_arch = "wasm32")]
                {
                    state.error = Some("File open dialog is not supported on wasm".to_string());
                }
            }

            ui.separator();

            // Quality slider (logarithmic scale for better UX)
            // Higher quality = finer mesh (smaller tessellation factor)
            // Quality maps to -log10(tessellation_factor): 1.5 (low) to 4.0 (ultra)
            let mut quality = -state.tessellation_factor.log10();
            ui.label("Quality:");
            let slider = ui.add(
                egui::Slider::new(&mut quality, 1.5_f64..=4.0_f64)
                    .show_value(false)
                    .custom_formatter(|v, _| {
                        if v > 3.5 {
                            "Ultra".to_string()
                        } else if v > 2.8 {
                            "High".to_string()
                        } else if v > 2.2 {
                            "Medium".to_string()
                        } else {
                            "Low".to_string()
                        }
                    }),
            );
            let new_factor = 10_f64.powf(-quality);
            if slider.changed() {
                state.tessellation_factor = new_factor;
            }
            // Reload when slider released and factor differs from what was used to load
            let factor_changed =
                (state.tessellation_factor - state.applied_tessellation_factor).abs() > 1e-10;
            if !slider.dragged()
                && factor_changed
                && state.loaded_path.is_some()
                && state.loading_job.is_none()
            {
                state.pending_path = state.loaded_path.clone();
            }
            slider.on_hover_text("Tessellation quality");

            ui.separator();

            if let Some(path) = &state.loaded_path {
                ui.label(format!("Loaded: {}", path.display()));
            } else {
                ui.label("No file loaded");
            }
        });
    });

    let panel_response = egui::SidePanel::left("entities")
        .default_width(340.0)
        .resizable(true)
        .show(ctx, |ui| {
            ui.heading("Model Hierarchy");
            ui.separator();

            if state.shells.is_empty() && state.loading_job.is_none() {
                ui.label("Load a STEP file to see hierarchy");
            } else if state.shells.is_empty() {
                ui.label("Loading...");
            } else {
                // Track if any visibility checkbox was toggled
                let vis_changed = std::cell::Cell::new(false);

                egui::ScrollArea::vertical().show(ui, |ui| {
                    // We need to collect shell data first to avoid borrow issues
                    let shell_data: Vec<_> = state
                        .shells
                        .iter()
                        .map(|s| (s.id, s.name.clone(), s.expanded, s.face_ids.clone()))
                        .collect();

                    for (shell_id, shell_name, expanded, face_ids) in shell_data {
                        let face_count = face_ids.len();
                        let header = egui::CollapsingHeader::new(format!(
                            "{} ({} faces)",
                            shell_name, face_count
                        ))
                        .id_salt(format!("shell_{}", shell_id))
                        .default_open(expanded);

                        header.show(ui, |ui| {
                            for &face_id in &face_ids {
                                if let Some(face) = state.faces.iter_mut().find(|f| f.id == face_id)
                                {
                                    let color = egui::Color32::from_rgb(
                                        (face.ui_color[0] * 255.0) as u8,
                                        (face.ui_color[1] * 255.0) as u8,
                                        (face.ui_color[2] * 255.0) as u8,
                                    );
                                    ui.horizontal(|ui| {
                                        let prev_visible = face.visible;
                                        ui.checkbox(&mut face.visible, "");
                                        if face.visible != prev_visible {
                                            vis_changed.set(true);
                                        }
                                        ui.colored_label(color, "■");
                                        ui.label(format!(
                                            "{} ({} tris)",
                                            face.name, face.triangles
                                        ));
                                    });
                                }
                            }
                        });
                    }
                });

                // Set visibility_changed flag if any checkbox was toggled
                if vis_changed.get() {
                    state.visibility_changed = true;
                }
            }
        });

    // Track left panel width
    let left_panel_width = panel_response.response.rect.width();

    let right_panel_response = egui::SidePanel::right("metadata")
        .resizable(true)
        .default_width(380.0)
        .show(ctx, |ui| {
            ui.heading("File Information");
            ui.separator();
            if let Some(meta) = &state.metadata {
                ui.label(format!("Entity Count: {}", meta.entity_count));
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for entry in &meta.headers {
                        egui::CollapsingHeader::new(&entry.name)
                            .id_salt(&entry.name)
                            .default_open(entry.name == "FILE_NAME" || entry.name == "FILE_SCHEMA")
                            .show(ui, |ui| {
                                parameter_ui(ui, &entry.parameter, "value", 0);
                            });
                    }
                });
            } else {
                ui.label("No metadata available");
            }
        });

    // Track right panel width
    let right_panel_width = right_panel_response.response.rect.width();
    state.panel_width = left_panel_width;

    // Get window info for viewport overlay positioning
    let window_info = windows.single().ok().map(|w| (w.width(), w.height()));

    // Update camera viewport to account for UI panels
    if let Ok(mut camera) = camera_query.single_mut()
        && let Ok(window) = windows.single()
    {
        let scale_factor = window.scale_factor();
        let left_panel_physical = (left_panel_width * scale_factor) as u32;
        let right_panel_physical = (right_panel_width * scale_factor) as u32;
        let window_width_physical = window.physical_width();
        let window_height_physical = window.physical_height();

        let viewport_width = window_width_physical
            .saturating_sub(left_panel_physical)
            .saturating_sub(right_panel_physical);

        camera.viewport = Some(Viewport {
            physical_position: UVec2::new(left_panel_physical, 0),
            physical_size: UVec2::new(viewport_width, window_height_physical),
            ..Default::default()
        });
    }

    // Show viewport toolbar and overlays
    if let Some((window_width, window_height)) = window_info {
        let viewport_x = left_panel_width;
        let viewport_width = window_width - left_panel_width - right_panel_width;

        // Viewport toolbar (top-right of 3D viewport, not main window)
        if state.scene_data.is_some() {
            let toolbar_margin = 8.0;
            // Position relative to right edge of viewport (before the right panel)
            let toolbar_x = left_panel_width + viewport_width - toolbar_margin;
            let toolbar_y = toolbar_margin + 24.0; // Below menu bar

            egui::Area::new(egui::Id::new("viewport_toolbar"))
                .anchor(egui::Align2::RIGHT_TOP, egui::vec2(0.0, 0.0))
                .fixed_pos(egui::pos2(toolbar_x, toolbar_y))
                .show(ctx, |ui| {
                    ui.visuals_mut().widgets.inactive.weak_bg_fill =
                        egui::Color32::from_rgba_unmultiplied(40, 40, 40, 220);
                    ui.visuals_mut().widgets.hovered.weak_bg_fill =
                        egui::Color32::from_rgba_unmultiplied(60, 60, 60, 230);
                    ui.visuals_mut().widgets.active.weak_bg_fill =
                        egui::Color32::from_rgba_unmultiplied(80, 80, 80, 240);

                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(4.0, 0.0);

                        // Random colors toggle (dice icon: 🎲)
                        let colors_btn = ui.selectable_label(
                            state.show_random_colors,
                            egui::RichText::new("🎲").size(18.0),
                        );
                        if colors_btn.clicked() {
                            state.show_random_colors = !state.show_random_colors;
                            state.needs_mesh_rebuild = true;
                        }
                        colors_btn.on_hover_text("Random colors");

                        // Bounding box toggle (box icon: ⬜)
                        let bbox_btn = ui.selectable_label(
                            state.show_bounding_box,
                            egui::RichText::new("⬜").size(18.0),
                        );
                        if bbox_btn.clicked() {
                            state.show_bounding_box = !state.show_bounding_box;
                        }
                        bbox_btn.on_hover_text("Bounding box");

                        // Wireframe toggle (grid icon: ▦)
                        let wire_btn = ui.selectable_label(
                            state.show_wireframe,
                            egui::RichText::new("▦").size(18.0),
                        );
                        if wire_btn.clicked() {
                            state.show_wireframe = !state.show_wireframe;
                        }
                        wire_btn.on_hover_text("Wireframe edges");

                        // Axes visibility toggle (🧭)
                        let axes_btn = ui.selectable_label(
                            state.show_axes,
                            egui::RichText::new("🧭").size(18.0),
                        );
                        if axes_btn.clicked() {
                            state.show_axes = !state.show_axes;
                        }
                        axes_btn.on_hover_text("Show axes indicator");

                        // Axes mode combo (World vs Screen)
                        egui::ComboBox::from_id_salt("axes_mode_combo")
                            .selected_text(match state.axes_mode { AxesMode::World => "World", AxesMode::ScreenTriad => "Screen" })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut state.axes_mode, AxesMode::World, "World");
                                ui.selectable_value(&mut state.axes_mode, AxesMode::ScreenTriad, "Screen");
                            });

                        // Grids toggle (grid icon: ⌗)
                        let grids_btn = ui.selectable_label(
                            state.show_grids,
                            egui::RichText::new("⌗").size(18.0),
                        );
                        if grids_btn.clicked() {
                            state.show_grids = !state.show_grids;
                        }
                        grids_btn.on_hover_text("Show XY/YZ/XZ grids");

                        // Grid planes selection via dropdown with checkboxes
                        let mut planes: Vec<&str> = Vec::new();
                        if state.show_grid_xy { planes.push("XY"); }
                        if state.show_grid_yz { planes.push("YZ"); }
                        if state.show_grid_xz { planes.push("XZ"); }
                        let selected_text = if planes.is_empty() { "None".to_string() } else { planes.join("/") };
                        egui::ComboBox::from_id_salt("grid_planes_combo")
                            .selected_text(format!("Plane: {}", selected_text))
                            .show_ui(ui, |ui| {
                                let _ = ui.checkbox(&mut state.show_grid_xy, "XY");
                                let _ = ui.checkbox(&mut state.show_grid_yz, "YZ");
                                let _ = ui.checkbox(&mut state.show_grid_xz, "XZ");
                            });

                        ui.separator();

                        // Selection mode
                        ui.label("Select:");
                        egui::ComboBox::from_id_salt("selection_mode_combo")
                            .selected_text(match state.selection_mode { SelectionMode::Point => "Point", SelectionMode::Edge => "Edge", SelectionMode::Face => "Face" })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut state.selection_mode, SelectionMode::Point, "Point");
                                ui.selectable_value(&mut state.selection_mode, SelectionMode::Edge, "Edge");
                                ui.selectable_value(&mut state.selection_mode, SelectionMode::Face, "Face");
                            });
                        ui.label("Ctrl Multi/Cancel");

                        // Scale ruler toggle (ruler icon: 📏)
                        let ruler_btn = ui.selectable_label(
                            state.show_scale_ruler,
                            egui::RichText::new("📏").size(18.0),
                        );
                        if ruler_btn.clicked() {
                            state.show_scale_ruler = !state.show_scale_ruler;
                        }
                        ruler_btn.on_hover_text("Show scale ruler");
                    });
                });
        }

        if let Some(err) = &state.error {
            // Error overlay
            egui::Area::new(egui::Id::new("error_overlay"))
                .fixed_pos(egui::pos2(viewport_x + 10.0, window_height - 40.0))
                .show(ctx, |ui| {
                    ui.colored_label(egui::Color32::RED, err);
                });
        } else if let Some(job) = &state.loading_job {
            // Progress bar at bottom of viewport
            let bar_height = 24.0;
            let bar_y = window_height - bar_height - 10.0;

            let current = job.current_shell;
            let total = job.total_shells;
            let fraction = if total > 0 {
                current as f32 / total as f32
            } else {
                0.0
            };

            egui::Area::new(egui::Id::new("progress_overlay"))
                .fixed_pos(egui::pos2(viewport_x, bar_y))
                .show(ctx, |ui| {
                    let rect = egui::Rect::from_min_size(
                        ui.cursor().min,
                        egui::vec2(viewport_width, bar_height),
                    );

                    // Background
                    ui.painter().rect_filled(
                        rect,
                        4.0,
                        egui::Color32::from_rgba_unmultiplied(0, 0, 0, 200),
                    );

                    // Progress bar fill
                    if fraction > 0.0 {
                        let progress_rect = egui::Rect::from_min_size(
                            rect.min,
                            egui::vec2(viewport_width * fraction, bar_height),
                        );
                        ui.painter().rect_filled(
                            progress_rect,
                            4.0,
                            egui::Color32::from_rgb(100, 149, 237),
                        );
                    }

                    // Text
                    let text = if total > 0 {
                        format!(
                            "Tessellating shell {}/{} ({:.0}%)",
                            current,
                            total,
                            fraction * 100.0
                        )
                    } else {
                        "Parsing STEP file...".to_string()
                    };

                    ui.painter().text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        text,
                        egui::FontId::proportional(14.0),
                        egui::Color32::WHITE,
                    );
                });

            // Request repaint to update progress
            ctx.request_repaint();
        }
    }

    // Allow escape to quit quickly on desktop
    if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
        exit.write(AppExit::Success);
    }

    // Scale ruler overlay (bottom-left of viewport)
    if state.show_scale_ruler {
        if let Some(bounds) = state.current_bounds {
            let viewport_x = left_panel_width;
            let ruler_margin = 12.0;
            let ruler_width = 160.0;
            let ruler_height = 50.0;
            let ruler_pos = egui::pos2(viewport_x + ruler_margin, window_info.map(|(_,h)| h - ruler_height - ruler_margin).unwrap_or(0.0));

            egui::Area::new(egui::Id::new("scale_ruler_overlay"))
                .fixed_pos(ruler_pos)
                .show(ctx, |ui| {
                    let rect = egui::Rect::from_min_size(
                        ui.cursor().min,
                        egui::vec2(ruler_width, ruler_height),
                    );
                    // Background
                    ui.painter().rect_filled(
                        rect,
                        6.0,
                        egui::Color32::from_rgba_unmultiplied(20, 20, 20, 200),
                    );

                    // Compute scale info: 1 normalized unit -> original STEP units
                    let max_dim_original = if state.scene_scale > 0.0 { 1.0 / state.scene_scale } else { 0.0 };
                    // Adaptive grid spacing to match gizmo grid
                    let (_, grid_unit) = compute_grid_params(&state);
                    let grid_unit_original = grid_unit / state.scene_scale;

                    // Draw a simple ruler bar
                    let bar_margin = 12.0;
                    let bar_len_px = ruler_width - 2.0 * bar_margin;
                    let bar_y = rect.top() + ruler_height - 22.0;
                    let bar_min = egui::pos2(rect.left() + bar_margin, bar_y);
                    let bar_max = egui::pos2(rect.left() + bar_margin + bar_len_px, bar_y);
                    ui.painter().line_segment([
                        bar_min,
                        bar_max,
                    ], egui::Stroke { width: 2.0, color: egui::Color32::LIGHT_GRAY });
                    // End ticks
                    ui.painter().line_segment([
                        egui::pos2(bar_min.x, bar_y - 6.0),
                        egui::pos2(bar_min.x, bar_y + 6.0),
                    ], egui::Stroke { width: 2.0, color: egui::Color32::LIGHT_GRAY });
                    ui.painter().line_segment([
                        egui::pos2(bar_max.x, bar_y - 6.0),
                        egui::pos2(bar_max.x, bar_y + 6.0),
                    ], egui::Stroke { width: 2.0, color: egui::Color32::LIGHT_GRAY });

                    // Labels
                    let title = format!("Scale: 1.0 → {:.3} STEP units", max_dim_original);
                    ui.painter().text(
                        egui::pos2(rect.center().x, rect.top() + 16.0),
                        egui::Align2::CENTER_CENTER,
                        title,
                        egui::FontId::proportional(14.0),
                        egui::Color32::WHITE,
                    );
                    let sub = format!("Grid {:.3} → {:.3}", grid_unit, grid_unit_original);
                    ui.painter().text(
                        egui::pos2(rect.center().x, rect.top() + 32.0),
                        egui::Align2::CENTER_CENTER,
                        sub,
                        egui::FontId::proportional(12.0),
                        egui::Color32::LIGHT_GRAY,
                    );
                });
            }
        }

    // Screen-space axes triad (bottom-right of viewport)
    if state.show_axes && matches!(state.axes_mode, AxesMode::ScreenTriad) {
        if let Ok(cam_tf) = cam_tf_query.single() {
            if let Some((window_width, window_height)) = window_info {
                let viewport_width = window_width - left_panel_width - right_panel_width;
                let triad_margin = 12.0;
                let triad_size = 90.0;
                let triad_x = left_panel_width + viewport_width - triad_size - triad_margin;
                let triad_y = window_height - triad_size - triad_margin;

                egui::Area::new(egui::Id::new("axes_triad_overlay"))
                    .fixed_pos(egui::pos2(triad_x, triad_y))
                    .show(ctx, |ui| {
                        let rect = egui::Rect::from_min_size(
                            ui.cursor().min,
                            egui::vec2(triad_size, triad_size),
                        );
                        // Background
                        ui.painter().rect_filled(
                            rect,
                            6.0,
                            egui::Color32::from_rgba_unmultiplied(20, 20, 20, 160),
                        );

                        let center = egui::pos2(rect.center().x, rect.center().y);
                        let scale = triad_size * 0.35; // arrow length

                        // Compute camera-space directions for world axes
                        // Convert world axes to view (camera) space
                        let inv = cam_tf.rotation.inverse();
                        let dx = inv * Vec3::X;
                        let dy = inv * Vec3::Y;
                        let dz = inv * Vec3::Z;

                        // Helper to draw an arrow in 2D from center
                        let mut draw_axis = |dir3: Vec3, color: egui::Color32, label: &str| {
                            let v = glam::Vec2::new(dir3.x, dir3.y);
                            let v = if v.length() > 1e-6 { v.normalize() } else { glam::Vec2::new(0.0, 0.0) };
                            let end = egui::pos2(center.x + v.x * scale, center.y - v.y * scale);
                            ui.painter().line_segment([center, end], egui::Stroke { width: 2.0, color });
                            // small head
                            let head = egui::pos2(end.x, end.y);
                            ui.painter().circle_filled(head, 2.5, color);
                            // label near end
                            ui.painter().text(
                                egui::pos2(end.x + 6.0, end.y),
                                egui::Align2::LEFT_CENTER,
                                label,
                                egui::FontId::proportional(12.0),
                                color,
                            );
                        };

                        draw_axis(dx, egui::Color32::from_rgb(255, 64, 64), "X");
                        draw_axis(dy, egui::Color32::from_rgb(64, 220, 64), "Y");
                        draw_axis(dz, egui::Color32::from_rgb(64, 160, 255), "Z");
                    });
            }
        }
    }
}

fn apply_face_visibility(
    mut state: ResMut<ViewerState>,
    mut query: Query<(&FaceMesh, &mut Visibility)>,
) {
    if !state.visibility_changed {
        return;
    }
    state.visibility_changed = false;

    for (mesh, mut visibility) in query.iter_mut() {
        if let Some(record) = state.faces.iter().find(|f| f.id == mesh.face_id) {
            *visibility = if record.visible {
                Visibility::Visible
            } else {
                Visibility::Hidden
            };
        }
    }
}

fn normalize_scene_and_setup_camera(
    mut state: ResMut<ViewerState>,
    mut camera_query: Query<(&mut Transform, &mut PanOrbitCamera), With<MainCamera>>,
    mesh_query: Query<&Transform, (With<FaceMesh>, Without<MainCamera>)>,
) {
    let Some(bounds) = state.pending_bounds else {
        return;
    };

    // Wait until meshes are actually available in the query (ECS delay)
    let mesh_count = mesh_query.iter().count();
    let expected_faces = state.faces.len();
    if mesh_count < expected_faces {
        // Meshes not ready yet, try again next frame
        return;
    }

    // Now we can consume pending_bounds
    state.pending_bounds = None;

    // Calculate scene dimensions
    let size = bounds.max - bounds.min;
    let max_dim = size.x.max(size.y).max(size.z);

    // Store bounds for bounding box gizmo
    state.current_bounds = Some(bounds);

    log::info!(
        "DEBUG: About to setup camera. Bounds center=({:.2}, {:.2}, {:.2}), max_dim={:.2}",
        bounds.center.x,
        bounds.center.y,
        bounds.center.z,
        max_dim
    );

    // Set up camera to view the scene from appropriate distance
    // Use ~1.5x the max dimension for good framing
    let camera_distance = max_dim * 1.5;
    if let Ok((mut transform, mut pan_orbit)) = camera_query.single_mut() {
        log::info!("DEBUG: Found camera, updating PanOrbitCamera");
        pan_orbit.focus = bounds.center;
        pan_orbit.radius = Some(camera_distance);
        pan_orbit.yaw = Some(std::f32::consts::FRAC_PI_4); // 45 degrees
        pan_orbit.pitch = Some(std::f32::consts::FRAC_PI_6); // 30 degrees
        pan_orbit.force_update = true;
        pan_orbit.initialized = false; // Force re-initialization

        // Set initial transform position
        let yaw = std::f32::consts::FRAC_PI_4;
        let pitch = std::f32::consts::FRAC_PI_6;
        let offset = Vec3::new(
            camera_distance * yaw.cos() * pitch.cos(),
            camera_distance * pitch.sin(),
            camera_distance * yaw.sin() * pitch.cos(),
        );
        transform.translation = bounds.center + offset;
        *transform = transform.looking_at(bounds.center, Vec3::Y);

        log::info!(
            "Camera setup: focus=({:.2}, {:.2}, {:.2}), distance={:.2}",
            bounds.center.x,
            bounds.center.y,
            bounds.center.z,
            camera_distance
        );
    } else {
        state.pending_bounds = Some(bounds);
    }
}

fn rebuild_meshes_on_toggle(mut state: ResMut<ViewerState>, mut meshes: ResMut<Assets<Mesh>>) {
    if !state.needs_mesh_rebuild {
        return;
    }
    state.needs_mesh_rebuild = false;

    let Some(scene) = &state.scene_data else {
        return;
    };

    let use_random_colors = state.show_random_colors;

    // Update vertex colors in-place on existing meshes (no despawn/respawn)
    // Iterate through all faces in all shells
    for shell in &scene.shells {
        // STEP-defined colors always show; random colors only when toggle is on
        let apply_colors = use_random_colors || shell.color.is_some();

        for step_face in &shell.faces {
            // Find the corresponding FaceRecord
            if let Some(face_record) = state
                .faces
                .iter()
                .find(|f| f.shell_id == shell.id && f.name == step_face.name)
                && let Some(mesh) = meshes.get_mut(&face_record.mesh_handle)
            {
                let colors =
                    recompute_colors_for_mesh(&step_face.mesh, face_record.ui_color, apply_colors);
                mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
            }
        }
    }
}

fn color_for_index(idx: usize) -> (Color, [f32; 3]) {
    use bevy::color::Hsva;
    // Use golden ratio for hue spread (in degrees for Hsva)
    let hue = (idx as f32 * 0.618_034 * 360.0) % 360.0;
    // Vary saturation and value to distinguish similar hues
    let s = 0.5 + 0.4 * ((idx as f32 * 0.317) % 1.0); // 0.5-0.9
    let v = 0.7 + 0.25 * ((idx as f32 * 0.513) % 1.0); // 0.7-0.95
    let hsva = Hsva::new(hue, s, v, 1.0);
    let color = Color::from(hsva);
    let srgba = color.to_srgba();
    (color, [srgba.red, srgba.green, srgba.blue])
}

/// Recompute vertex colors for a mesh without rebuilding geometry.
/// Returns colors in the same vertex order as bevy_mesh_from_polygon.
fn recompute_colors_for_mesh(
    mesh: &PolygonMesh,
    shell_color: [f32; 3],
    use_random_colors: bool,
) -> Vec<[f32; 4]> {
    // Count total vertices
    let mut vertex_count = 0usize;
    vertex_count += mesh.tri_faces().len() * 3;
    vertex_count += mesh.quad_faces().len() * 6; // 2 triangles per quad
    for face in mesh.other_faces() {
        if face.len() >= 3 {
            vertex_count += (face.len() - 2) * 3;
        }
    }

    // Use shell's distinct color if random colors enabled, otherwise neutral gray
    let color = if use_random_colors {
        [shell_color[0], shell_color[1], shell_color[2], 1.0]
    } else {
        [0.7, 0.7, 0.7, 1.0] // Neutral gray
    };

    vec![color; vertex_count]
}

/// Disable PanOrbitCamera when egui wants pointer input (e.g., during panel resize).
fn disable_camera_when_egui_wants_input(
    mut contexts: EguiContexts,
    mut camera_query: Query<&mut PanOrbitCamera, With<MainCamera>>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };

    let egui_wants_input = ctx.wants_pointer_input() || ctx.is_pointer_over_area();

    if let Ok(mut pan_orbit) = camera_query.single_mut() {
        pan_orbit.enabled = !egui_wants_input;
    }
}

/// Draw bounding box and wireframe gizmos when enabled.
fn draw_gizmos(state: Res<ViewerState>, mut gizmos: Gizmos) {
    // World-space axes (if selected)
    if state.show_axes && matches!(state.axes_mode, AxesMode::World) {
        let axis_len = 1.2_f32; // slightly larger than normalized unit
        gizmos.line(Vec3::ZERO, Vec3::X * axis_len, Color::srgb(1.0, 0.0, 0.0)); // X - red
        gizmos.line(Vec3::ZERO, Vec3::Y * axis_len, Color::srgb(0.0, 1.0, 0.0)); // Y - green
        gizmos.line(Vec3::ZERO, Vec3::Z * axis_len, Color::srgb(0.0, 0.6, 1.0)); // Z - blue
    }

    // Draw semi-transparent grids on XY, YZ, XZ
    if state.show_grids {
        let (grid_extent, grid_spacing) = compute_grid_params(&state);
        let grid_color = Color::srgba(0.8, 0.8, 0.8, 0.25);

        // XY plane (z=0)
        if state.show_grid_xy {
            let mut x = -grid_extent;
            while x <= grid_extent + 1e-6 {
                gizmos.line(Vec3::new(x, -grid_extent, 0.0), Vec3::new(x, grid_extent, 0.0), grid_color);
                x += grid_spacing;
            }
            let mut y = -grid_extent;
            while y <= grid_extent + 1e-6 {
                gizmos.line(Vec3::new(-grid_extent, y, 0.0), Vec3::new(grid_extent, y, 0.0), grid_color);
                y += grid_spacing;
            }
        }

        // YZ plane (x=0)
        if state.show_grid_yz {
            let mut y2 = -grid_extent;
            while y2 <= grid_extent + 1e-6 {
                gizmos.line(Vec3::new(0.0, y2, -grid_extent), Vec3::new(0.0, y2, grid_extent), grid_color);
                y2 += grid_spacing;
            }
            let mut z2 = -grid_extent;
            while z2 <= grid_extent + 1e-6 {
                gizmos.line(Vec3::new(0.0, -grid_extent, z2), Vec3::new(0.0, grid_extent, z2), grid_color);
                z2 += grid_spacing;
            }
        }

        // XZ plane (y=0)
        if state.show_grid_xz {
            let mut x3 = -grid_extent;
            while x3 <= grid_extent + 1e-6 {
                gizmos.line(Vec3::new(x3, 0.0, -grid_extent), Vec3::new(x3, 0.0, grid_extent), grid_color);
                x3 += grid_spacing;
            }
            let mut z3 = -grid_extent;
            while z3 <= grid_extent + 1e-6 {
                gizmos.line(Vec3::new(-grid_extent, 0.0, z3), Vec3::new(grid_extent, 0.0, z3), grid_color);
                z3 += grid_spacing;
            }
        }
    }

    // Draw wireframe edges (STEP geometry boundary edges stored in scene_data)
    if state.show_wireframe {
        if let Some(scene) = &state.scene_data {
            let color = Color::srgba(0.0, 0.0, 0.0, 0.7);
            let center = state.scene_center;
            let scale = state.scene_scale;

            for shell in &scene.shells {
                for (p0_arr, p1_arr) in &shell.edges {
                    // Apply same normalization as mesh vertices: (pos - center) * scale
                    let p0_raw = Vec3::new(p0_arr[0] as f32, p0_arr[1] as f32, p0_arr[2] as f32);
                    let p1_raw = Vec3::new(p1_arr[0] as f32, p1_arr[1] as f32, p1_arr[2] as f32);
                    let p0 = (p0_raw - center) * scale;
                    let p1 = (p1_raw - center) * scale;
                    gizmos.line(p0, p1, color);
                }
            }
        }
    }

    // Draw bounding box
    if state.show_bounding_box
        && let Some(bounds) = state.current_bounds
    {
        let min = bounds.min;
        let max = bounds.max;
        let color = Color::srgb(0.0, 1.0, 0.0); // Green

        // 12 edges of the bounding box
        // Bottom face
        gizmos.line(
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(max.x, min.y, min.z),
            color,
        );
        gizmos.line(
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(max.x, min.y, max.z),
            color,
        );
        gizmos.line(
            Vec3::new(max.x, min.y, max.z),
            Vec3::new(min.x, min.y, max.z),
            color,
        );
        gizmos.line(
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(min.x, min.y, min.z),
            color,
        );
        // Top face
        gizmos.line(
            Vec3::new(min.x, max.y, min.z),
            Vec3::new(max.x, max.y, min.z),
            color,
        );
        gizmos.line(
            Vec3::new(max.x, max.y, min.z),
            Vec3::new(max.x, max.y, max.z),
            color,
        );
        gizmos.line(
            Vec3::new(max.x, max.y, max.z),
            Vec3::new(min.x, max.y, max.z),
            color,
        );
        gizmos.line(
            Vec3::new(min.x, max.y, max.z),
            Vec3::new(min.x, max.y, min.z),
            color,
        );
        // Vertical edges
        gizmos.line(
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(min.x, max.y, min.z),
            color,
        );
        gizmos.line(
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(max.x, max.y, min.z),
            color,
        );
        gizmos.line(
            Vec3::new(max.x, min.y, max.z),
            Vec3::new(max.x, max.y, max.z),
            color,
        );
        gizmos.line(
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(min.x, max.y, max.z),
            color,
        );
    }

    // Selection gizmos are drawn in draw_selection_gizmos()
}

/// Draw selection overlays for points and edges
fn draw_selection_gizmos(
    state: Res<ViewerState>,
    sel: Res<SelectionState>,
    mut gizmos: Gizmos,
) {
    let (_, spacing) = compute_grid_params(&state);
    let cross = spacing * 0.15_f32;
    let pcolor = Color::srgb(1.0, 0.9, 0.2);
    let ecolor = Color::srgb(1.0, 0.6, 0.1);

    // Points: draw small cross
    for &p in &sel.selected_points {
        gizmos.line(p + Vec3::new(-cross, 0.0, 0.0), p + Vec3::new(cross, 0.0, 0.0), pcolor);
        gizmos.line(p + Vec3::new(0.0, -cross, 0.0), p + Vec3::new(0.0, cross, 0.0), pcolor);
        gizmos.line(p + Vec3::new(0.0, 0.0, -cross), p + Vec3::new(0.0, 0.0, cross), pcolor);
    }

    // Edges: re-draw with highlight color
    for &(a, b) in &sel.selected_edges {
        gizmos.line(a, b, ecolor);
    }
}

/// Handle mouse picking for points, edges, and faces with single/multi-select
fn handle_picking_clicks(
    mut contexts: EguiContexts,
    mut state: ResMut<ViewerState>,
    mut sel: ResMut<SelectionState>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window>,
    cam_q: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    meshes: Res<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Only when clicking in the 3D viewport, not over egui
    let Ok(ctx) = contexts.ctx_mut() else { return; };
    if ctx.is_pointer_over_area() || ctx.wants_pointer_input() {
        return;
    }

    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }

    let Ok(window) = windows.single() else { return; };
    let Some(cursor) = window.cursor_position() else { return; };
    let Ok((camera, cam_gt)) = cam_q.single() else { return; };

    let Ok(ray) = camera.viewport_to_world(cam_gt, cursor) else { return; };

    let multi = keys.any_pressed([KeyCode::ControlLeft, KeyCode::ControlRight]);
    if !multi {
        // Clear previous selection and remove face highlights
        for &fid in sel.selected_faces.iter() {
            if let Some(face) = state.faces.iter().find(|f| f.id == fid) {
                if let Some(mat) = materials.get_mut(&face.material_handle) {
                    mat.emissive = Color::BLACK.into();
                }
            }
        }
        sel.selected_faces.clear();
        sel.selected_edges.clear();
        sel.selected_points.clear();
    }

    match state.selection_mode {
        SelectionMode::Face => {
            if let Some((fid, _t)) = pick_face(&state, &meshes, &ray) {
                if multi && sel.selected_faces.contains(&fid) {
                    sel.selected_faces.remove(&fid);
                    if let Some(face) = state.faces.iter().find(|f| f.id == fid) {
                        if let Some(mat) = materials.get_mut(&face.material_handle) {
                            mat.emissive = Color::BLACK.into();
                        }
                    }
                } else {
                    sel.selected_faces.insert(fid);
                    if let Some(face) = state.faces.iter().find(|f| f.id == fid) {
                        if let Some(mat) = materials.get_mut(&face.material_handle) {
                            mat.emissive = Color::srgb(1.0, 0.9, 0.2).into();
                        }
                    }
                }
            }
        }
        SelectionMode::Edge => {
            if let Some((a, b)) = pick_edge(&state, &ray) {
                if multi {
                    // Toggle if already selected
                    if let Some(idx) = sel
                        .selected_edges
                        .iter()
                        .position(|&(pa, pb)| (pa - a).length() < 1e-6 && (pb - b).length() < 1e-6)
                    {
                        sel.selected_edges.remove(idx);
                    } else {
                        sel.selected_edges.push((a, b));
                    }
                } else {
                    sel.selected_edges.push((a, b));
                }
            }
        }
        SelectionMode::Point => {
            if let Some(p) = pick_point(&state, &meshes, &ray) {
                if multi {
                    if let Some(idx) = sel
                        .selected_points
                        .iter()
                        .position(|&pp| (pp - p).length() < 1e-6)
                    {
                        sel.selected_points.remove(idx);
                    } else {
                        sel.selected_points.push(p);
                    }
                } else {
                    sel.selected_points.push(p);
                }
            }
        }
    }
}

fn pick_face(
    state: &ViewerState,
    meshes: &Assets<Mesh>,
    ray: &bevy::prelude::Ray3d,
) -> Option<(usize, f32)> {
    let mut best: Option<(usize, f32)> = None;
    for face in &state.faces {
        if !face.visible {
            continue;
        }
        let Some(mesh) = meshes.get(&face.mesh_handle) else { continue; };
        let Some(positions) = mesh.attribute(Mesh::ATTRIBUTE_POSITION).and_then(|a| a.as_float3()) else { continue; };
        // Triangle list; step by 3
        for tri in positions.chunks_exact(3) {
            let v0 = Vec3::from_array(tri[0]);
            let v1 = Vec3::from_array(tri[1]);
            let v2 = Vec3::from_array(tri[2]);
            if let Some(t) = ray_triangle(ray.origin, ray.direction.into(), v0, v1, v2) {
                match best {
                    Some((_, bt)) if t >= bt => {}
                    _ => best = Some((face.id, t)),
                }
            }
        }
    }
    best
}

fn pick_edge(state: &ViewerState, ray: &bevy::prelude::Ray3d) -> Option<(Vec3, Vec3)> {
    let Some(scene) = &state.scene_data else { return None; };
    let mut best: Option<(Vec3, Vec3, f32)> = None;
    let (_, spacing) = compute_grid_params(state);
    let thresh = spacing * 0.25;
    let center = state.scene_center;
    let scale = state.scene_scale;

    for shell in &scene.shells {
        for &(p0_arr, p1_arr) in &shell.edges {
            let p0r = Vec3::new(p0_arr[0] as f32, p0_arr[1] as f32, p0_arr[2] as f32);
            let p1r = Vec3::new(p1_arr[0] as f32, p1_arr[1] as f32, p1_arr[2] as f32);
            let p0 = (p0r - center) * scale;
            let p1 = (p1r - center) * scale;
            let dist = ray_segment_distance(ray.origin, ray.direction.into(), p0, p1);
            if dist <= thresh {
                let t = dist; // use distance as tie-breaker
                match best {
                    Some((_, _, bd)) if t >= bd => {}
                    _ => best = Some((p0, p1, t)),
                }
            }
        }
    }
    best.map(|(a, b, _)| (a, b))
}

fn pick_point(state: &ViewerState, meshes: &Assets<Mesh>, ray: &bevy::prelude::Ray3d) -> Option<Vec3> {
    let mut best: Option<(Vec3, f32)> = None;
    let (_, spacing) = compute_grid_params(state);
    let thresh = spacing * 0.15;
    for face in &state.faces {
        if !face.visible { continue; }
        let Some(mesh) = meshes.get(&face.mesh_handle) else { continue; };
        let Some(positions) = mesh.attribute(Mesh::ATTRIBUTE_POSITION).and_then(|a| a.as_float3()) else { continue; };
        for &p in positions.iter() {
            let v = Vec3::from_array(p);
            let dist = point_ray_distance(v, ray.origin, ray.direction.into());
            if dist <= thresh {
                match best {
                    Some((_, bd)) if dist >= bd => {}
                    _ => best = Some((v, dist)),
                }
            }
        }
    }
    best.map(|(v, _)| v)
}

fn ray_triangle(origin: Vec3, dir: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<f32> {
    let eps = 1e-6;
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let pvec = dir.cross(e2);
    let det = e1.dot(pvec);
    if det.abs() < eps { return None; }
    let inv_det = 1.0 / det;
    let tvec = origin - v0;
    let u = tvec.dot(pvec) * inv_det;
    if u < 0.0 || u > 1.0 { return None; }
    let qvec = tvec.cross(e1);
    let v = dir.dot(qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 { return None; }
    let t = e2.dot(qvec) * inv_det;
    if t > 0.0 { Some(t) } else { None }
}

fn ray_segment_distance(origin: Vec3, dir: Vec3, p0: Vec3, p1: Vec3) -> f32 {
    let l = p1 - p0;
    let a = dir.dot(dir);
    let b = dir.dot(l);
    let c = l.dot(l);
    let w0 = origin - p0;
    let d = dir.dot(w0);
    let e = l.dot(w0);
    let denom = a * c - b * b;
    let mut s = if denom.abs() > 1e-6 { (a * e - b * d) / denom } else { 0.0 };
    s = s.clamp(0.0, 1.0);
    let t = (b * s + d) / a;
    let t = t.max(0.0);
    let closest_ray = origin + dir * t;
    let closest_seg = p0 + l * s;
    (closest_ray - closest_seg).length()
}

fn point_ray_distance(p: Vec3, origin: Vec3, dir: Vec3) -> f32 {
    let t = (p - origin).dot(dir);
    let closest = if t < 0.0 { origin } else { origin + dir * t };
    (p - closest).length()
}

/// Compute adaptive grid extent and spacing from current bounds.
fn compute_grid_params(state: &ViewerState) -> (f32, f32) {
    // Default values when no bounds are available
    let default_extent = 1.0_f32;
    let default_spacing = 0.1_f32;

    let Some(bounds) = state.current_bounds else {
        return (default_extent, default_spacing);
    };

    // Half-extent around origin in normalized space
    let half_x = bounds.max.x.abs().max(bounds.min.x.abs());
    let half_y = bounds.max.y.abs().max(bounds.min.y.abs());
    let half_z = bounds.max.z.abs().max(bounds.min.z.abs());
    let half_extent = half_x.max(half_y).max(half_z);
    let half_extent = if half_extent > 0.0 { half_extent } else { default_extent };

    // Target ~20 lines across max dimension; choose a "nice" step (1-2-5 * 10^k)
    let target_lines = 20.0_f32;
    let raw_step = (half_extent * 2.0) / target_lines; // full extent spans [-E, +E]
    let spacing = nice_step(raw_step).max(0.001);

    // Extend grid to nearest multiple, add one spacing margin beyond bounds
    let extent = ((half_extent / spacing).ceil() + 1.0) * spacing;

    (extent, spacing)
}

fn nice_step(step: f32) -> f32 {
    if step <= 0.0 {
        return 0.1;
    }
    let exp = step.log10().floor();
    let base = 10.0_f32.powf(exp);
    let norm = step / base; // in [1,10)
    let nice = if norm < 1.5 {
        1.0
    } else if norm < 3.5 {
        2.0
    } else if norm < 7.5 {
        5.0
    } else {
        10.0
    };
    nice * base
}
