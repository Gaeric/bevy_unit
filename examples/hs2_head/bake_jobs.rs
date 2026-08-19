//! Texture preprocessing - job update side (main world).
//!
//! This module is the *producer* half of the texture baking pipeline:
//!
//! - it waits for a glTF scene to be instantiated(`WorldInstanceReady`),
//! - identifies meshes whose glTF material name maps to a [`BakePreset`]
//!   (mirroring the name prefixes `mat_convert.rs` registered),
//! - creates the output storage textures (a one-time cost, they stay alive
//!   as long as the material is used),
//! - pushes [`BakeJob`]s into [`BakeQueue`] with `version: 1` (dirty), and
//! - points the mesh's `StandardMaterial.base_color_texture` at the baked
//!   output so both the raster pipeline and the ray tracing pipeline sample
//!   the same pre-baked texture.
//!
//! Because the base data (source textures, recipe parameters) rarely changes,
//! the bake is performed **once**: the render world only dispatches the compute
//! pass when `job.version != last_baked` (see `texture_bake.rs`). After that
//! the output texture is just sampled very frame, which is exactly the point -
//! raster and ray tracing consume identical material data with zero per-frame
//! preprocessing cost.
//!
//! The compute side (pipeline + dispatch) lives in `texture_bake.rs`

use bevy::{
    asset::AssetPath,
    gltf::GltfMaterialName,
    platform::collections::HashMap,
    prelude::*,
    render::render_resource::{Extent3d, TextureFormat, TextureUsages},
    world_serialization::WorldInstanceReady,
};

use crate::texture_bake::{BakeJob, BakeKind, BakeParams, BakeQueue};

/// glTF material name prefixes handled by the baking pipeline.
/// (Kept in sync with the registrations in `mat_convert.rs`.)
const EYE_MATERIAL_PREFIX: &str = "Eyes_";
const EYELASH_MATERIAL_PREFIX: &str = "Eyelashes_";
const EYESHADOW_MATERIAL_PREFIX: &str = "Eyeshadow_";
const HEAD_MATERIAL_PREFIX: &str = "Head_";
const TORSO_MATERIAL_PREFIX: &str = "Torso_";

/// Output size used when the material does not have a base color texture yet.
const DEFAULT_BAKE_SIZE: Extent3d = Extent3d {
    width: 2048,
    height: 2048,
    depth_or_array_layers: 1,
};

/// A baked preset: which [`BakeKind`], which source textures, the fixed
/// [`BakeParams`] used for the bake, and the alpha mode to apply to the
/// resulting `StandardMaterial`.
#[derive(Clone)]
struct BakePreset {
    kind: BakeKind,
    inputs: Vec<AssetPath<'static>>,
    params: BakeParams,
    alpha_mode: Option<AlphaMode>,
}

impl BakePreset {
    /// Map a glTF material name to a preset.
    ///
    /// Materials that are not part of the baking pipeline return `None` and
    /// keep their original `StandardMaterial` untouched.
    fn from_material_name(name: &str) -> Option<Self> {
        if name.starts_with(EYE_MATERIAL_PREFIX) {
            Some(BakePreset {
                kind: BakeKind::Eye,
                inputs: vec!["materials/c_t_eye_white_01-DXT1.dss".into()],
                params: BakeParams {
                    input_count: 4,
                    iris_color: LinearRgba::from(Srgba::new(0.0, 0.0, 0.8, 1.0)).to_vec4(),
                },
                alpha_mode: None,
            })
        } else {
            None
        }
    }
}

/// Tracks which `StandardMaterial`s have already been baked, so shared
/// materials are only processed once.
#[derive(Resource, Default)]
struct BakedMaterials {
    /// `StandardMaterial` asset id -> baked output image handle.
    outputs: HashMap<AssetId<StandardMaterial>, Handle<Image>>,
}

/// Create the storage texture that the compute shader writes into.
///
/// `Rgba16Float` is linear and directly usable as
/// `StandardMaterial.base_color_texture` in both the raster and the ray
/// tracing pipelines.
fn create_bake_target(size: Extent3d) -> Image {
    let mut image =
        Image::new_target_texture(size.width, size.height, TextureFormat::Rgba16Float, None);
    // `new_target_texture` gives TEXTURE_BINDING | COPY_DST | RENDER_ATTACHMENT;
    // add STORAGE_BINDING so the compute shader can write into it.
    image.texture_descriptor.usage |= TextureUsages::STORAGE_BINDING;
    image
}

/// Main-world job producer.
///
/// Runs once per instantiated scene; pushes one [`BakeJob`] per baked material
/// and rewrites the material's `base_color_texture` to the baked output.
fn bake_materials_on_scene_load(
    scene_ready: On<WorldInstanceReady>,
    children: Query<&Children>,
    mesh_materials: Query<(&MeshMaterial3d<StandardMaterial>, &GltfMaterialName)>,
    mut asset_materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    asset_server: Res<AssetServer>,
    mut queue: ResMut<BakeQueue>,
    mut baked: ResMut<BakedMaterials>,
) {
    for descendant in children.iter_descendants(scene_ready.entity) {
        let Ok((handle, mat_name)) = mesh_materials.get(descendant) else {
            continue;
        };

        let material_id = handle.id();

        // Shared materials only need to be baked once.
        if baked.outputs.contains_key(&material_id) {
            continue;
        }

        let Some(preset) = BakePreset::from_material_name(&mat_name.0) else {
            continue;
        };

        let Some(base_mat) = asset_materials.get(material_id) else {
            continue;
        };

        // The output resolution follows the material's current base color
        // texture so the baked result replaces it 1:1.
        let size = base_mat
            .base_color_texture
            .as_ref()
            .and_then(|h| images.get(h.id()))
            .map(|img| Extent3d {
                width: img.width(),
                height: img.height(),
                ..default()
            })
            .unwrap_or(DEFAULT_BAKE_SIZE);

        let inputs: Vec<Handle<Image>> = preset
            .inputs
            .iter()
            .map(|path| asset_server.load(path.clone()))
            .collect();

        let output = images.add(create_bake_target(size));

        // New job: version 1 marks it dirty so the render world dispatches the
        // compute pass exactly once, then sits in the clean state.
        queue.jobs.push(BakeJob {
            kind: preset.kind,
            inputs,
            output: output.clone(),
            params: preset.params,
            version: 1,
        });

        baked.outputs.insert(material_id, output.clone());

        // Point the shared StandardMaterial at the baked output texture.
        let mut mat = asset_materials.get_mut(material_id).unwrap();
        mat.base_color_texture = Some(output);
        if let Some(alpha_mode) = preset.alpha_mode {
            mat.alpha_mode = alpha_mode;
        }
    }
}

/// Adds the main-world side of the baking pipeline.
pub struct BakeJobsPlugin;

impl Plugin for BakeJobsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BakedMaterials>()
            .add_observer(bake_materials_on_scene_load);
    }
}
