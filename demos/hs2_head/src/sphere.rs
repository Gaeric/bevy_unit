use bevy::{
    mesh::{SphereKind, SphereMeshBuilder},
    prelude::*,
    render::render_resource::{AsBindGroup, TextureFormat},
    shader::ShaderRef,
};

use crate::bake::{
    BakeChannel, BakeOutputSpec, BakeRecipe, BakeRecipePlugin, PendingBakeRequests, RecipeMat,
};

const SPHERE_LABEL: &str = "sphere";
const SPHERE_BAKE_SHADER_PATH: &str = "materials/shaders/hs2_head_bake_sphere.wgsl";
const SPHERE_BAKE_TEXTURE: &str = "materials/uv_checker_bw.png";
const SIZE: UVec2 = UVec2::new(256, 256);

pub struct SphereBakePlugin;

impl Plugin for SphereBakePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(BakeRecipePlugin::<SphereBake>::default());

        app.add_systems(Startup, setup);
        // app.add_systems(Update, rotate_sphere);
        app.add_systems(Update, hotkey_compute_texture);
    }
}

#[derive(Asset, Default, Clone, Reflect, AsBindGroup)]
pub struct SphereBake {
    #[texture(60, visibility(compute))]
    #[sampler(61, visibility(compute))]
    origin_texture: Handle<Image>,

    #[storage_texture(62, image_format = Rgba8Unorm, access = ReadWrite)]
    output: Handle<Image>,
}

impl BakeRecipe for SphereBake {
    type Params = ();
    type Output = StandardMaterial;

    const LABEL: &'static str = SPHERE_LABEL;

    fn shader() -> ShaderRef {
        SPHERE_BAKE_SHADER_PATH.into()
    }

    fn entry_point() -> &'static str {
        "bake"
    }

    fn output_size() -> UVec2 {
        SIZE
    }

    fn output_specs() -> &'static [BakeOutputSpec] {
        &[BakeOutputSpec {
            channel: BakeChannel::BaseColor,
            format: TextureFormat::Rgba8Unorm,
        }]
    }

    fn new(inputs: &[Handle<Image>], outputs: &[Handle<Image>], _params: &Self::Params) -> Self {
        Self {
            origin_texture: inputs[0].clone(),
            output: outputs[0].clone(),
        }
    }

    fn material(&self, _asset_server: &AssetServer) -> StandardMaterial {
        StandardMaterial {
            base_color: Color::WHITE,
            base_color_texture: Some(self.output.clone()),
            ..default()
        }
    }
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    let texture = asset_server.load::<Image>(SPHERE_BAKE_TEXTURE);

    let (mut recipe_mat, baked) = SphereBake::bake(
        vec![texture],
        (),
        &mut images,
        &mut materials,
        &asset_server,
    );

    recipe_mat.debug_save = true;

    commands.spawn((
        Mesh3d(meshes.add(SphereMeshBuilder::new(
            1.0,
            SphereKind::Uv {
                sectors: 20,
                stacks: 20,
            },
        ))),
        MeshMaterial3d(baked.material.clone()),
        Transform::from_xyz(0.0, 0.5, 0.0),
    ));

    commands.spawn(recipe_mat);
}

fn hotkey_compute_texture(
    input: Res<ButtonInput<KeyCode>>,
    mat_components: Query<(Entity, &mut RecipeMat<SphereBake>)>,
    mut request: ResMut<PendingBakeRequests<SphereBake>>,
) {
    if input.just_pressed(KeyCode::KeyT) {
        for (entity, mut recipe_mat) in mat_components {
            recipe_mat.version += 1;
            request.items.insert(entity, recipe_mat.clone());
        }
    }
}
