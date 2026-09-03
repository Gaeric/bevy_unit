use bevy::{
    light::light_consts::lux::MOONLESS_NIGHT,
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
        app.add_systems(Update, rotate_sphere);
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

    commands.spawn((
        DirectionalLight {
            illuminance: MOONLESS_NIGHT,
            ..default()
        },
        Transform::from_xyz(1.0, 1.0, 1.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn(recipe_mat);
}

// `SphereMeshBuilder::uv` generates the sphere with its poles aligned to the
// Z axis, while Bevy's scene convention uses Y as the up axis. Apply this
// fixed rotation to align the sphere with the expected world orientation.
// The original `extended_material_bindless` example includes the same
// correction as part of its animated rotation. Without it, the sphere appears
// tilted when it is stationary, even though the UV mapping itself is correct.
fn rotate_sphere(mut meshes: Query<&mut Transform, With<Mesh3d>>, time: Res<Time>) {
    for mut transform in &mut meshes {
        transform.rotation = Quat::from_euler(
            EulerRot::YXZ,
            -time.elapsed_secs(),
            std::f32::consts::FRAC_PI_2 * 3.0,
            0.0,
        );
    }
}

fn hotkey_compute_texture(
    input: Res<ButtonInput<KeyCode>>,
    mat_components: Query<(Entity, &mut RecipeMat<SphereBake>)>,
    mut request: ResMut<PendingBakeRequests<SphereBake>>,
) {
    if input.just_pressed(KeyCode::KeyR) {
        for (entity, mut recipe_mat) in mat_components {
            recipe_mat.version += 1;
            request.items.insert(entity, recipe_mat.clone());
        }
    }
}

// fn main() {
//     App::new()
//         .add_plugins(DefaultPlugins)
//         .add_plugins(SphereBakePlugin)
//         .insert_resource(ClearColor(Color::BLACK))
//         .run();
// }
