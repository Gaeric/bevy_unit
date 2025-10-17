use bevy::{mesh::skinning::SkinnedMesh, prelude::*};

pub(crate) fn main() {
    App::new()
        .insert_resource(AmbientLight {
            brightness: 750.0,
            ..default()
        })
        .add_systems(Startup, setup)
        .add_systems(Update, gltf_joints)
        .add_systems(Update, joint_animation)
        .add_systems(Update, update_bone_transform);
}

#[derive(Component, Debug)]
pub struct Demo;

#[derive(Resource)]

pub struct GltfHandle {
    pub gltf_handle: Handle<Gltf>,
}

impl GltfHandle {
    pub fn new(gltf_handle: Handle<Gltf>) -> Self {
        Self { gltf_handle }
    }
}

// const GLTF_PATH: &str = "bone/bone.gltf";
const GLTF_PATH: &str = "female_base/ani-model4.gltf";

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
    ));

    commands.insert_resource(GltfHandle::new(asset_server.load(GLTF_PATH)));

    commands.spawn(SceneRoot(
        asset_server.load(GltfAssetLabel::Scene(0).from_asset(GLTF_PATH)),
    ));
}

fn joint_query(querys: Query<&Demo>) {
    for demo in &querys {
        println!("{:?}", demo)
    }
}

fn gltf_joints(gltf_handle: Res<GltfHandle>, gltf_assets: Res<Assets<Gltf>>) {
    let gltf = gltf_assets.get(&gltf_handle.gltf_handle).unwrap();
    info!("gltf nodes: {:?}", gltf.nodes);
}

fn update_bone_transform(
    mut query: Query<(&mut Transform, &Name), With<SkinnedMesh>>,
    childof: Query<&ChildOf>,
    children_query: Query<&Children>,
) {
    for (mut transform, name) in query.iter_mut() {
        println!("name is {:?}", name);
        if name.as_str() == "Bone.004" {
            println!("lalalalal")
        }
    }
}

/// The scene hierarchy currently looks somewhat like this:
///
/// ```text
/// <Parent entity>
///   + Mesh node (without `PbrBundle` or `SkinnedMesh` component)
///     + Skinned mesh entity (with `PbrBundle` and `SkinnedMesh` component, created by glTF loader)
///     + First joint
///       + Second joint
/// ```
///
/// In this example, we want to get and animate the second joint.
/// It is similar to the animation defined in `models/SimpleSkin/SimpleSkin.gltf`.
fn joint_animation(
    parent_query: Query<&ChildOf, With<SkinnedMesh>>,
    skinned_mesh: Query<&SkinnedMesh>,
    children_query: Query<&Children>,
    mut transform_query: Query<&mut Transform>,
    name_query: Query<&Name>,
) {
    // Iter skinned mesh entity
    for skinned_mesh_childof in &parent_query {
        println!("skinned_mesh_parent is {:?}", skinned_mesh_childof);
        // Mesh node is the parent of the skinned mesh entity.
        let mesh_node_entity = skinned_mesh_childof.parent();
        println!("mesh_node_entity is {:?}", mesh_node_entity);

        // Get `Children` in the mesh node.
        let mesh_node_children = children_query.get(mesh_node_entity).unwrap();
        println!("mesh_node_children leng {}", mesh_node_children.len());
        for children in mesh_node_children {
            info!("children is {:?}", children);
        }
    }

    for skin in &skinned_mesh {
        let joints = &skin.joints;
        let mut joint_name: String = String::new();

        for &joint in joints {
            if let Ok(mut transform) = transform_query.get_mut(joint) {
                transform.rotate_y(0.05);
                // info!("joint can be transform");
            }

            if let Ok(name) = name_query.get(joint) {
                joint_name = name.into();
            }

            info!("joint is {:?}, name is {}", joint, joint_name);
        }
    }
}
