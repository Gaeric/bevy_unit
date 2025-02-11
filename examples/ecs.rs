use bevy::{
    prelude::*,
    render::{camera::ExtractedCamera, Render, RenderApp, RenderSet},
};

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins).add_systems(Startup, setup);

    let render_app = app.sub_app_mut(RenderApp);

    render_app.add_systems(
        Render,
        (
            generate_demo_data.in_set(RenderSet::Prepare),
            query_demo_data.in_set(RenderSet::Queue),
        ),
    );

    app.run();
}

fn setup(mut commands: Commands) {
    // Camera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..Default::default()
    });
}

#[derive(Component, Debug)]
pub struct DemoData {
    value: u32,
}

fn generate_demo_data(mut commands: Commands, camera: Query<(Entity, &ExtractedCamera)>) {
    for (entity, _camera) in &camera {
        commands.entity(entity).insert(DemoData { value: 0 });
        info!("insert demo data done");
    }
}

fn query_demo_data(query: Query<(Entity, &DemoData)>) {
    info!("query demo data");

    for (entity, demo_data) in &query {
        info!("demo data is {:?}", demo_data)
    }
}
