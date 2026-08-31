use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(
            Update,
            (
                // will panic without res
                // run_without_res,
                run_without_component,
                single_without_component,
            ),
        )
        .run();
}

#[derive(Resource, Default)]
pub struct Teach;

#[derive(Component)]
pub struct Student;

#[derive(Component)]
pub struct Book;

fn run_without_res(_teach: Res<Teach>) {
    info!("function run without teach");
}

fn run_without_component(query: Query<&Student>) {
    info!("function run without student");
}

fn single_without_component(single: Single<&Book>) {
    info!("function run without book");
}
