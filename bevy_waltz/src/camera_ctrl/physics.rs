use avian3d::prelude::*;

#[derive(PhysicsLayer, Debug, Default)]
pub(crate) enum CollisionLayer {
    #[default]
    Character,
    Terrain,
    CameraObstacle,
    Sensor,
}
