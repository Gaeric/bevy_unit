use bevy::{color::palettes::css, ecs::system::Query, gizmos::gizmos::Gizmos};
use bevy_tnua::{TnuaObstacleRadar, math::AsF32, radar_lens::TnuaRadarLens};

use super::spatial_ext_facade::SpatialExtFacade;

pub fn character_control_radar_visualization_system(
    query: Query<&TnuaObstacleRadar>,
    spatial_ext: SpatialExtFacade,
    mut gizmos: Gizmos,
) {
    for obstacle_radar in query.iter() {
        let radar_lens = TnuaRadarLens::new(obstacle_radar, &spatial_ext);
        for blip in radar_lens.iter_blips() {
            let closest_point = blip.closest_point().get();
            gizmos.arrow(
                obstacle_radar.tracked_position(),
                closest_point.f32(),
                css::PALE_VIOLETRED,
            );
        }
    }
}
