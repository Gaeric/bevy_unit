use std::marker::PhantomData;

use bevy::ecs::entity::Entity;
use bevy::ecs::system::SystemParam;
use bevy_tnua::{
    math::Vector3,
    spatial_ext::{TnuaPointProjectionResult, TnuaSpatialExt},
};
use bevy_tnua_avian3d::TnuaSpatialExtAvian3d;

#[derive(SystemParam)]
pub struct SpatialExtFacade<'w, 's> {
    for_avian3d: TnuaSpatialExtAvian3d<'w, 's>,
    _phantom: PhantomData<(&'w (), &'s ())>,
}

pub struct ColliderDataFacade<'a, 'w, 's>
where
    Self: 'a,
{
    for_avian3d: <TnuaSpatialExtAvian3d<'w, 's> as TnuaSpatialExt>::ColliderData<'a>,
    _phantom: PhantomData<(&'a (), &'w (), &'s ())>,
}

impl<'w, 's> TnuaSpatialExt for SpatialExtFacade<'w, 's> {
    type ColliderData<'a>
        = ColliderDataFacade<'a, 'w, 's>
    where
        Self: 'a;

    fn fetch_collider_data(&self, entity: Entity) -> Option<Self::ColliderData<'_>> {
        Some(ColliderDataFacade {
            for_avian3d: self.for_avian3d.fetch_collider_data(entity)?,
            _phantom: PhantomData,
        })
    }

    fn project_point(
        &'_ self,
        point: Vector3,
        solid: bool,
        collider_data: &Self::ColliderData<'_>,
    ) -> TnuaPointProjectionResult {
        return self
            .for_avian3d
            .project_point(point, solid, &collider_data.for_avian3d);
    }

    fn cast_ray(
        &self,
        origin: Vector3,
        direction: Vector3,
        max_time_of_impact: bevy_tnua::math::Float,
        collider_data: &Self::ColliderData<'_>,
    ) -> Option<(bevy_tnua::math::Float, Vector3)> {
        return self.for_avian3d.cast_ray(
            origin,
            direction,
            max_time_of_impact,
            &collider_data.for_avian3d,
        );
    }

    fn can_interact(&self, entity1: Entity, entity2: Entity) -> bool {
        return self.for_avian3d.can_interact(entity1, entity2);
    }
}
