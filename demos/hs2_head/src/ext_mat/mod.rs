//! CPU-side `ExtendedMaterial` route — the alternative to the GPU `bake`
//! route. Each part material lives in its own submodule and implements
//! [`MaterialConverter`], mirroring how each bake recipe implements
//! [`crate::bake::MaterialBaker`].

mod body;
mod eye;
mod eyelash;
mod eyeshadow;
mod head;

// pub use body::BodyMaterialExt;
// pub use eye::EyeMaterialExt;
// pub use eyelash::EyelashMaterialExt;
// pub use eyeshadow::EyeshadowMaterialExt;
// pub use head::HeadMaterialExt;

use std::{marker::PhantomData, sync::Arc};

use bevy::{
    asset::Asset,
    pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin},
    prelude::*,
};

use crate::{
    ext_mat::{
        body::BodyMaterialExt, eye::EyeMaterialExt, eyelash::EyelashMaterialExt,
        eyeshadow::EyeshadowMaterialExt, head::HeadMaterialExt,
    },
    mat_convert::{MaterialApplier, MaterialRegistry},
};

/// CPU-side "conversion method" front door for the ExtendedMaterial route.
///
/// The scheduler layer only calls this; each part extension implements it and
/// keeps its own asset loading/params logic private (see `EyeMaterialExt`).
pub trait MaterialConverter<E: Asset + MaterialExtension> {
    fn convert(
        base: &StandardMaterial,
        asset_server: &AssetServer,
    ) -> ExtendedMaterial<StandardMaterial, E>;
}

/// Applier bridging one part's `ExtendedMaterial` into the shared registry.
pub struct ExtendedApplier<E>(PhantomData<E>);

impl<E> MaterialApplier for ExtendedApplier<E>
where
    E: Asset + MaterialExtension + MaterialConverter<E>,
{
    fn apply(&self, entity: Entity, base: &StandardMaterial, world: &mut World) {
        let asset_server = world.resource::<AssetServer>();
        let ext_mat = E::convert(base, asset_server);

        let mut assets = world.resource_mut::<Assets<ExtendedMaterial<StandardMaterial, E>>>();
        let handle = assets.add(ext_mat);

        if let Ok(mut e) = world.get_entity_mut(entity) {
            info!("insert new mat handle");
            e.remove::<MeshMaterial3d<StandardMaterial>>();
            e.insert(MeshMaterial3d(handle));
        }
    }
}

/// Register one part's `ExtendedMaterial` under its glTF material name.
pub fn register_ext<E>(registry: &mut MaterialRegistry, name: &str)
where
    E: Asset + MaterialExtension + MaterialConverter<E>,
{
    registry.register(name, Arc::new(ExtendedApplier::<E>(PhantomData)));
}

/// CPU `ExtendedMaterial` conversion route. Add it to the app (after
/// `MatConvertPlugin`) to switch conversion from the bake route.
pub struct ExtMatPlugin;

impl Plugin for ExtMatPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            MaterialPlugin::<ExtendedMaterial<StandardMaterial, EyeMaterialExt>>::default(),
            MaterialPlugin::<ExtendedMaterial<StandardMaterial, EyelashMaterialExt>>::default(),
            MaterialPlugin::<ExtendedMaterial<StandardMaterial, EyeshadowMaterialExt>>::default(),
            MaterialPlugin::<ExtendedMaterial<StandardMaterial, HeadMaterialExt>>::default(),
            MaterialPlugin::<ExtendedMaterial<StandardMaterial, BodyMaterialExt>>::default(),
        ));

        app.add_systems(Startup, |mut registry: ResMut<MaterialRegistry>| {
            register_ext::<EyeMaterialExt>(&mut registry, "Eyes_");
            register_ext::<EyelashMaterialExt>(&mut registry, "Eyelashes_");
            register_ext::<EyeshadowMaterialExt>(&mut registry, "Eyeshadow_");
            register_ext::<HeadMaterialExt>(&mut registry, "Head_");
            register_ext::<BodyMaterialExt>(&mut registry, "Torso_");
        });
    }
}
