use bevy::prelude::*;
use bone_attachments::{
    BoneAttachmentsPlugin, relationship::AttachedTo, scene::SceneAttachmentExt,
};

const PISTOL_PATH: &str = "waltz/pistol_skeleton2.glb";

pub(super) fn plugin(app: &mut App) {
    app.add_plugins(BoneAttachmentsPlugin);
}

#[derive(Debug, Clone, Reflect, Copy, Eq, PartialEq)]
pub enum WeaponKind {
    Pistol,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, EntityEvent, Reflect)]
pub struct EquipWeapon {
    entity: Entity,
    kind: WeaponKind,
}

#[derive(Component, Debug)]
pub struct Weapon {
    kind: WeaponKind,
}

impl EquipWeapon {
    pub fn new(entity: Entity, kind: WeaponKind) -> Self {
        Self { entity, kind }
    }
}

pub fn equip_weapon(
    equip_weapon: On<EquipWeapon>,
    mut commands: Commands,
    equiped_weapon: Query<(&Weapon, Entity), With<AttachedTo>>,
    asset_server: Res<AssetServer>,
) {
    let attachment_scene = asset_server.load(GltfAssetLabel::Scene(0).from_asset(PISTOL_PATH));

    for (weapon, entity) in equiped_weapon {
        if weapon.kind == equip_weapon.event().kind {
            info!("already equip weapon {:?}", weapon.kind);
            return;
        }
        commands.entity(entity).despawn();
    }

    commands
        .entity(equip_weapon.entity)
        .attach_scene_with_extras(
            attachment_scene,
            Weapon {
                kind: WeaponKind::Pistol,
            },
        );
}
