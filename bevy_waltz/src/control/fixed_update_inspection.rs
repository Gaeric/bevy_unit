/// copy from foxtrot fixed_update_inspection
use bevy::prelude::*;

#[derive(Resource, Reflect, Default, Debug, Deref, DerefMut)]
#[reflect(Resource)]
pub struct FixedUpdateHappen(bool);

pub(super) fn plugin(app: &mut App) {
    app.init_resource::<FixedUpdateHappen>();

    app.register_type::<FixedUpdateHappen>();
    app.add_systems(PreUpdate, reset_fixed_update_happen);
    app.add_systems(FixedFirst, set_fixed_update_happen);
}

fn reset_fixed_update_happen(mut fixed_update_happen: ResMut<FixedUpdateHappen>) {
    **fixed_update_happen = false
}

fn set_fixed_update_happen(mut fixed_update_happen: ResMut<FixedUpdateHappen>) {
    **fixed_update_happen = true
}

pub fn did_fixed_update_happen(did_fixed_update_happen: Res<FixedUpdateHappen>) -> bool {
    **did_fixed_update_happen
}
