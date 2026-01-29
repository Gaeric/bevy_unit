use std::time::Duration;

use bevy::log::LogPlugin;
use bevy::prelude::*;
use bevy::time::common_conditions::on_timer;
use rand::{Rng, rng, seq::IteratorRandom};

fn main() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, LogPlugin::default()));

    app.add_systems(Startup, setup);

    app.add_systems(
        Update,
        attack_armor.run_if(on_timer(Duration::from_millis(200))),
    );

    app.run();
}

/// An entity that can take damage.
#[derive(Component, Deref, DerefMut)]
struct HitPoints(u16);

/// For damage to reach the wearer, it must exceed the armor.
#[derive(Component, Deref)]
struct Armor(u16);

#[derive(Clone, Component, EntityEvent)]
#[entity_event(propagate, auto_propagate)]
struct Attack {
    entity: Entity,
    damage: u16,
}

fn setup(mut commands: Commands) {
    commands
        .spawn((Name::new("Demo"), HitPoints(50)))
        .observe(take_damage)
        .with_children(|parent| {
            parent
                .spawn((Name::new("Helmet"), Armor(5)))
                .observe(block_attack);

            parent
                .spawn((Name::new("Socks"), Armor(10)))
                .observe(block_attack);

            parent
                .spawn((Name::new("Shirt"), Armor(15)))
                .observe(block_attack);
        });
}

/// A callback placed on [`Armor`], checking if it absorbed all the [`Attack`] damage.
fn block_attack(mut attack: On<Attack>, armor: Query<(&Armor, &Name)>) {
    let (armor, name) = armor.get(attack.entity).unwrap();
    let damage = attack.damage.saturating_sub(**armor);
    if damage > 0 {
        info!("{} damage passed through {}", damage, name);
        // The attack isn't stopped by the armor. We reduce the damage of the attack, and allow
        // it to continue on to the goblin.
        attack.damage = damage;
    } else {
        info!("{} damage blocked by {}", attack.damage, name);
        // Armor stopped the attack, the event stops here.
        attack.propagate(false);
        info!("(propagation halted early)\n");
    }
}

/// A callback on the armor wearer, triggered when a piece of armor is not able to block an attack,
/// or the wearer is attacked directly.
fn take_damage(
    attack: On<Attack>,
    mut hp: Query<(&mut HitPoints, &Name)>,
    mut commands: Commands,
    mut app_exit: MessageWriter<AppExit>,
) {
    let (mut hp, name) = hp.get_mut(attack.entity).unwrap();
    **hp = hp.saturating_sub(attack.damage);

    if **hp > 0 {
        info!("{} has {:.1} HP", name, hp.0);
    } else {
        warn!("{} has died a gruesome death", name);
        commands.entity(attack.entity).despawn();
        app_exit.write(AppExit::Success);
    }

    info!("(propagation reached root)\n");
}

/// A normal bevy system that attacks a piece of the goblin's armor on a timer.
fn attack_armor(entities: Query<Entity, With<Armor>>, mut commands: Commands) {
    let mut rng = rng();
    if let Some(entity) = entities.iter().choose(&mut rng) {
        let damage = rng.random_range(1..20);
        commands.trigger(Attack { damage, entity });
        info!("Attack for {} damage", damage);
    }
}
