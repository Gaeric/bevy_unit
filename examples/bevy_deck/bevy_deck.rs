use bevy::prelude::*;
use libdeck::core::agent::AgentId;
use libdeck::core::interface::Interface;
use libdeck::core::room::Room;
use libdeck::core::{agent::Agent, category::Mode};

use libdeck_demo::standard::{
    abilities::gen_yingzi_ability, cards::init_cards, mode_rules::gen_standard_mode,
};

use crate::gui::GuiInterface;

#[derive(Resource)]
pub struct Deck {
    room: Room,
}

mod gui;

fn main() {
    println!("hello world");

    App::new()
        .add_plugins(DefaultPlugins)
        .init_resource::<Deck>()
        .add_systems(Update, print_room_info)
        .run();
}

impl FromWorld for Deck {
    fn from_world(world: &mut World) -> Self {
        debug!("------* unknown main process start *------");
        let deck_cards = init_cards();
        let yingzi_ability = gen_yingzi_ability();
        let mode: Mode = gen_standard_mode();

        let mut agent = Agent::new(1, "user".into(), 1);
        let fakeai = Agent::new(2, "fakeai".into(), 2);

        agent.load_abilitys(vec![yingzi_ability]);

        let mut agent_ids: Vec<(AgentId, Box<dyn Interface>)> = Vec::new();
        agent_ids.push((agent.id(), Box::new(GuiInterface::default())));

        let mut room = Room::new(mode, vec![agent, fakeai], deck_cards);
        room.ready();
        debug!("{:?}", room);

        Deck { room }
    }
}

fn print_room_info(mut deck: ResMut<Deck>, input: Res<ButtonInput<KeyCode>>) {
    if input.just_pressed(KeyCode::KeyP) {
        deck.room.ready();
        info!("{:?}", deck.room)
    }
}
