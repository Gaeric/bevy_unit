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
    owner: AgentId,
}

mod gui;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .init_resource::<Deck>()
        .init_resource::<GuiInterface>()
        .add_systems(Update, gui_interface)
        // .add_systems(PostUpdate, read_events)
        .run();
}

impl FromWorld for Deck {
    fn from_world(_world: &mut World) -> Self {
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

        Deck {
            room,
            owner: agent_ids[0].0,
            // interface: GuiInterface::default(),
            // events: VecDeque::new(),
        }
    }
}

fn gui_interface(
    mut deck: ResMut<Deck>,
    input: Res<ButtonInput<KeyCode>>,
    mut interface: ResMut<GuiInterface>,
    mut status: Local<u32>,
) {
    if input.just_pressed(KeyCode::KeyS) {
        deck.room.ready();
        deck.room.game_start();
        info!("{:?}", deck.room);
        *status |= 0b1;
    }

    if *status & 0b1 == 0 {
        return;
    }

    if input.just_pressed(KeyCode::KeyN) {
        if let Some(event) = interface.tracker.track_next(&mut deck.room) {
            info!("event is {:?}", event)
        } else {
            info!("unreadable event")
        };
    }
}
