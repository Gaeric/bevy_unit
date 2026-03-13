use bevy::prelude::*;
use libdeck::core::{
    agent::{Agent, AgentId},
    card::CardArea,
    event::{CardUseContent, Event, EventTracker},
    interface::Interface,
    room::Room,
    timing::{Phase, Stage, Timing},
};

use std::{collections::VecDeque, io};

#[derive(Debug, Clone, Resource)]
pub struct GuiInterface {
    pub events: VecDeque<Event>,
    pub tracker: EventTracker,
}

impl Default for GuiInterface {
    fn default() -> Self {
        Self {
            events: VecDeque::new(),
            tracker: EventTracker::new(),
        }
    }
}

/// We have two options: eigher 'Room' can initiate response requests
/// or 'Interface' can proactively excute actions based on the current status.
/// For now, we are opting for the latter, And we assume all interface are online.
impl Interface for GuiInterface {
    fn handle_event(&mut self, room: &mut Room, owner: AgentId) -> Option<Event> {
        let events = self.tracker.track(&room.analytics);
        let mut action: Option<Event>;

        // 1. first process the room/server request response event
        for (index, event) in events.iter().enumerate() {
            action = match event.content {
                _ => {
                    if room
                        .proc
                        .as_ref()
                        .is_some_and(|proc| proc.presenter() == owner)
                    {
                        debug!("interface owner {owner:?} ignore event {:?}", event.content);
                    } else {
                        trace!("interface owner {owner:?} ignore event {:?}", event.content);
                    }

                    None
                }
            };

            if action.is_some() {
                if index != events.len() - 1 {
                    panic!("there are some events mask.")
                }
                return action;
            }
        }

        // If the owner is presenter, and the timing equals 'PlayIn', then play.
        // Otherwise, if the owner is the presenter and the timing does not equal 'PlayIn', proceed to the next proc.
        // If the owner is not the presenter, skip.
        let Some(proc) = &room.proc else {
            return None;
        };

        if proc.presenter() != owner {
            return None;
        }

        let Some(timing) = proc.now() else {
            return Some(room.next_proc());
        };

        return if timing.eq(Timing::new(owner, Phase::Play, Stage::In)) {
            gui_handle_play(room, owner)
        } else {
            debug!("agent {owner:?} driver next proc");
            Some(room.next_proc())
        };
    }
}

fn gui_handle_play(room: &mut Room, presenter: AgentId) -> Option<Event> {
    let seat = room.get_agent_seat(presenter);
    let agent: &Agent = &room.agents[seat];
    let card_ids = agent.get_cards(CardArea::HandArea);

    card_ids.iter().enumerate().for_each(|(idx, _id)| {
        println!("[{idx}]: {:?}", room.get_card(card_ids[idx]).name);
    });

    println!("{} {:?} turn:", room.agents[seat].name(), room.agents[seat]);
    let mut buf = String::new();
    io::stdin().read_line(&mut buf).expect("invalid action");

    let input = buf.trim();

    if input == "e" {
        debug!(
            "{} {:?} end play phase.",
            room.agents[seat].name(),
            room.agents[seat]
        );
        return Some(room.next_proc());
    } else if let Ok(select) = input.parse::<usize>() {
        return Some(
            CardUseContent {
                agent: agent.id(),
                card: card_ids[select],
                target: agent.id(),
            }
            .into(),
        );
    }

    None
}
