// fork from bevy official example

use std::sync::mpsc::{Receiver, Sender};

use bevy::prelude::{Deref, Resource};

use crossbeam_channel::{Receiver, Sender};

#[derive(Resource, Deref, Debug)]
struct MainWorldReceiver(Receiver<Vec<u8>>);

#[derive(Resource, Deref, Debug)]
struct RenderWorldSender(Sender<Vec<u8>>);
