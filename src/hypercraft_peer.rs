use std::time::Instant;
use crate::protos::learning;
use crate::protos::hypercraft::ControlMessage;
use std::boxed::Box;

pub struct PhysicalAddress {
    ip: String,
    port: i32,
}

pub struct Neighbour {
    logical_address: u32,
    physical_address: PhysicalAddress,
    last_heard: Instant,
}

pub struct Peer {
    logical_address: u32,
    physical_address: PhysicalAddress,
    neighbours_table: Vec<Neighbour>,
    transport: Box<dyn Transport>,
    is_hroot: bool,
}

pub enum Message {
    Control(ControlMessage),
    Msg(learning::Message),
}

pub trait Transport {
    fn send(&self, msg: Message, dest: PhysicalAddress) -> bool;
}

impl Peer {
    fn handle_message(&self, msg: Message) {
        match msg {
            Message::Control(ctrl) => {

            },
            Message::Msg(lrn) => {},
        }
    }
}

