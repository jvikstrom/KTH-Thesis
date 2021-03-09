use std::time::Instant;
use crate::protos::learning::Message;
use crate::protos::hypercraft::ControlMessage;

struct Neighbour {
    logical_address: u32,
    physical_address: PhysicalAddress,
    last_heard: Instant,
}

struct Peer {
    logical_address: u32,
    physical_address: PhysicalAddress,
    neighbours_table: Vec<Neighbour>,
    transport: Transport,
}

struct PhysicalAddress {
    ip: String,
    port: i32,
}

enum Message {
    Control(ControlMessage),
    Msg(Message),
}

pub trait Transport {
    fn send(&self, data: Message, dest: PhysicalAddress) -> bool;
}

