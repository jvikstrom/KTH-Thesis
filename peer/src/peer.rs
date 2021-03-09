use std::vec::Vec;
use std::boxed::Box;
use crate::transport::{Transport,PhysicalAddress};

pub trait MessageHandler {
    fn handle_message(&self, bytes: Vec<u8>);
}

pub struct NopHandler {}
impl MessageHandler for NopHandler {
    fn handle_message(&self, bytes: Vec<u8>) {}
}

pub struct Peer {
    physical_address: PhysicalAddress,
    message_handler: Box<dyn MessageHandler>,
    transport: Box<dyn Transport>,
}

impl Peer {
    pub fn new(physical_address: PhysicalAddress, message_handler: Box<dyn MessageHandler>, transport: Box<dyn Transport>) -> Peer {
        return Peer{physical_address, message_handler, transport};
    }
}
