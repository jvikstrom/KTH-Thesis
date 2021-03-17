use std::time::Instant;
use std::vec::Vec;
use peer::peer::{Peer};
use peer::transport::PhysicalAddress;
use std::boxed::Box;

pub struct Neighbour {
    logical_address: u32,
    physical_address: PhysicalAddress,
    last_heard: Instant,
}

pub trait RDGuider {
    fn snapshot_neighbours(&self) -> &Vec<Neighbour>;
    fn send(&self, bytes: Vec<u8>, address: PhysicalAddress) -> bool;
}

struct HyperGuider {
    peer: Peer,
    logical_address: u32,
    neighbours_table: Vec<Neighbour>,
    is_hroot: bool,
}
impl HyperGuider {
    fn new(peer: Peer, logical_address: u32, neighbours_table: Vec<Neighbour>, is_hroot: bool) -> HyperGuider {
        return HyperGuider{
            peer: peer,
            logical_address: logical_address,
            neighbours_table: neighbours_table,
            is_hroot: is_hroot,
        };
    }
}

impl RDGuider for HyperGuider {
    fn snapshot_neighbours(&self) -> &Vec<Neighbour> {
        return &self.neighbours_table;
    }
    fn send(&self, bytes: Vec<u8>, address: PhysicalAddress) -> bool {
        // Try sending until the node leaves the cube or we can't send anymore.
        return false;
    }
}
pub enum JoinType {
    Existing(Vec<PhysicalAddress>),
    Create,
}

pub fn join(peer: Peer, join_type: JoinType) -> Box<dyn RDGuider> {
    match join_type {
        JoinType::Existing(addresses) => {},
        JoinType::Create => {},
    }
    return Box::new(HyperGuider::new(peer, 0, vec![], false));
}

pub fn leave(peer: Peer) {
    // Leaves the current hypercube. May be called whenever. If already in a hypercube it will send leave messages. Causes the peer to stop responding to messages.
}