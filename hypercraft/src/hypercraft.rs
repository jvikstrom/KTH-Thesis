use std::vec::Vec;
use peer::peer::{Peer};
use peer::transport::PhysicalAddress;

pub enum JoinType {
    Existing(Vec<PhysicalAddress>),
    Create,
}

pub fn join(peer: Peer, join_type: JoinType) {
    match join_type {
        JoinType::Existing(addresses) => {},
        JoinType::Create => {},
    }
}

pub fn leave(peer: Peer) {
    // Leaves the current hypercube. May be called whenever. If already in a hypercube it will send leave messages. Causes the peer to stop responding to messages.
}