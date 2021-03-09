use crate::transport::{Transport,PhysicalAddress};

pub struct UDP {

}

impl Transport for UDP {
    fn send(&self, bytes: Vec<u8>, dest: PhysicalAddress) -> bool {
        return false;
    }
}
