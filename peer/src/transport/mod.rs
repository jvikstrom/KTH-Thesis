pub mod udp;

pub struct PhysicalAddress {
    ip: String,
    port: i32,
}
impl PhysicalAddress {
    pub fn new(ip: String, port: i32) -> PhysicalAddress {
        PhysicalAddress{ip, port}
    }
}

pub trait Transport {
    fn send(&self, bytes: Vec<u8>, dest: PhysicalAddress) -> bool;
}