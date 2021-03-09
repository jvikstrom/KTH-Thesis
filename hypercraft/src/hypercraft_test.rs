#[cfg(test)]
mod tests {
    use crate::hypercraft::{join,JoinType};
    use peer::peer::{Peer, NopHandler};
    use peer::transport::{udp::UDP,PhysicalAddress};
    use std::boxed::Box;
    #[test]
    fn create_cube() {
        join(Peer::new(PhysicalAddress::new(String::from(""), 123), Box::new(NopHandler{}), Box::new(UDP{})), JoinType::Create)
    }
}
