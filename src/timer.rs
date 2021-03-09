use std::time::{Duration,Instant};
use std::collections::BinaryHeap;
use std::cmp::Reverse;

type TimerEvent = fn();


pub struct Timer {
    events: BinaryHeap<(Instant, TimerEvent)>,
}

pub impl Timer {
    fn add_event(&mut self, at: Instant, event: TimerEvent) {
        self.events.push(Reverse(at, event));
    }
    fn poll(&self, time: Instant) {
        loop {
            match self.events.peek() {
                Some((at, event)) => {
                    if at >= time {
                        event();
                        self.events.pop();
                    } else {
                        return;
                    }
                },
                None => return,
            }    
        }
    }
}

pub trait Clock {
    fn now(&self) -> Instant;
    fn after(&self, duration: Duration) -> Instant;
    fn step(&self);
}

pub struct RealClock {

}

impl Clock for RealClock {

}

