use leptos::prelude::*;
use log::{Log, Metadata, Record};
use std::sync::{
    Mutex,
    atomic::{AtomicU32, Ordering},
};

pub struct MessagesLogger {
    writer: WriteSignal<Vec<(u32, String)>>,
    prev: Mutex<Option<(u32, String)>>,
    id: AtomicU32,
}

impl MessagesLogger {
    pub fn new(writer: WriteSignal<Vec<(u32, String)>>) -> Self {
        Self {
            writer,
            id: AtomicU32::new(0),
            prev: Mutex::new(None),
        }
    }
}

impl Log for MessagesLogger {
    fn enabled(&self, _metadata: &Metadata) -> bool {
        true
    }

    fn log(&self, record: &Record) {
        let Ok(mut prev) = self.prev.lock() else {
            return;
        };

        let log = format!("[{}] {}", record.level(), record.args());
        let next_id = self.id.fetch_add(1, Ordering::Relaxed);
        self.writer.update(|v| {
            if let Some((ref mut prev_repeated, ref prev_log)) = *prev
                && let Some((last_id, last)) = v.last_mut()
                && *prev_log == log
            {
                *last_id = next_id;
                *prev_repeated += 1;
                *last = format!("{log} (x{})", *prev_repeated);
            } else {
                *prev = Some((1, log.clone()));
                v.push((next_id, log));
            }
        });
    }

    fn flush(&self) {}
}
