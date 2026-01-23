use leptos::prelude::*;
use log::{Log, Metadata, Record};
use std::sync::atomic::{AtomicU32, Ordering};

pub struct MessagesLogger {
    writer: WriteSignal<Vec<(u32, String)>>,
    id: AtomicU32,
}

impl MessagesLogger {
    pub fn new(writer: WriteSignal<Vec<(u32, String)>>) -> Self {
        Self {
            writer,
            id: AtomicU32::new(0),
        }
    }
}

impl Log for MessagesLogger {
    fn enabled(&self, _metadata: &Metadata) -> bool {
        true
    }

    fn log(&self, record: &Record) {
        self.writer.update(|v| {
            v.push((
                self.id.fetch_add(1, Ordering::SeqCst),
                format!("[{}] {}", record.level(), record.args()),
            ));
        });
    }

    fn flush(&self) {}
}
