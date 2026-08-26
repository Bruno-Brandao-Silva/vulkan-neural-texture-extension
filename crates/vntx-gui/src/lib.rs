//! Library interface for `vntx-gui`.

#![deny(unsafe_code)]
#![deny(missing_docs)]
#![deny(warnings)]

pub mod app;
pub mod views;

pub use app::{CompressionStatus, Tab, VntxGuiApp};
