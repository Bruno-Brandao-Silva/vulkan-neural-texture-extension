//! # VNTX Trainer
//!
//! Offline neural texture training pipeline and orchestrator for the **VNTX** framework.

#![deny(unsafe_code)]
#![deny(missing_docs)]
#![deny(warnings)]

pub mod orchestrator;

pub use orchestrator::{BatchTrainingSummary, TrainingOrchestrator};
