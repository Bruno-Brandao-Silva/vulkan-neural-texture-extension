//! Local NTC cache maintenance and storage manager (`~/.cache/ntc/<app_id>/`).

use crate::error::VntxError;
use crate::format::{NtcHeader, HEADER_SIZE_BYTES};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Represents a cached `.ntc` file found on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedFile {
    /// Absolute path to the cached file.
    pub path: PathBuf,

    /// Name of the `.ntc` file (e.g. `a3f8b9c1d2e4f567.ntc`).
    pub file_name: String,

    /// Decoded 64-bit texture checksum.
    pub texture_hash: u64,

    /// Associated Steam `AppID` or game ID.
    pub app_id: u32,

    /// Total file size on disk in bytes.
    pub size_bytes: u64,

    /// Parsed and validated `NtcHeader`.
    pub header: NtcHeader,
}

/// Consolidated statistics for the local NTC cache.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CacheStats {
    /// Total number of cached `.ntc` texture assets.
    pub total_files: usize,

    /// Total disk space consumed by cache in bytes.
    pub total_size_bytes: u64,

    /// Estimated uncompressed VRAM size replaced by these assets.
    pub total_original_bytes: u64,

    /// Total VRAM bytes saved (`total_original_bytes - total_size_bytes`).
    pub estimated_saved_bytes: u64,
}

/// Cache management interface for querying, writing, and purging `.ntc` assets.
#[derive(Debug, Clone)]
pub struct CacheManager {
    root_dir: PathBuf,
}

impl CacheManager {
    /// Creates a new `CacheManager` with the specified root cache directory.
    #[must_use]
    pub fn new<P: Into<PathBuf>>(root_dir: P) -> Self {
        Self {
            root_dir: root_dir.into(),
        }
    }

    /// Returns the root cache directory path.
    #[must_use]
    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }

    /// Returns the directory path for a specific game `AppID`: `<root>/<app_id>/`.
    #[must_use]
    pub fn get_app_dir(&self, app_id: u32) -> PathBuf {
        self.root_dir.join(app_id.to_string())
    }

    /// Lists all valid `.ntc` files, optionally filtered by game `AppID`.
    ///
    /// # Errors
    ///
    /// Returns [`VntxError::CacheError`] if the directory cannot be read.
    pub fn list_cached_files(&self, app_id: Option<u32>) -> Result<Vec<CachedFile>, VntxError> {
        let mut cached_files = Vec::new();
        if !self.root_dir.exists() {
            return Ok(cached_files);
        }

        if let Some(target_app_id) = app_id {
            let app_dir = self.get_app_dir(target_app_id);
            if app_dir.exists() {
                Self::collect_app_files(target_app_id, &app_dir, &mut cached_files);
            }
        } else if let Ok(entries) = fs::read_dir(&self.root_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        if let Ok(parsed_app_id) = name.parse::<u32>() {
                            Self::collect_app_files(parsed_app_id, &path, &mut cached_files);
                        }
                    }
                }
            }
        }

        cached_files.sort_by(|a, b| a.file_name.cmp(&b.file_name));
        Ok(cached_files)
    }

    /// Computes aggregate statistics and VRAM savings for all cached files.
    ///
    /// # Errors
    ///
    /// Returns [`VntxError`] if cache scanning fails.
    pub fn calculate_total_savings(&self) -> Result<CacheStats, VntxError> {
        let files = self.list_cached_files(None)?;
        let mut stats = CacheStats {
            total_files: files.len(),
            ..Default::default()
        };

        for file in files {
            stats.total_size_bytes += file.size_bytes;
            let orig_w = u64::from(file.header.get_original_width());
            let orig_h = u64::from(file.header.get_original_height());
            let ch = u64::from(file.header.get_channels());
            let original_size = orig_w * orig_h * ch;

            stats.total_original_bytes += original_size;
        }

        stats.estimated_saved_bytes = stats
            .total_original_bytes
            .saturating_sub(stats.total_size_bytes);

        Ok(stats)
    }

    /// Purges cache files. If `purge_all` is true, deletes all cache directories.
    /// If `app_id` is provided, purges only that game's cache folder.
    ///
    /// # Returns
    ///
    /// Number of files deleted.
    ///
    /// # Errors
    ///
    /// Returns [`VntxError::CacheError`] if deletion fails.
    pub fn clean_cache(&self, app_id: Option<u32>, purge_all: bool) -> Result<usize, VntxError> {
        let mut deleted_count = 0;

        if purge_all {
            let files = self.list_cached_files(None)?;
            deleted_count = files.len();
            if self.root_dir.exists() {
                fs::remove_dir_all(&self.root_dir).map_err(|e| {
                    VntxError::CacheError(format!(
                        "Failed to wipe root cache directory {}: {}",
                        self.root_dir.display(),
                        e
                    ))
                })?;
            }
        } else if let Some(target_app_id) = app_id {
            let app_dir = self.get_app_dir(target_app_id);
            if app_dir.exists() {
                let files = self.list_cached_files(Some(target_app_id))?;
                deleted_count = files.len();
                fs::remove_dir_all(&app_dir).map_err(|e| {
                    VntxError::CacheError(format!(
                        "Failed to purge game cache directory {}: {}",
                        app_dir.display(),
                        e
                    ))
                })?;
            }
        }

        Ok(deleted_count)
    }

    /// Saves an NTC file to the appropriate cache location: `<root>/<app_id>/<hash>.ntc`.
    ///
    /// # Errors
    ///
    /// Returns [`VntxError`] if serialization or writing fails.
    pub fn save_ntc_file(
        &self,
        app_id: u32,
        header: &NtcHeader,
        weights: &[u8],
    ) -> Result<PathBuf, VntxError> {
        header.validate()?;

        let app_dir = self.get_app_dir(app_id);
        fs::create_dir_all(&app_dir).map_err(|e| {
            VntxError::CacheError(format!(
                "Failed to create app cache directory {}: {}",
                app_dir.display(),
                e
            ))
        })?;

        let hash_hex = format!("{:016x}", header.get_texture_hash());
        let file_path = app_dir.join(format!("{hash_hex}.ntc"));

        let mut file = File::create(&file_path)?;
        header.write_to(&mut file)?;
        file.write_all(weights)?;

        Ok(file_path)
    }

    fn collect_app_files(app_id: u32, dir: &Path, out_files: &mut Vec<CachedFile>) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("ntc") {
                    if let Ok(mut f) = File::open(&path) {
                        let mut header_buf = [0u8; HEADER_SIZE_BYTES];
                        if f.read_exact(&mut header_buf).is_ok() {
                            if let Ok(header) = NtcHeader::from_bytes(&header_buf) {
                                let size_bytes = path.metadata().map_or(0, |m| m.len());
                                let file_name = path
                                    .file_name()
                                    .unwrap_or_default()
                                    .to_string_lossy()
                                    .to_string();

                                out_files.push(CachedFile {
                                    path,
                                    file_name,
                                    texture_hash: header.get_texture_hash(),
                                    app_id,
                                    size_bytes,
                                    header,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
}
