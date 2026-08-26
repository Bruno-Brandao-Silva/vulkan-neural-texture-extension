//! Binary file format definitions and header parsing for `.ntc` v1.0.

use crate::error::VntxError;
use std::io::{Read, Write};

/// Magic identifier bytes for NTC v1.0 ("NTC1").
pub const NTC_MAGIC: [u8; 4] = *b"NTC1";

/// Current file format version.
pub const NTC_VERSION: u32 = 1;

/// Exact fixed header size in bytes.
pub const HEADER_SIZE_BYTES: usize = 64;

/// Default byte offset from start of file to weight payload.
pub const WEIGHTS_OFFSET_DEFAULT: u64 = 64;

/// Padding size in bytes to align the header to 64 bytes.
pub const PADDING_SIZE_BYTES: usize = 16;

/// Default number of layers in standard MLP architecture.
pub const DEFAULT_LAYERS_COUNT: u16 = 3;

/// Default hidden dimension (neurons per layer).
pub const DEFAULT_HIDDEN_DIM: u16 = 64;

/// Input coordinate dimension (U, V).
pub const INPUT_DIM: u64 = 2;

/// Supported storage precision for neural network weights.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum NtcPrecision {
    /// 16-bit Floating Point (IEEE 754-2008 binary16 / FP16).
    Fp16 = 0,
    /// 8-bit Quantized Integer.
    Int8 = 1,
}

impl NtcPrecision {
    /// Returns the size in bytes for a single weight element.
    #[must_use]
    pub const fn bytes_per_element(self) -> u64 {
        match self {
            Self::Fp16 => 2,
            Self::Int8 => 1,
        }
    }

    /// Converts a raw byte value into `NtcPrecision`.
    ///
    /// # Errors
    ///
    /// Returns [`VntxError::InvalidPrecision`] if the byte does not match a valid variant.
    pub const fn from_u8(value: u8) -> Result<Self, VntxError> {
        match value {
            0 => Ok(Self::Fp16),
            1 => Ok(Self::Int8),
            _ => Err(VntxError::InvalidPrecision { precision: value }),
        }
    }
}

/// Supported color channel configurations.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum NtcChannels {
    /// 3 Channels (Red, Green, Blue).
    Rgb = 3,
    /// 4 Channels (Red, Green, Blue, Alpha).
    Rgba = 4,
}

impl NtcChannels {
    /// Converts a raw byte value into `NtcChannels`.
    ///
    /// # Errors
    ///
    /// Returns [`VntxError::InvalidChannelCount`] if the value is not 3 or 4.
    pub const fn from_u8(value: u8) -> Result<Self, VntxError> {
        match value {
            3 => Ok(Self::Rgb),
            4 => Ok(Self::Rgba),
            _ => Err(VntxError::InvalidChannelCount { channels: value }),
        }
    }
}

/// 64-byte packed binary header for `.ntc` v1.0 files.
///
/// All multi-byte integers are serialized in Little-Endian byte order.
#[repr(C, packed)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NtcHeader {
    /// Magic identifier bytes (must be `"NTC1"`).
    pub magic: [u8; 4],
    /// Format version (must be `1`).
    pub version: u32,
    /// 64-bit xxHash3 checksum of original raw texture payload.
    pub texture_hash: u64,
    /// Original uncompressed texture width in pixels.
    pub original_width: u32,
    /// Original uncompressed texture height in pixels.
    pub original_height: u32,
    /// Color channel count (3 = RGB, 4 = RGBA).
    pub channels: u8,
    /// Weight storage precision (0 = FP16, 1 = INT8).
    pub precision: u8,
    /// Total number of layers in MLP (minimum: 2).
    pub layers_count: u16,
    /// Neurons per hidden layer.
    pub hidden_dim: u16,
    /// Reserved flags for future expansion (default: 0x0000).
    pub reserved_flags: u16,
    /// Absolute byte offset from file start to raw weight payload (default: 64).
    pub weights_offset: u64,
    /// Total payload size in bytes of raw neural weights.
    pub weights_size: u64,
    /// Zero-filled padding to align header to 64-byte boundary.
    pub padding: [u8; PADDING_SIZE_BYTES],
}

impl NtcHeader {
    /// Constructs a new `NtcHeader` with the specified texture and model parameters.
    ///
    /// Automatically computes the expected `weights_size` based on the network dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`VntxError`] if any configuration parameter is invalid.
    pub fn new(
        texture_hash: u64,
        original_width: u32,
        original_height: u32,
        channels: NtcChannels,
        precision: NtcPrecision,
        layers_count: u16,
        hidden_dim: u16,
    ) -> Result<Self, VntxError> {
        let mut header = Self {
            magic: NTC_MAGIC,
            version: NTC_VERSION,
            texture_hash,
            original_width,
            original_height,
            channels: channels as u8,
            precision: precision as u8,
            layers_count,
            hidden_dim,
            reserved_flags: 0,
            weights_offset: WEIGHTS_OFFSET_DEFAULT,
            weights_size: 0,
            padding: [0u8; PADDING_SIZE_BYTES],
        };

        let calculated_size = header.calculate_expected_weights_size()?;
        header.weights_size = calculated_size;
        header.validate()?;
        Ok(header)
    }

    /// Validates the structural integrity and semantic consistency of the header.
    ///
    /// # Errors
    ///
    /// Returns a specific [`VntxError`] variant if validation fails.
    pub fn validate(&self) -> Result<(), VntxError> {
        let magic = self.magic;
        if magic != NTC_MAGIC {
            return Err(VntxError::InvalidMagic {
                expected: NTC_MAGIC,
                found: magic,
            });
        }

        let version = self.version;
        if version != NTC_VERSION {
            return Err(VntxError::UnsupportedVersion {
                version,
                supported: NTC_VERSION,
            });
        }

        let _ = NtcChannels::from_u8(self.channels)?;
        let _ = NtcPrecision::from_u8(self.precision)?;

        let layers_count = self.layers_count;
        if layers_count < 2 {
            return Err(VntxError::InvalidLayersCount { layers_count });
        }

        let hidden_dim = self.hidden_dim;
        if hidden_dim == 0 {
            return Err(VntxError::InvalidHiddenDim { hidden_dim });
        }

        let weights_offset = self.weights_offset;
        if weights_offset != WEIGHTS_OFFSET_DEFAULT {
            return Err(VntxError::WeightsOffsetMismatch {
                declared: weights_offset,
                expected: WEIGHTS_OFFSET_DEFAULT,
            });
        }

        let expected_size = self.calculate_expected_weights_size()?;
        let weights_size = self.weights_size;
        if weights_size != expected_size {
            return Err(VntxError::WeightsSizeMismatch {
                declared: weights_size,
                calculated: expected_size,
            });
        }

        Ok(())
    }

    /// Calculates the expected byte size of the weight payload based on the MLP architecture.
    ///
    /// # Errors
    ///
    /// Returns [`VntxError`] if channel count or precision are invalid.
    pub fn calculate_expected_weights_size(&self) -> Result<u64, VntxError> {
        let precision = NtcPrecision::from_u8(self.precision)?;
        let channels = u64::from(self.channels);
        let hidden = u64::from(self.hidden_dim);
        let layers = u64::from(self.layers_count);

        let layer1_elements = (INPUT_DIM * hidden) + hidden;
        let hidden_layers_count = layers.saturating_sub(2);
        let hidden_elements_per_layer = (hidden * hidden) + hidden;
        let hidden_elements = hidden_layers_count * hidden_elements_per_layer;
        let output_elements = (hidden * channels) + channels;

        let total_elements = layer1_elements + hidden_elements + output_elements;
        let bytes_per_element = precision.bytes_per_element();

        Ok(total_elements * bytes_per_element)
    }

    /// Serializes the header into a 64-byte array using Little-Endian byte order.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE_BYTES] {
        let mut buf = [0u8; HEADER_SIZE_BYTES];

        let magic = self.magic;
        let version = self.version;
        let texture_hash = self.texture_hash;
        let original_width = self.original_width;
        let original_height = self.original_height;
        let channels = self.channels;
        let precision = self.precision;
        let layers_count = self.layers_count;
        let hidden_dim = self.hidden_dim;
        let reserved_flags = self.reserved_flags;
        let weights_offset = self.weights_offset;
        let weights_size = self.weights_size;
        let padding = self.padding;

        buf[0..4].copy_from_slice(&magic);
        buf[4..8].copy_from_slice(&version.to_le_bytes());
        buf[8..16].copy_from_slice(&texture_hash.to_le_bytes());
        buf[16..20].copy_from_slice(&original_width.to_le_bytes());
        buf[20..24].copy_from_slice(&original_height.to_le_bytes());
        buf[24] = channels;
        buf[25] = precision;
        buf[26..28].copy_from_slice(&layers_count.to_le_bytes());
        buf[28..30].copy_from_slice(&hidden_dim.to_le_bytes());
        buf[30..32].copy_from_slice(&reserved_flags.to_le_bytes());
        buf[32..40].copy_from_slice(&weights_offset.to_le_bytes());
        buf[40..48].copy_from_slice(&weights_size.to_le_bytes());
        buf[48..64].copy_from_slice(&padding);

        buf
    }

    /// Deserializes and validates an `NtcHeader` from a 64-byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`VntxError::InvalidHeaderSize`] if `bytes.len() < 64`, or validation errors.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VntxError> {
        if bytes.len() < HEADER_SIZE_BYTES {
            return Err(VntxError::InvalidHeaderSize {
                expected: HEADER_SIZE_BYTES,
                size: bytes.len(),
            });
        }

        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let texture_hash = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        let original_width = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        let original_height = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
        let channels = bytes[24];
        let precision = bytes[25];
        let layers_count = u16::from_le_bytes([bytes[26], bytes[27]]);
        let hidden_dim = u16::from_le_bytes([bytes[28], bytes[29]]);
        let reserved_flags = u16::from_le_bytes([bytes[30], bytes[31]]);
        let weights_offset = u64::from_le_bytes([
            bytes[32], bytes[33], bytes[34], bytes[35], bytes[36], bytes[37], bytes[38], bytes[39],
        ]);
        let weights_size = u64::from_le_bytes([
            bytes[40], bytes[41], bytes[42], bytes[43], bytes[44], bytes[45], bytes[46], bytes[47],
        ]);

        let mut padding = [0u8; PADDING_SIZE_BYTES];
        padding.copy_from_slice(&bytes[48..64]);

        let header = Self {
            magic,
            version,
            texture_hash,
            original_width,
            original_height,
            channels,
            precision,
            layers_count,
            hidden_dim,
            reserved_flags,
            weights_offset,
            weights_size,
            padding,
        };

        header.validate()?;
        Ok(header)
    }

    /// Reads and parses an `NtcHeader` from an I/O stream.
    ///
    /// # Errors
    ///
    /// Returns [`VntxError`] on read error or header validation failure.
    pub fn from_reader<R: Read>(reader: &mut R) -> Result<Self, VntxError> {
        let mut buf = [0u8; HEADER_SIZE_BYTES];
        reader.read_exact(&mut buf)?;
        Self::from_bytes(&buf)
    }

    /// Writes the 64-byte header into an I/O stream.
    ///
    /// # Errors
    ///
    /// Returns [`VntxError`] if writing fails.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<(), VntxError> {
        self.validate()?;
        let bytes = self.to_bytes();
        writer.write_all(&bytes)?;
        Ok(())
    }

    /// Returns a copy of the magic bytes.
    #[must_use]
    pub const fn get_magic(&self) -> [u8; 4] {
        self.magic
    }

    /// Returns the format version.
    #[must_use]
    pub const fn get_version(&self) -> u32 {
        self.version
    }

    /// Returns the 64-bit texture checksum.
    #[must_use]
    pub const fn get_texture_hash(&self) -> u64 {
        self.texture_hash
    }

    /// Returns the original texture width.
    #[must_use]
    pub const fn get_original_width(&self) -> u32 {
        self.original_width
    }

    /// Returns the original texture height.
    #[must_use]
    pub const fn get_original_height(&self) -> u32 {
        self.original_height
    }

    /// Returns the channel count.
    #[must_use]
    pub const fn get_channels(&self) -> u8 {
        self.channels
    }

    /// Returns the weight precision flag.
    #[must_use]
    pub const fn get_precision(&self) -> u8 {
        self.precision
    }

    /// Returns the total layers count.
    #[must_use]
    pub const fn get_layers_count(&self) -> u16 {
        self.layers_count
    }

    /// Returns the hidden dimension.
    #[must_use]
    pub const fn get_hidden_dim(&self) -> u16 {
        self.hidden_dim
    }

    /// Returns the reserved flags.
    #[must_use]
    pub const fn get_reserved_flags(&self) -> u16 {
        self.reserved_flags
    }

    /// Returns the byte offset to weights payload.
    #[must_use]
    pub const fn get_weights_offset(&self) -> u64 {
        self.weights_offset
    }

    /// Returns the declared weights payload size in bytes.
    #[must_use]
    pub const fn get_weights_size(&self) -> u64 {
        self.weights_size
    }

    /// Returns a copy of the padding bytes.
    #[must_use]
    pub const fn get_padding(&self) -> [u8; PADDING_SIZE_BYTES] {
        self.padding
    }
}
