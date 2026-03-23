//! Zarr error conversions.

use crate::error::MolRsError;

impl From<zarrs::group::GroupCreateError> for MolRsError {
    fn from(e: zarrs::group::GroupCreateError) -> Self {
        MolRsError::zarr(e.to_string())
    }
}

impl From<zarrs::storage::StorageError> for MolRsError {
    fn from(e: zarrs::storage::StorageError) -> Self {
        MolRsError::zarr(e.to_string())
    }
}

impl From<zarrs::array::ArrayCreateError> for MolRsError {
    fn from(e: zarrs::array::ArrayCreateError) -> Self {
        MolRsError::zarr(e.to_string())
    }
}

impl From<zarrs::array::ArrayError> for MolRsError {
    fn from(e: zarrs::array::ArrayError) -> Self {
        MolRsError::zarr(e.to_string())
    }
}

impl From<zarrs::node::NodeCreateError> for MolRsError {
    fn from(e: zarrs::node::NodeCreateError) -> Self {
        MolRsError::zarr(e.to_string())
    }
}
