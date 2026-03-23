use std::path::PathBuf;

pub fn get_test_data_path(relative_path: &str) -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let dir = PathBuf::from(manifest).join("target/tests-data");

    if !dir.exists() {
        panic!("Test data directory not found. Run: scripts/fetch-test-data.sh");
    }

    dir.join(relative_path)
}
