//! Vibrational and dielectric spectra (IR, Raman, VCD, ROA) — WASM face of
//! the `molrs::compute` spectroscopy family.

use super::{SeriesOut, array1, array2, js_value};
use molrs::compute::Compute;
use molrs::compute::Fit;
use molrs::op::types::F;
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SpectrumOut {
    frequencies: Vec<F>,
    intensities: Vec<F>,
    resolution: usize,
    n_frames: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RamanSpectrumOut {
    frequencies: Vec<F>,
    isotropic: Vec<F>,
    anisotropic: Vec<F>,
    parallel: Option<Vec<F>>,
    perpendicular: Option<Vec<F>>,
    resolution: usize,
    n_frames: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DielectricSpectrumOut {
    frequencies: Vec<F>,
    eps_real: Vec<F>,
    eps_imag: Vec<F>,
}

fn spectrum_out(r: molrs::compute::SpectrumResult) -> SpectrumOut {
    SpectrumOut {
        frequencies: r.frequencies_cm1.to_vec(),
        intensities: r.intensities.to_vec(),
        resolution: r.resolution,
        n_frames: r.n_frames,
    }
}

fn raman_out(r: molrs::compute::RamanSpectrumResult) -> RamanSpectrumOut {
    RamanSpectrumOut {
        frequencies: r.frequencies_cm1.to_vec(),
        isotropic: r.isotropic.to_vec(),
        anisotropic: r.anisotropic.to_vec(),
        parallel: r.parallel.map(|v| v.to_vec()),
        perpendicular: r.perpendicular.map(|v| v.to_vec()),
        resolution: r.resolution,
        n_frames: r.n_frames,
    }
}

fn dielectric_spectrum_out(r: molrs::compute::DielectricSpectrumResult) -> DielectricSpectrumOut {
    DielectricSpectrumOut {
        frequencies: r.frequencies.to_vec(),
        eps_real: r.eps_real.to_vec(),
        eps_imag: r.eps_imag.to_vec(),
    }
}

#[wasm_bindgen(js_name = WasmIRFlux)]
pub struct WasmIRFlux {
    dt: F,
    resolution: usize,
}

#[wasm_bindgen(js_class = WasmIRFlux)]
impl WasmIRFlux {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F, resolution: usize) -> Self {
        Self { dt, resolution }
    }

    pub fn compute(&self, dipoles: &[F], n_frames: usize) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        let resolution = self.resolution;
        let dipoles = array2(dipoles, n_frames, 3, "IRFlux dipoles")?;
        let frames: [&molrs::store::Frame; 0] = [];
        let calc = molrs::compute::IRFlux;
        let r = calc
            .compute(&frames, (&dipoles, dt, resolution))
            .map_err(|e| JsValue::from_str(&format!("IRFlux: {e}")))?;
        js_value(&SeriesOut {
            lag_times: r.lag_times.to_vec(),
            values: r.acf.to_vec(),
        })
    }
}

#[wasm_bindgen(js_name = WasmRamanTensor)]
pub struct WasmRamanTensor {
    dt: F,
    resolution: usize,
}

#[wasm_bindgen(js_class = WasmRamanTensor)]
impl WasmRamanTensor {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F, resolution: usize) -> Self {
        Self { dt, resolution }
    }

    pub fn compute(&self, polarizabilities: &[F], n_frames: usize) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        let resolution = self.resolution;
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            lag_times: Vec<F>,
            acf_iso: Vec<F>,
            acf_aniso: Vec<F>,
        }
        let p = array2(
            polarizabilities,
            n_frames,
            6,
            "RamanTensor polarizabilities",
        )?;
        let frames: [&molrs::store::Frame; 0] = [];
        let calc = molrs::compute::RamanTensor;
        let r = calc
            .compute(&frames, (&p, dt, resolution))
            .map_err(|e| JsValue::from_str(&format!("RamanTensor: {e}")))?;
        js_value(&Out {
            lag_times: r.lag_times.to_vec(),
            acf_iso: r.acf_iso.to_vec(),
            acf_aniso: r.acf_aniso.to_vec(),
        })
    }
}

#[wasm_bindgen(js_name = WasmVcdCrossFlux)]
pub struct WasmVcdCrossFlux {
    dt: F,
    resolution: usize,
}

#[wasm_bindgen(js_class = WasmVcdCrossFlux)]
impl WasmVcdCrossFlux {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F, resolution: usize) -> Self {
        Self { dt, resolution }
    }

    pub fn compute(
        &self,
        electric: &[F],
        magnetic: &[F],
        n_frames: usize,
    ) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        let resolution = self.resolution;
        let e = array2(electric, n_frames, 3, "VcdCrossFlux electric")?;
        let m = array2(magnetic, n_frames, 3, "VcdCrossFlux magnetic")?;
        let frames: [&molrs::store::Frame; 0] = [];
        let calc = molrs::compute::VcdCrossFlux;
        let r = calc
            .compute(&frames, (&e, &m, dt, resolution))
            .map_err(|e| JsValue::from_str(&format!("VcdCrossFlux: {e}")))?;
        js_value(&SeriesOut {
            lag_times: r.lag_times.to_vec(),
            values: r.acf.to_vec(),
        })
    }
}

#[wasm_bindgen(js_name = WasmRoaCrossTensor)]
pub struct WasmRoaCrossTensor {
    dt: F,
    resolution: usize,
}

#[wasm_bindgen(js_class = WasmRoaCrossTensor)]
impl WasmRoaCrossTensor {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F, resolution: usize) -> Self {
        Self { dt, resolution }
    }

    pub fn compute(
        &self,
        electric_pol: &[F],
        g_tensor: &[F],
        n_frames: usize,
    ) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        let resolution = self.resolution;
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            lag_times: Vec<F>,
            acf_iso: Vec<F>,
            acf_aniso: Vec<F>,
        }
        let a = array2(electric_pol, n_frames, 6, "RoaCrossTensor electricPol")?;
        let g = array2(g_tensor, n_frames, 6, "RoaCrossTensor gTensor")?;
        let frames: [&molrs::store::Frame; 0] = [];
        let calc = molrs::compute::RoaCrossTensor;
        let r = calc
            .compute(&frames, (&a, &g, dt, resolution))
            .map_err(|e| JsValue::from_str(&format!("RoaCrossTensor: {e}")))?;
        js_value(&Out {
            lag_times: r.lag_times.to_vec(),
            acf_iso: r.acf_iso.to_vec(),
            acf_aniso: r.acf_aniso.to_vec(),
        })
    }
}

macro_rules! spectrum_fit_class {
    ($name:ident, $calc:expr) => {
        #[wasm_bindgen(js_name = $name)]
        pub struct $name {
            dt_fs: F,
        }
        #[wasm_bindgen(js_class = $name)]
        impl $name {
            /// `dt_fs` is the trajectory timestep in femtoseconds.
            #[wasm_bindgen(constructor)]
            pub fn new(dt_fs: F) -> Self {
                Self { dt_fs }
            }
            pub fn fit(&self, acf: &[F]) -> Result<JsValue, JsValue> {
                let y = array1(acf);
                let r = $calc
                    .fit((&y, self.dt_fs))
                    .map_err(|e| JsValue::from_str(&format!("spectrum fit: {e}")))?;
                js_value(&spectrum_out(r))
            }
        }
    };
}

spectrum_fit_class!(WasmPowerSpectrum, molrs::compute::PowerSpectrum);
spectrum_fit_class!(WasmIRSpectrum, molrs::compute::IRSpectrum);
spectrum_fit_class!(WasmVcdSpectrum, molrs::compute::VcdSpectrum);

#[wasm_bindgen(js_name = WasmRamanSpectrum)]
pub struct WasmRamanSpectrum {
    dt_fs: F,
    incident_frequency_cm1: F,
    temperature_k: F,
    averaged: bool,
}

#[wasm_bindgen(js_class = WasmRamanSpectrum)]
impl WasmRamanSpectrum {
    #[wasm_bindgen(constructor)]
    pub fn new(
        incident_frequency_cm1: Option<F>,
        temperature_k: Option<F>,
        averaged: Option<bool>,
        dt_fs: F,
    ) -> Self {
        Self {
            dt_fs,
            incident_frequency_cm1: incident_frequency_cm1.unwrap_or(0.0),
            temperature_k: temperature_k.unwrap_or(0.0),
            averaged: averaged.unwrap_or(false),
        }
    }

    pub fn fit(&self, acf_iso: &[F], acf_aniso: &[F]) -> Result<JsValue, JsValue> {
        let dt_fs = self.dt_fs;
        let iso = array1(acf_iso);
        let aniso = array1(acf_aniso);
        let calc = molrs::compute::RamanSpectrum {
            incident_frequency_cm1: self.incident_frequency_cm1,
            temperature_k: self.temperature_k,
            averaged: self.averaged,
        };
        let r = calc
            .fit((&iso, &aniso, dt_fs))
            .map_err(|e| JsValue::from_str(&format!("RamanSpectrum: {e}")))?;
        js_value(&raman_out(r))
    }
}

#[wasm_bindgen(js_name = WasmRoaSpectrum)]
pub struct WasmRoaSpectrum(WasmRamanSpectrum);

#[wasm_bindgen(js_class = WasmRoaSpectrum)]
impl WasmRoaSpectrum {
    /// Same optical configuration as [`WasmRamanSpectrum`]; ROA differs only in
    /// which cross-correlations are supplied to [`fit`](Self::fit).
    #[wasm_bindgen(constructor)]
    pub fn new(
        incident_frequency_cm1: Option<F>,
        temperature_k: Option<F>,
        averaged: Option<bool>,
        dt_fs: F,
    ) -> Self {
        Self(WasmRamanSpectrum::new(
            incident_frequency_cm1,
            temperature_k,
            averaged,
            dt_fs,
        ))
    }

    pub fn fit(&self, acf_iso: &[F], acf_aniso: &[F]) -> Result<JsValue, JsValue> {
        let iso = array1(acf_iso);
        let aniso = array1(acf_aniso);
        let calc = molrs::compute::RoaSpectrum {
            incident_frequency_cm1: self.0.incident_frequency_cm1,
            temperature_k: self.0.temperature_k,
            averaged: self.0.averaged,
        };
        let r = calc
            .fit((&iso, &aniso, self.0.dt_fs))
            .map_err(|e| JsValue::from_str(&format!("RoaSpectrum: {e}")))?;
        js_value(&raman_out(r))
    }
}

#[wasm_bindgen(js_name = WasmEinsteinHelfandDielectricSpectrum)]
pub struct WasmEinsteinHelfandDielectricSpectrum {
    dt: F,
    volume: F,
    temperature: F,
    epsilon_inf: F,
    zero_lag_variance: F,
}

#[wasm_bindgen(js_class = WasmEinsteinHelfandDielectricSpectrum)]
impl WasmEinsteinHelfandDielectricSpectrum {
    #[wasm_bindgen(constructor)]
    pub fn new(
        dt: F,
        volume: F,
        temperature: F,
        epsilon_inf: Option<F>,
        zero_lag_variance: F,
    ) -> Self {
        Self {
            dt,
            volume,
            temperature,
            epsilon_inf: epsilon_inf.unwrap_or(1.0),
            zero_lag_variance,
        }
    }

    pub fn fit(&self, acf: &[F]) -> Result<JsValue, JsValue> {
        let acf = array1(acf);
        let calc = molrs::compute::EinsteinHelfandSpectrum {
            dt: self.dt,
            volume: self.volume,
            temperature: self.temperature,
            epsilon_inf: self.epsilon_inf,
            zero_lag_variance: self.zero_lag_variance,
        };
        let r = calc
            .fit(&acf)
            .map_err(|e| JsValue::from_str(&format!("EinsteinHelfandDielectricSpectrum: {e}")))?;
        js_value(&dielectric_spectrum_out(r))
    }
}

#[wasm_bindgen(js_name = WasmGreenKuboDielectricSpectrum)]
pub struct WasmGreenKuboDielectricSpectrum {
    dt: F,
    volume: F,
    temperature: F,
    epsilon_inf: F,
    window_type: String,
}

#[wasm_bindgen(js_class = WasmGreenKuboDielectricSpectrum)]
impl WasmGreenKuboDielectricSpectrum {
    #[wasm_bindgen(constructor)]
    pub fn new(
        dt: F,
        volume: F,
        temperature: F,
        epsilon_inf: Option<F>,
        window_type: Option<String>,
    ) -> Self {
        Self {
            dt,
            volume,
            temperature,
            epsilon_inf: epsilon_inf.unwrap_or(1.0),
            window_type: window_type.unwrap_or_else(|| "cosine_sq".to_string()),
        }
    }

    pub fn fit(&self, jacf: &[F]) -> Result<JsValue, JsValue> {
        let jacf = array1(jacf);
        let calc = molrs::compute::GreenKuboSpectrum {
            dt: self.dt,
            volume: self.volume,
            temperature: self.temperature,
            epsilon_inf: self.epsilon_inf,
            window_type: self.window_type.clone(),
        };
        let r = calc
            .fit(&jacf)
            .map_err(|e| JsValue::from_str(&format!("GreenKuboDielectricSpectrum: {e}")))?;
        js_value(&dielectric_spectrum_out(r))
    }
}
