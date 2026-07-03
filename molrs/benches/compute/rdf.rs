//! `rdf` category — radial distribution function g(r).
//!
//! Two regression points: the batch [`RDF::compute`] and the streaming
//! [`RDFAccumulator`] hot path (per-frame `accumulate` + `finalize`).

use criterion::{Criterion, criterion_group};
use molrs::compute::rdf::{RDF, RDFAccumulator};
use molrs::compute::traits::Compute;

use crate::helpers;

fn bench_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("rdf/batch");
    helpers::configure(&mut group);
    let rdf = RDF::new(helpers::RDF_BINS, helpers::CUTOFF, 0.0).unwrap();
    let (frames_owned, nlists) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(rdf.compute(&frames, &nlists).unwrap());
        })
    });

    group.finish();
}

fn bench_accumulator(c: &mut Criterion) {
    let mut group = c.benchmark_group("rdf/accumulator");
    helpers::configure(&mut group);
    let (frames_owned, nlists) = helpers::reg_pool();

    group.bench_function("reg", |b| {
        b.iter(|| {
            let rdf = RDF::new(helpers::RDF_BINS, helpers::CUTOFF, 0.0).unwrap();
            let mut acc = RDFAccumulator::new(rdf);
            for (f, nl) in frames_owned.iter().zip(nlists.iter()) {
                acc.accumulate(f, nl).unwrap();
            }
            std::hint::black_box(acc.finalize().unwrap());
        })
    });

    group.finish();
}

criterion_group!(benches, bench_batch, bench_accumulator);
