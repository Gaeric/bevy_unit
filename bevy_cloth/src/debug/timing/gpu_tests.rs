use super::*;

#[test]
fn features_gate_each_timing_mode() {
    use Features as F;
    let mode = |features| Capabilities::new(features, u64::MAX, 1.0);
    let unsupported = mode(F::TIMESTAMP_QUERY_INSIDE_ENCODERS | F::TIMESTAMP_QUERY_INSIDE_PASSES);
    assert!(!unsupported.output && !unsupported.totals && !unsupported.detail);
    let output = mode(F::TIMESTAMP_QUERY | F::TIMESTAMP_QUERY_INSIDE_PASSES);
    assert!(output.output && !output.totals && !output.detail);
    let totals = mode(F::TIMESTAMP_QUERY | F::TIMESTAMP_QUERY_INSIDE_ENCODERS);
    assert!(totals.output && totals.totals && !totals.detail);
    assert_eq!(totals.tick_queries(2, 3, true), 2);
    assert!(!Capabilities::new(F::TIMESTAMP_QUERY, 8, 1.0).output);
}

#[test]
fn detailed_queries_use_conservative_bound_and_fall_back_to_totals() {
    let features = Features::TIMESTAMP_QUERY
        | Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
        | Features::TIMESTAMP_QUERY_INSIDE_PASSES;
    let limited = Capabilities::new(features, 30 * u64::from(QUERY_SIZE), 1.0);
    assert_eq!(limited.tick_queries(2, 3, true), 30);
    assert_eq!(limited.tick_queries(2, 3, false), 2);
    assert_eq!(limited.tick_queries(2, 4, true), 2);
    assert_eq!(limited.tick_queries(u32::MAX, u32::MAX, true), 2);
    let standard = Capabilities::new(features, u64::MAX, 1.0);
    assert_eq!(standard.capacity, QUERY_SET_MAX_QUERIES);
    assert_eq!(standard.tick_queries(1, QUERY_SET_MAX_QUERIES, true), 2);
}

#[test]
fn aggregates_repeated_stages_without_including_them_in_the_total() {
    use TimingStage::*;
    let tick = decode_tick(
        9,
        &[100, 200, 102, 110, 113, 115, 120, 124, 125, 133],
        &[
            VbdVertexSolve,
            VbdContactRefresh,
            VbdVertexSolve,
            ParticleSelfContact,
        ],
        0.125,
    )
    .unwrap();
    assert_eq!(tick.tick, 9);
    assert_eq!(tick.milliseconds, 12.5);
    assert_eq!(
        tick.stages,
        [
            StageTime {
                stage: VbdContactRefresh,
                milliseconds: 0.25,
                calls: 1
            },
            StageTime {
                stage: VbdVertexSolve,
                milliseconds: 1.5,
                calls: 2
            },
            StageTime {
                stage: ParticleSelfContact,
                milliseconds: 1.0,
                calls: 1
            },
        ]
    );
    assert!(
        decode_tick(9, &[100, 105], &[], 1.0)
            .unwrap()
            .stages
            .is_empty()
    );
    assert!(decode_tick(9, &[100, 99], &[], 1.0).is_none());
    assert!(decode_tick(9, &[100, 200, 110], &[VbdVertexSolve], 1.0).is_none());
}
