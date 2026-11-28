use super::*;

fn sample(tick: u64, milliseconds: f64) -> PeakTick {
    PeakTick {
        tick,
        milliseconds,
        stages: vec![StageTime {
            stage: TimingStage::VbdVertexSolve,
            milliseconds: milliseconds / 2.0,
            calls: tick as u32,
        }],
    }
}

#[test]
fn out_of_order_ticks_preserve_peak_details_and_mean() {
    let mut session = Session::default();
    session.select(true);
    let stamp = session.capture(session.id).unwrap();
    session.add_tick(stamp, sample(8, 2.0));
    session.add_tick(stamp, sample(6, 4.0));
    session.add_tick(stamp, sample(7, 3.0));
    assert_eq!(session.snapshot.samples, 3);
    assert_eq!(session.snapshot.mean_tick_ms, Some(3.0));
    assert_eq!(session.snapshot.peak, Some(sample(6, 4.0)));
}

#[test]
fn reset_rejects_peaks_captured_before_reset_but_keeps_their_mean() {
    let mut timings = ClothGpuTimings::default();
    timings.set_target(Some(Entity::PLACEHOLDER));
    let extracted_session = timings.session().id;
    let in_flight = timings.session().capture(extracted_session).unwrap();
    timings.session().add_tick(in_flight, sample(1, 3.0));
    timings.reset_peak();
    assert_eq!(timings.snapshot().peak, None);

    // Same extraction, but recording happens after the reset.
    let after_reset = timings.session().capture(extracted_session).unwrap();
    timings.session().add_tick(after_reset, sample(3, 1.0));
    timings.session().add_tick(in_flight, sample(2, 5.0));
    let snapshot = timings.snapshot();
    assert_eq!(snapshot.samples, 3);
    assert_eq!(snapshot.mean_tick_ms, Some(3.0));
    assert_eq!(snapshot.peak, Some(sample(3, 1.0)));
}

#[test]
fn changing_target_clears_everything_and_rejects_old_work_even_on_reselection() {
    let mut timings = ClothGpuTimings::default();
    assert_eq!(timings.snapshot().status, TimingStatus::Off);
    timings.set_target(Some(Entity::PLACEHOLDER));
    let id = timings.session().id;
    let stamp = timings.session().capture(id).unwrap();
    timings.session().add_tick(stamp, sample(1, 3.0));
    timings.session().skip_tick(id);
    timings.session().add_output(id, 1, 0.5);
    timings.set_target(Some(Entity::PLACEHOLDER));
    assert_eq!(timings.snapshot().samples, 1);
    assert_eq!(timings.snapshot().skipped, 1);
    assert_eq!(timings.snapshot().output_ms, Some(0.5));

    timings.set_target(None);
    assert_eq!(timings.snapshot(), TimingSnapshot::default());
    timings.set_target(Some(Entity::PLACEHOLDER));
    timings.session().add_tick(stamp, sample(2, 4.0));
    timings.session().skip_tick(id);
    timings.session().add_output(id, 10, 0.9);
    assert!(timings.session().capture(id).is_none());
    assert_eq!(
        timings.snapshot(),
        TimingSnapshot {
            status: TimingStatus::Pending,
            ..Default::default()
        }
    );
}

#[test]
fn output_keeps_the_latest_sampled_frame_without_affecting_ticks() {
    let mut session = Session::default();
    session.select(true);
    session.add_output(session.id, 5, 1.0);
    session.add_output(session.id, 4, 2.0);
    session.reset_peak();
    assert_eq!(session.snapshot.output_ms, Some(1.0));
    assert_eq!(session.snapshot.samples, 0);
    assert_eq!(session.snapshot.mean_tick_ms, None);
    assert_eq!(session.snapshot.peak, None);
}
