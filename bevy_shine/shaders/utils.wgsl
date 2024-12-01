fn is_nan(val: f32) -> bool {
    return !(0.0 < val || val < 0.0 || val == 0.0);
}
