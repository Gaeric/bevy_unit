mod bending;
mod data;
mod mesh;
mod systems;

pub(crate) use data::{ColoredGsSchedule, ConstraintColorRange, PreparedCloth};
pub(crate) use systems::prepare_added_cloths;
