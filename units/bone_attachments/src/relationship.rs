//! Relationship between a host model and the models attached to it.
//!
//! Mirrors Bevy's [`Parent`]/[`Children`] hierarchy, but for "attachment"
//! semantics: a child model (e.g. a weapon, or a spawned scene root) is bound
//! to a host model (e.g. a character) without entering the standard transform
//! hierarchy.
//!
//! - [`AttachedTo`] lives on the **attachment** (the child) and points at its
//!   host — analogous to [`Parent`].
//! - [`Attachments`] lives on the **host** (the parent) and lists every
//!   attachment bound to it — analogous to [`Children`].

use alloc::vec::Vec;
use bevy::{
    ecs::{component::Component, entity::Entity},
    prelude::Deref,
    reflect::Reflect,
};

/// Every attachment bound to this host model.
///
/// Lives on the **host** (parent) entity — for example a character — and lists
/// the child models that are attached to it via [`AttachedTo`].
///
/// This is the target half of the [`AttachedTo`] relationship, analogous to
/// Bevy's [`Children`](bevy::ecs::hierarchy::Children).
#[derive(Debug, Component, Reflect)]
#[relationship_target(relationship = AttachedTo)]
pub struct Attachments(Vec<Entity>);

/// The host model this attachment is bound to.
///
/// Lives on the **attachment** (child) entity — for example a weapon or a
/// spawned scene root. The stored [`Entity`] is the **host** (parent) that this
/// model is attached to.
///
/// This is the source half of the [`Attachments`] relationship, analogous to
/// Bevy's [`Parent`](bevy::ecs::hierarchy::Parent).
#[derive(Debug, Component, Reflect, Deref)]
#[relationship(relationship_target = Attachments)]
pub struct AttachedTo(Entity);

impl From<Entity> for AttachedTo {
    fn from(entity: Entity) -> Self {
        Self(entity)
    }
}
