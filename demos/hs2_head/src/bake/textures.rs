//! Texture manifest for the HS2 character materials.
//!
//! The prefab `.blend` only stores node graphs - its image slots are empty -
//! so the mapping from a shader slot to a file lives here. Every slot carries
//! the loader settings that `hs2_blender_export/importer.py` applies through
//! `set_tex(...)`:
//!
//! * `set_tex(..., csp=None)` -> sRGB decode,
//! * `set_tex(..., csp='Non-Color')` -> linear, no decode,
//! * Blender's image extension mode (`REPEAT`/`EXTEND`/`CLIP`) -> sampler
//!   address mode (`Repeat`/`ClampToEdge`/`ClampToBorder`).
//!
//! Slots whose file is not available yet are declared with
//! [`TextureSlot::missing`] and resolve to a 1x1 placeholder, so baking runs
//! before the art is delivered instead of producing black or transparent
//! output.

// The manifest is consumed by the bake recipes implemented in the following
// steps; silence dead-code warnings until every slot is wired up.
#![allow(dead_code)]

use bevy::{
    asset::RenderAssetUsages,
    image::{
        ImageAddressMode, ImageLoaderSettings, ImageSampler, ImageSamplerBorderColor,
        ImageSamplerDescriptor,
    },
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

/// Blender image extension modes used by the material graphs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AddressMode {
    /// `REPEAT`: tile the image.
    #[default]
    Repeat,
    /// `EXTEND`: clamp to the border texel.
    ClampEdge,
    /// `CLIP`: clamp to a transparent border.
    ClampBorder,
}

/// Image Texture node interpolation, independent of the image's color space.
///
/// Cubic is Blender's B-spline filter, evaluated by the recipe shader rather
/// than by a hardware sampler. Currently only the skin eyebrow path implements
/// it; do not assign Cubic to other slots without adding shader support.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum TextureInterpolation {
    Nearest = 0,
    Linear = 1,
    Cubic = 2,
}

/// Stand-in used while a slot has no file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placeholder {
    /// Opaque white; albedo and detail slots.
    White,
    /// Flat tangent-space normal `(0.5, 0.5, 1.0)`.
    FlatNormal,
    /// Opaque white; gloss maps that are read through their alpha channel.
    GlossNeutral,
    /// Transparent black; optional layers such as subsurface albedo.
    TransparentBlack,
}

/// One image slot of a material.
#[derive(Clone, Debug)]
pub struct TextureSlot {
    /// `None` means the file is not available yet and a placeholder is used.
    pub path: Option<&'static str>,
    /// Whether the image is decoded from sRGB (Blender `csp=None`).
    pub is_srgb: bool,
    pub address: AddressMode,
    pub interpolation: TextureInterpolation,
    pub placeholder: Placeholder,
}

impl TextureSlot {
    /// sRGB image, tiled; interpolation is specified separately.
    pub const fn srgb(path: &'static str, interpolation: TextureInterpolation) -> Self {
        Self {
            path: Some(path),
            is_srgb: true,
            address: AddressMode::Repeat,
            interpolation,
            placeholder: Placeholder::White,
        }
    }

    /// Non-color image, tiled. `linear` refers to color space, not filtering.
    pub const fn linear(path: &'static str, interpolation: TextureInterpolation) -> Self {
        Self {
            path: Some(path),
            is_srgb: false,
            address: AddressMode::Repeat,
            interpolation,
            placeholder: Placeholder::White,
        }
    }

    /// Non-color image clamped to the edge (Blender `EXTEND`).
    pub const fn linear_clamped(path: &'static str, interpolation: TextureInterpolation) -> Self {
        Self {
            path: Some(path),
            is_srgb: false,
            address: AddressMode::ClampEdge,
            interpolation,
            placeholder: Placeholder::FlatNormal,
        }
    }

    /// Non-color, edge-clamped mask with a transparent-black placeholder.
    ///
    /// Masks such as the eyebrow sheet are read through one channel (the skin
    /// shader uses B as the overlay factor), so the normal-map placeholder of
    /// [`Self::linear_clamped`] would paint the whole face with the overlay
    /// color whenever the file is missing.
    pub const fn linear_clamped_mask(
        path: &'static str,
        interpolation: TextureInterpolation,
    ) -> Self {
        Self {
            path: Some(path),
            is_srgb: false,
            address: AddressMode::ClampEdge,
            interpolation,
            placeholder: Placeholder::TransparentBlack,
        }
    }

    /// Non-color image clamped to a transparent border (Blender `CLIP`).
    pub const fn linear_clipped(path: &'static str, interpolation: TextureInterpolation) -> Self {
        Self {
            path: Some(path),
            is_srgb: false,
            address: AddressMode::ClampBorder,
            interpolation,
            placeholder: Placeholder::TransparentBlack,
        }
    }

    /// A slot whose file has not been delivered yet.
    pub const fn missing(placeholder: Placeholder) -> Self {
        Self {
            path: None,
            is_srgb: false,
            address: AddressMode::Repeat,
            interpolation: TextureInterpolation::Linear,
            placeholder,
        }
    }

    /// Load the slot, or return the placeholder when the file is missing.
    pub fn load(&self, asset_server: &AssetServer, images: &mut Assets<Image>) -> Handle<Image> {
        let Some(path) = self.path else {
            return placeholder_texture(images, self.placeholder);
        };

        let is_srgb = self.is_srgb;
        let address = self.address;
        let interpolation = self.interpolation;

        asset_server
            .load_builder()
            .with_settings(move |settings: &mut ImageLoaderSettings| {
                settings.is_srgb = is_srgb;
                settings.sampler =
                    ImageSampler::Descriptor(sampler_descriptor(address, interpolation));
            })
            .load(path)
    }
}

fn sampler_descriptor(
    address: AddressMode,
    interpolation: TextureInterpolation,
) -> ImageSamplerDescriptor {
    let mut descriptor = match interpolation {
        TextureInterpolation::Linear => ImageSamplerDescriptor::linear(),
        // Cubic is NOT implemented by this nearest sampler. The skin eyebrow
        // shader uses textureLoad and explicitly sums the B-spline footprint.
        TextureInterpolation::Nearest | TextureInterpolation::Cubic => {
            ImageSamplerDescriptor::nearest()
        }
    };
    let mode = match address {
        AddressMode::Repeat => ImageAddressMode::Repeat,
        AddressMode::ClampEdge => ImageAddressMode::ClampToEdge,
        AddressMode::ClampBorder => {
            descriptor.border_color = Some(ImageSamplerBorderColor::TransparentBlack);
            ImageAddressMode::ClampToBorder
        }
    };
    descriptor.address_mode_u = mode;
    descriptor.address_mode_v = mode;
    descriptor
}

/// Create a 1x1 stand-in image.
///
/// The slot list is short (a handful of materials), so allocating one image
/// per missing slot is cheaper than threading a cache through the baker trait.
/// Values are written as non-sRGB: `255` decodes to `1.0` either way, and the
/// flat normal needs the raw `0.5`.
fn placeholder_texture(images: &mut Assets<Image>, kind: Placeholder) -> Handle<Image> {
    let pixel: [u8; 4] = match kind {
        Placeholder::White => [255, 255, 255, 255],
        Placeholder::FlatNormal => [128, 128, 255, 255],
        Placeholder::GlossNeutral => [255, 255, 255, 255],
        Placeholder::TransparentBlack => [0, 0, 0, 0],
    };

    images.add(Image::new(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        pixel.to_vec(),
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    ))
}

/// Image slots of the skin shader shared by `Head_` and `Torso_`.
pub struct SkinTextureSet {
    pub main: TextureSlot,
    pub detail: TextureSlot,
    pub detail_gloss: TextureSlot,
    pub eyebrow: TextureSlot,
    pub bump: TextureSlot,
    pub bump2: TextureSlot,
    pub subsurface: TextureSlot,
}

/// `Head_`: everything wired in `load_textures()` for `skin_head`.
pub const HEAD_SKIN: SkinTextureSet = SkinTextureSet {
    main: TextureSlot::srgb(
        "materials/head/cf_m_skin_head_01_MainTex.png",
        TextureInterpolation::Linear,
    ),
    detail: TextureSlot::linear(
        "materials/head/cf_m_skin_head_01_DetailMainTex.png",
        TextureInterpolation::Linear,
    ),
    detail_gloss: TextureSlot::linear(
        "materials/head/cf_m_skin_head_01_DetailGlossMap.png",
        TextureInterpolation::Linear,
    ),
    eyebrow: TextureSlot::linear_clamped_mask(
        "materials/head/cf_m_skin_head_01_Texture3.png",
        TextureInterpolation::Cubic,
    ),
    bump: TextureSlot::linear_clamped(
        "materials/head/cf_m_skin_head_01_BumpMap_converted.png",
        TextureInterpolation::Linear,
    ),
    bump2: TextureSlot::linear_clamped(
        "materials/head/cf_m_skin_head_01_BumpMap2_converted.png",
        TextureInterpolation::Linear,
    ),
    // `SubsurfaceAlbedo` has no local copy; importer.py disconnects the node.
    subsurface: TextureSlot::missing(Placeholder::TransparentBlack),
};

/// `Torso_`: the same shader with the torso texture set.
pub const BODY_SKIN: SkinTextureSet = SkinTextureSet {
    main: TextureSlot::srgb(
        "materials/body/cf_m_skin_body_00_MainTex.png",
        TextureInterpolation::Linear,
    ),
    detail: TextureSlot::missing(Placeholder::White),
    detail_gloss: TextureSlot::linear(
        "materials/body/cf_m_skin_body_00_DetailGlossMap.png",
        TextureInterpolation::Linear,
    ),
    eyebrow: TextureSlot::missing(Placeholder::TransparentBlack),
    bump: TextureSlot::linear_clamped(
        "materials/body/cf_m_skin_body_00_BumpMap_converted.png",
        TextureInterpolation::Linear,
    ),
    bump2: TextureSlot::linear_clamped(
        "materials/body/cf_m_skin_body_00_BumpMap2_converted.png",
        TextureInterpolation::Linear,
    ),
    subsurface: TextureSlot::missing(Placeholder::TransparentBlack),
};

/// `hair_*`: the template only reads alpha and the green channel.
///
/// Every hair part is its own material, so each one gets its own set of slots.
/// The file names have no relation to the glTF material name (`importer.py`
/// names the textures after the *source* material), so the paths are written
/// out by hand, per material, like the skin and eye slots.
///
/// `importer.py` loads the hair `MainTex` with `csp='Non-Color'`, i.e. as
/// linear data: it only supplies the alpha mask and the green (metallic)
/// channel, the colour comes from the material constant. The normal map is
/// the `*_converted` file, exactly like the skin slots.
pub struct HairTextureSet {
    pub main: TextureSlot,
    pub bump: TextureSlot,
}

// The delivered files are de-duplicated by content: `b_03_00`, `b_03_01` and
// `s_04_00` `MainTex` share one MD5, and every `*_BumpMap_converted` except the
// accessory one is the same file. Only these four slots are ever loaded, so the
// GPU sees one copy per unique pixel set instead of one per material.
const HAIR_BACK_MAIN: TextureSlot = TextureSlot::linear(
    "materials/hair/hair_material_b_03_00_MainTex.png",
    TextureInterpolation::Linear,
);
const HAIR_BACK_BUMP: TextureSlot = TextureSlot::linear_clamped(
    "materials/hair/hair_material_b_03_00_BumpMap_converted.png",
    TextureInterpolation::Linear,
);
const HAIR_FRONT_MAIN: TextureSlot = TextureSlot::linear(
    "materials/hair/hair_material_f_13_00_MainTex.png",
    TextureInterpolation::Linear,
);
const HAIR_ACS_MAIN: TextureSlot = TextureSlot::linear(
    "materials/hair/hair_material_b_03_acs_MainTex.png",
    TextureInterpolation::Linear,
);
const HAIR_ACS_BUMP: TextureSlot = TextureSlot::linear_clamped(
    "materials/hair/hair_material_b_03_acs_BumpMap_converted.png",
    TextureInterpolation::Linear,
);

/// `hair_`: outer back-hair shell around the skull.
pub const HAIR_00: HairTextureSet = HairTextureSet {
    main: HAIR_BACK_MAIN,
    bump: HAIR_BACK_BUMP,
};

/// `hair_.001`: back-hair cap covering the whole skull.
pub const HAIR_01: HairTextureSet = HairTextureSet {
    main: HAIR_BACK_MAIN,
    bump: HAIR_BACK_BUMP,
};

/// `hair_.002`: long back-hair mass hanging behind the shoulders.
pub const HAIR_02: HairTextureSet = HairTextureSet {
    main: HAIR_BACK_MAIN,
    bump: HAIR_BACK_BUMP,
};

/// `hair_.003`: small accessory tuft at the crown.
///
/// This is the only hair set whose own `MainTex`/normal map differ from the
/// shared `b_03_00` sheet, and it is also the only one without an alpha mask
/// (256x256, fully opaque).
pub const HAIR_03: HairTextureSet = HairTextureSet {
    main: HAIR_ACS_MAIN,
    bump: HAIR_ACS_BUMP,
};

/// `hair_.004`: inner back-hair shell (same topology as `hair_`).
pub const HAIR_04: HairTextureSet = HairTextureSet {
    main: HAIR_BACK_MAIN,
    bump: HAIR_BACK_BUMP,
};

/// `hair_.005`: front bangs across the forehead.
pub const HAIR_05: HairTextureSet = HairTextureSet {
    main: HAIR_FRONT_MAIN,
    bump: HAIR_BACK_BUMP,
};

/// `hair_.006`: paired side locks beside the ears.
pub const HAIR_06: HairTextureSet = HairTextureSet {
    main: HAIR_BACK_MAIN,
    bump: HAIR_BACK_BUMP,
};

/// `Eyes_` slots, mapped from the four `Image Texture` nodes.
pub struct EyeTextureSet {
    /// `Image Texture` -> `eye/MainTex`, sclera.
    pub sclera: TextureSlot,
    /// `Image Texture.001` -> `eye/Texture2`, iris.
    pub iris: TextureSlot,
    /// `Image Texture.002` -> `eye/Texture3`, pupil (Blender `CLIP`).
    pub pupil: TextureSlot,
    /// `Image Texture.003` -> `eye/Texture4`, highlight.
    pub highlight: TextureSlot,
}

pub const EYES: EyeTextureSet = EyeTextureSet {
    sclera: TextureSlot::srgb(
        "materials/c_t_eye_white_01-DXT1.dds",
        TextureInterpolation::Linear,
    ),
    iris: TextureSlot::srgb(
        "materials/c_t_eye_00-DXT1.dds",
        TextureInterpolation::Linear,
    ),
    pupil: TextureSlot::linear_clipped(
        "materials/c_m_eye_01_Texture3.png",
        TextureInterpolation::Linear,
    ),
    highlight: TextureSlot::linear(
        "materials/c_m_eye_01_Texture4.png",
        TextureInterpolation::Linear,
    ),
};

/// `Eyelashes_`: one texture, alpha in the red channel.
pub const EYELASH: TextureSlot = TextureSlot::linear(
    "materials/c_t_eyelash_04-DXT1.dds",
    TextureInterpolation::Linear,
);

/// `Eyeshadow_`: one texture, alpha blended.
pub const EYESHADOW: TextureSlot = TextureSlot::srgb(
    "materials/c_t_eyeshadow_02-DXT5.dds",
    TextureInterpolation::Linear,
);

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::image::ImageFilterMode;

    #[test]
    fn interpolation_does_not_override_addressing() {
        for (address, expected_address) in [
            (AddressMode::Repeat, ImageAddressMode::Repeat),
            (AddressMode::ClampEdge, ImageAddressMode::ClampToEdge),
            (AddressMode::ClampBorder, ImageAddressMode::ClampToBorder),
        ] {
            for (interpolation, expected_filter) in [
                (TextureInterpolation::Nearest, ImageFilterMode::Nearest),
                (TextureInterpolation::Linear, ImageFilterMode::Linear),
                // Cubic bypasses this sampler in the eyebrow shader.
                (TextureInterpolation::Cubic, ImageFilterMode::Nearest),
            ] {
                let sampler = sampler_descriptor(address, interpolation);
                assert_eq!(sampler.address_mode_u, expected_address);
                assert_eq!(sampler.address_mode_v, expected_address);
                assert_eq!(sampler.min_filter, expected_filter);
                assert_eq!(sampler.mag_filter, expected_filter);
                assert_eq!(sampler.mipmap_filter, expected_filter);
                assert_eq!(
                    sampler.border_color,
                    (address == AddressMode::ClampBorder)
                        .then_some(ImageSamplerBorderColor::TransparentBlack),
                );
            }
        }
    }

    #[test]
    fn head_manifest_preserves_node_interpolation_and_color_space() {
        let set = HEAD_SKIN;
        assert!(set.main.is_srgb);
        assert!(!set.eyebrow.is_srgb);
        assert_eq!(set.eyebrow.interpolation, TextureInterpolation::Cubic);
        assert_eq!(set.eyebrow.address, AddressMode::ClampEdge);
        for slot in [set.main, set.detail, set.detail_gloss, set.bump, set.bump2] {
            assert_eq!(slot.interpolation, TextureInterpolation::Linear);
        }
    }
}
