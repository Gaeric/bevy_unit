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
    pub placeholder: Placeholder,
}

impl TextureSlot {
    /// sRGB image, tiled - the usual setting for albedo maps.
    pub const fn srgb(path: &'static str) -> Self {
        Self {
            path: Some(path),
            is_srgb: true,
            address: AddressMode::Repeat,
            placeholder: Placeholder::White,
        }
    }

    /// Linear image, tiled - the usual setting for `csp='Non-Color'` maps.
    pub const fn linear(path: &'static str) -> Self {
        Self {
            path: Some(path),
            is_srgb: false,
            address: AddressMode::Repeat,
            placeholder: Placeholder::White,
        }
    }

    /// Linear image clamped to the edge (Blender `EXTEND`).
    pub const fn linear_clamped(path: &'static str) -> Self {
        Self {
            path: Some(path),
            is_srgb: false,
            address: AddressMode::ClampEdge,
            placeholder: Placeholder::FlatNormal,
        }
    }

    /// Clamped linear mask whose placeholder is transparent black.
    ///
    /// Masks such as the eyebrow sheet are read through one channel (the skin
    /// shader uses B as the overlay factor), so the normal-map placeholder of
    /// [`Self::linear_clamped`] would paint the whole face with the overlay
    /// color whenever the file is missing.
    pub const fn linear_clamped_mask(path: &'static str) -> Self {
        Self {
            path: Some(path),
            is_srgb: false,
            address: AddressMode::ClampEdge,
            placeholder: Placeholder::TransparentBlack,
        }
    }

    /// Linear image clamped to a transparent border (Blender `CLIP`).
    pub const fn linear_clipped(path: &'static str) -> Self {
        Self {
            path: Some(path),
            is_srgb: false,
            address: AddressMode::ClampBorder,
            placeholder: Placeholder::TransparentBlack,
        }
    }

    /// A slot whose file has not been delivered yet.
    pub const fn missing(placeholder: Placeholder) -> Self {
        Self {
            path: None,
            is_srgb: false,
            address: AddressMode::Repeat,
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

        asset_server
            .load_builder()
            .with_settings(move |settings: &mut ImageLoaderSettings| {
                settings.is_srgb = is_srgb;
                settings.sampler = ImageSampler::Descriptor(sampler_descriptor(address));
            })
            .load(path)
    }
}

fn sampler_descriptor(address: AddressMode) -> ImageSamplerDescriptor {
    match address {
        AddressMode::Repeat => ImageSamplerDescriptor::default(),
        AddressMode::ClampEdge => ImageSamplerDescriptor {
            address_mode_u: ImageAddressMode::ClampToEdge,
            address_mode_v: ImageAddressMode::ClampToEdge,
            ..default()
        },
        AddressMode::ClampBorder => ImageSamplerDescriptor {
            address_mode_u: ImageAddressMode::ClampToBorder,
            address_mode_v: ImageAddressMode::ClampToBorder,
            border_color: Some(ImageSamplerBorderColor::TransparentBlack),
            ..default()
        },
    }
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
    main: TextureSlot::srgb("materials/head/cf_m_skin_head_01_MainTex.png"),
    detail: TextureSlot::linear("materials/head/cf_m_skin_head_01_DetailMainTex.png"),
    detail_gloss: TextureSlot::linear("materials/head/cf_m_skin_head_01_DetailGlossMap.png"),
    eyebrow: TextureSlot::linear_clamped_mask("materials/head/cf_m_skin_head_01_Texture3.png"),
    bump: TextureSlot::linear_clamped("materials/head/cf_m_skin_head_01_BumpMap_converted.png"),
    bump2: TextureSlot::linear_clamped("materials/head/cf_m_skin_head_01_BumpMap2_converted.png"),
    // `SubsurfaceAlbedo` has no local copy; importer.py disconnects the node.
    subsurface: TextureSlot::missing(Placeholder::TransparentBlack),
};

/// `Torso_`: the same shader with the torso texture set.
pub const BODY_SKIN: SkinTextureSet = SkinTextureSet {
    main: TextureSlot::srgb("materials/body/cf_m_skin_body_00_MainTex.png"),
    detail: TextureSlot::missing(Placeholder::White),
    detail_gloss: TextureSlot::linear("materials/body/cf_m_skin_body_00_DetailGlossMap.png"),
    eyebrow: TextureSlot::missing(Placeholder::TransparentBlack),
    bump: TextureSlot::linear_clamped("materials/body/cf_m_skin_body_00_BumpMap_converted.png"),
    bump2: TextureSlot::linear_clamped("materials/body/cf_m_skin_body_00_BumpMap2_converted.png"),
    subsurface: TextureSlot::missing(Placeholder::TransparentBlack),
};

/// `hair_*`: the template only reads alpha and the green channel.
///
/// Every hair part is its own material, so each one gets its own set of slots.
/// The file names have no relation to the glTF material name (`importer.py`
/// names the textures after the *source* material), so the paths are written
/// out by hand, per material, like the skin and eye slots.
pub struct HairTextureSet {
    pub main: TextureSlot,
    pub bump: TextureSlot,
}

/// `hair_` textures. Files not delivered yet; replace each `missing` with
/// `TextureSlot::linear("materials/hair/<file>.png")` (and `linear_clamped`
/// for the normal map) once the art is available.
pub const HAIR_00: HairTextureSet = HairTextureSet {
    main: TextureSlot::missing(Placeholder::White),
    bump: TextureSlot::missing(Placeholder::FlatNormal),
};

/// `hair_.001` textures.
pub const HAIR_01: HairTextureSet = HairTextureSet {
    main: TextureSlot::missing(Placeholder::White),
    bump: TextureSlot::missing(Placeholder::FlatNormal),
};

/// `hair_.002` textures.
pub const HAIR_02: HairTextureSet = HairTextureSet {
    main: TextureSlot::missing(Placeholder::White),
    bump: TextureSlot::missing(Placeholder::FlatNormal),
};

/// `hair_.003` textures.
pub const HAIR_03: HairTextureSet = HairTextureSet {
    main: TextureSlot::missing(Placeholder::White),
    bump: TextureSlot::missing(Placeholder::FlatNormal),
};

/// `hair_.004` textures.
pub const HAIR_04: HairTextureSet = HairTextureSet {
    main: TextureSlot::missing(Placeholder::White),
    bump: TextureSlot::missing(Placeholder::FlatNormal),
};

/// `hair_.005` textures.
pub const HAIR_05: HairTextureSet = HairTextureSet {
    main: TextureSlot::missing(Placeholder::White),
    bump: TextureSlot::missing(Placeholder::FlatNormal),
};

/// `hair_.006` textures.
pub const HAIR_06: HairTextureSet = HairTextureSet {
    main: TextureSlot::missing(Placeholder::White),
    bump: TextureSlot::missing(Placeholder::FlatNormal),
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
    sclera: TextureSlot::srgb("materials/c_t_eye_white_01-DXT1.dds"),
    iris: TextureSlot::srgb("materials/c_t_eye_00-DXT1.dds"),
    pupil: TextureSlot::linear_clipped("materials/c_m_eye_01_Texture3.png"),
    highlight: TextureSlot::linear("materials/c_m_eye_01_Texture4.png"),
};

/// `Eyelashes_`: one texture, alpha in the red channel.
pub const EYELASH: TextureSlot = TextureSlot::linear("materials/c_t_eyelash_04-DXT1.dds");

/// `Eyeshadow_`: one texture, alpha blended.
pub const EYESHADOW: TextureSlot = TextureSlot::srgb("materials/c_t_eyeshadow_02-DXT5.dds");
