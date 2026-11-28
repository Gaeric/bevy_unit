//! Read the versioned text cache produced by bake_cloth_comparison.py.

use bevy::prelude::*;

use std::{fmt::Display, fs, path::Path, str::FromStr};

#[derive(Resource)]
pub struct Reference {
    pub version: String,
    pub quality: u32,
    pub fps: u32,
    pub indices: Vec<u32>,
    pub frames: Vec<Vec<[f32; 3]>>,
}

impl Reference {
    pub fn load(path: &Path) -> Result<Self, String> {
        let data = fs::read_to_string(path).map_err(|error| error.to_string())?;
        Self::parse(&data)
    }

    fn parse(data: &str) -> Result<Self, String> {
        let mut tokens = data.split_whitespace();
        if tokens.next() != Some("cloth-visual-v1") {
            return Err("expected cloth-visual-v1 cache".into());
        }
        let version = read(&mut tokens, "Blender version")?;
        let quality = read(&mut tokens, "Quality")?;
        let fps = read(&mut tokens, "frame rate")?;
        let frame_count: usize = read(&mut tokens, "frame count")?;
        let vertex_count: u32 = read(&mut tokens, "vertex count")?;
        let triangle_count: usize = read(&mut tokens, "triangle count")?;
        let pin_count: usize = read(&mut tokens, "pin count")?;
        positive(&mut tokens, "vertex mass")?;
        positive(&mut tokens, "gravity")?;
        if fps != 60
            || quality == 0
            || frame_count < 2
            || vertex_count < 3
            || triangle_count == 0
            || pin_count == 0
            || pin_count >= vertex_count as usize
        {
            return Err("invalid frame rate, quality or scene counts".into());
        }
        let pins = (0..pin_count)
            .map(|_| read_index(&mut tokens, vertex_count, "pin"))
            .collect::<Result<Vec<_>, _>>()?;
        if pins
            .iter()
            .enumerate()
            .any(|(i, pin)| pins[..i].contains(pin))
        {
            return Err("duplicate pin index".into());
        }
        let mut indices = Vec::new();
        for _ in 0..triangle_count {
            let a = read_index(&mut tokens, vertex_count, "triangle index")?;
            let b = read_index(&mut tokens, vertex_count, "triangle index")?;
            let c = read_index(&mut tokens, vertex_count, "triangle index")?;
            if a == b || b == c || c == a {
                return Err("triangle repeats a vertex index".into());
            }
            indices.extend([a, b, c]);
        }
        let mut frames = Vec::new();
        for _ in 0..frame_count {
            let mut frame = Vec::new();
            for _ in 0..vertex_count {
                let point = [
                    read(&mut tokens, "position x")?,
                    read(&mut tokens, "position y")?,
                    read(&mut tokens, "position z")?,
                ];
                if !Vec3::from(point).is_finite() {
                    return Err("non-finite vertex in Blender cache".into());
                }
                frame.push(point);
            }
            frames.push(frame);
        }
        if tokens.next().is_some() {
            return Err("unexpected extra data at end of Blender cache".into());
        }
        Ok(Self {
            version,
            quality,
            fps,
            indices,
            frames,
        })
    }
}

fn read<T: FromStr>(tokens: &mut dyn Iterator<Item = &str>, field: &str) -> Result<T, String>
where
    T::Err: Display,
{
    tokens
        .next()
        .ok_or_else(|| format!("missing {field}"))?
        .parse()
        .map_err(|error| format!("invalid {field}: {error}"))
}

fn read_index(
    tokens: &mut dyn Iterator<Item = &str>,
    count: u32,
    field: &str,
) -> Result<u32, String> {
    let index = read(tokens, field)?;
    if index >= count {
        return Err(format!("{field} {index} is outside {count} vertices"));
    }
    Ok(index)
}

fn positive(tokens: &mut dyn Iterator<Item = &str>, field: &str) -> Result<f32, String> {
    let value: f32 = read(tokens, field)?;
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("{field} must be finite and positive"));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CACHE: &str = "cloth-visual-v1 4.5.3 40 60 2 3 1 1 0.001 9.81
        0
        0 1 2
        0 0 0  1 0 0  0 0 1
        0 0 0  1 -0.1 0  0 -0.2 1";

    #[test]
    fn loads_rest_pose_and_sample_at_tick_one() {
        let cache = Reference::parse(CACHE).unwrap();
        assert_eq!(cache.indices, [0, 1, 2]);
        assert_eq!(cache.frames[0][1], [1., 0., 0.]);
        assert_eq!(cache.frames[1][1], [1., -0.1, 0.]);
    }

    #[test]
    fn rejects_bad_indices_and_incomplete_or_non_finite_samples() {
        assert!(Reference::parse(&CACHE.replace("0 1 2", "0 1 3")).is_err());
        assert!(Reference::parse(&CACHE.replace("-0.1", "NaN")).is_err());
        assert!(Reference::parse(CACHE.rsplit_once(' ').unwrap().0).is_err());
        assert!(Reference::parse(&format!("{CACHE} 1")).is_err());
    }
}
